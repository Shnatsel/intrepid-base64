//! Dedicated AVX-512 implementation based on
//! "Base64 encoding and decoding at almost the speed of a memory copy"
//! by Wojciech Muła, Daniel Lemire
//! https://arxiv.org/abs/1910.05109

use super::{
    ALPHABET, DECODE_HIGH, DECODE_LOW, DecodeError, INVALID, decode_byte, decode_tail, encode_tail,
};
use fearless_simd::{mask8x64, prelude::*, u8x16, u8x32, u8x64, u16x32};

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512i, _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_movepi8_mask,
    _mm512_multishift_epi64_epi8, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8,
    _mm512_ternarylogic_epi32,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512i, _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_movepi8_mask,
    _mm512_multishift_epi64_epi8, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8,
    _mm512_ternarylogic_epi32,
};

// AVX-512/VBMI encoding and decoding follow the algorithms described by
// Wojciech Mula and Daniel Lemire in <https://arxiv.org/abs/1910.05109>.
const ENCODE_GATHER: [u8; 64] = [
    1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10, 13, 12, 14, 13, 16, 15, 17, 16, 19, 18, 20,
    19, 22, 21, 23, 22, 25, 24, 26, 25, 28, 27, 29, 28, 31, 30, 32, 31, 34, 33, 35, 34, 37, 36, 38,
    37, 40, 39, 41, 40, 43, 42, 44, 43, 46, 45, 47, 46,
];

const ENCODE_SHIFTS: [u8; 64] = [
    10, 4, 22, 16, 42, 36, 54, 48, 10, 4, 22, 16, 42, 36, 54, 48, 10, 4, 22, 16, 42, 36, 54, 48,
    10, 4, 22, 16, 42, 36, 54, 48, 10, 4, 22, 16, 42, 36, 54, 48, 10, 4, 22, 16, 42, 36, 54, 48,
    10, 4, 22, 16, 42, 36, 54, 48, 10, 4, 22, 16, 42, 36, 54, 48,
];

const DECODE_MERGE_BYTES: [u8; 64] = [
    64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64,
    1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1, 64, 1,
    64, 1, 64, 1, 64, 1, 64, 1, 64, 1,
];

const DECODE_MERGE_WORDS: [u16; 32] = [
    4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096,
    1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1,
];

// After the multiply-adds, each decoded triple is stored as `[o2, o1, o0, 0]`.
const DECODE_COMPACT: [u8; 64] = [
    2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 18, 17, 16, 22, 21, 20, 26, 25, 24, 30, 29, 28, 34, 33,
    32, 38, 37, 36, 42, 41, 40, 46, 45, 44, 50, 49, 48, 54, 53, 52, 58, 57, 56, 62, 61, 60, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

const DECODE_LOW_INVALID_PADDING: [u8; 64] = decode_low_invalid_padding();

const fn decode_low_invalid_padding() -> [u8; 64] {
    let mut table = DECODE_LOW;
    // Expected terminal padding is replaced with `A` before the lookup, so any
    // remaining `=` is misplaced and should mark the block bad.
    table[b'=' as usize] = INVALID;
    table
}

fearless_simd::kernel!(
    pub(super) fn encode(simd: Avx512, input: &[u8], output: &mut [u8]) {
        let mut input = input;
        let mut output = output;
        let gather: __m512i = u8x64::simd_from(simd, ENCODE_GATHER).into();
        let shifts: __m512i = u8x64::simd_from(simd, ENCODE_SHIFTS).into();
        let alphabet: __m512i = u8x64::simd_from(simd, ALPHABET).into();

        while input.len() >= 48 {
            let low = u8x32::from_slice(simd, &input[..32]);
            let tail = u8x16::from_slice(simd, &input[32..48]);
            let zero = u8x16::splat(simd, 0);
            let bytes: __m512i = low.combine(tail.combine(zero)).into();
            let gathered = _mm512_permutexvar_epi8(gather, bytes);
            let sextets = _mm512_multishift_epi64_epi8(shifts, gathered);
            let encoded: u8x64<_> = _mm512_permutexvar_epi8(sextets, alphabet).simd_into(simd);
            encoded.store_slice(&mut output[..64]);

            input = &input[48..];
            output = &mut output[64..];
        }

        encode_tail(input, output);
    }
);

fearless_simd::kernel!(
    pub(super) fn decode(
        simd: Avx512,
        input: &[u8],
        output: &mut [u8],
        padding: usize,
    ) -> Result<(), DecodeError> {
        let mut input = input;
        let mut output = output;
        let original_input = input;
        let input_len = input.len();
        let low_table: __m512i = u8x64::simd_from(simd, DECODE_LOW_INVALID_PADDING).into();
        let high_table: __m512i = u8x64::simd_from(simd, DECODE_HIGH).into();
        let merge_bytes: __m512i = u8x64::simd_from(simd, DECODE_MERGE_BYTES).into();
        let merge_words: __m512i = u16x32::simd_from(simd, DECODE_MERGE_WORDS).into();
        let compact: __m512i = u8x64::simd_from(simd, DECODE_COMPACT).into();
        let mut errors: __m512i = u8x64::splat(simd, 0).into();
        let mut input_offset = 0;

        while input.len() >= 64 {
            let mut chars = u8x64::from_slice(simd, &input[..64]);

            // The inverse lookup deliberately marks `=` invalid. Replace only
            // padding positions validated by `padding_len` with sextet zero.
            if input_offset + 64 == input_len && padding != 0 {
                let padding_bits = match padding {
                    2 => (1_u64 << 62) | (1_u64 << 63),
                    1 => 1_u64 << 63,
                    _ => unreachable!(),
                };
                chars = mask8x64::from_bitmask(simd, padding_bits)
                    .select(u8x64::splat(simd, b'A'), chars);
            }

            let chars: __m512i = chars.into();
            let sextets = _mm512_permutex2var_epi8(low_table, chars, high_table);
            errors = _mm512_ternarylogic_epi32::<0xfe>(errors, chars, sextets);
            let pairs = _mm512_maddubs_epi16(sextets, merge_bytes);
            let triples = _mm512_madd_epi16(pairs, merge_words);
            let compacted: u8x64<_> = _mm512_permutexvar_epi8(compact, triples).simd_into(simd);
            let (first, second) = compacted.split();
            let (middle, _) = second.split();
            first.store_slice(&mut output[..32]);
            middle.store_slice(&mut output[32..48]);

            input = &input[64..];
            output = &mut output[48..];
            input_offset += 64;
        }

        if _mm512_movepi8_mask(errors) != 0 {
            return Err(find_decode_error(
                &original_input[..input_offset],
                input_len - padding,
            ));
        }

        decode_tail(input, output, input_offset, input_len - padding)
    }
);

fn find_decode_error(input: &[u8], padding_start: usize) -> DecodeError {
    for (index, &byte) in input.iter().enumerate() {
        if byte == b'=' {
            if index < padding_start {
                return DecodeError::InvalidPadding { index };
            }
        } else if decode_byte(byte).is_none() {
            return DecodeError::InvalidByte { index, byte };
        }
    }

    unreachable!("the AVX-512 decoder reported an error that was not present")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_impl, encode_impl, encoded_len, padding_len};
    use fearless_simd::{Level, dispatch};

    #[test]
    fn backend_matches_portable_backend() {
        let Some(avx512) = Level::new().as_avx512() else {
            return;
        };
        let baseline = Level::baseline();

        for len in 0..=257 {
            let input: Vec<u8> = (0..len)
                .map(|i| (i as u8).wrapping_mul(73).wrapping_add(29))
                .collect();
            let output_len = encoded_len(len).unwrap();
            let mut avx512_encoded = vec![0; output_len];
            let mut portable_encoded = vec![0; output_len];

            encode(avx512, &input, &mut avx512_encoded);
            dispatch!(baseline, simd => encode_impl(simd, &input, &mut portable_encoded));
            assert_eq!(avx512_encoded, portable_encoded, "encode length {len}");

            let padding = padding_len(&avx512_encoded).unwrap();
            let maximum_len = avx512_encoded.len() / 4 * 3;
            let mut avx512_decoded = vec![0; maximum_len];
            let mut portable_decoded = vec![0; maximum_len];

            decode(avx512, &avx512_encoded, &mut avx512_decoded, padding).unwrap();
            dispatch!(baseline, simd => decode_impl(
                simd,
                &portable_encoded,
                &mut portable_decoded,
                padding,
            ))
            .unwrap();
            avx512_decoded.truncate(maximum_len - padding);
            portable_decoded.truncate(maximum_len - padding);
            assert_eq!(avx512_decoded, portable_decoded, "decode length {len}");
            assert_eq!(avx512_decoded, input, "round trip length {len}");
        }
    }
}
