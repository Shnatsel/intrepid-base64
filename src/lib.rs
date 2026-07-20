//! A SIMD-accelerated implementation of standard padded Base64.
//!
//! Enable the `avx512` Cargo feature to compile the runtime-detected dedicated
//! AVX-512/VBMI implementation.

#![forbid(unsafe_code)]

use std::sync::OnceLock;

use fearless_simd::{Level, Simd, dispatch, mask8x64, prelude::*, u8x16, u8x32, u8x64, u32x16};

#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
mod avx512;

const ALPHABET: [u8; 64] = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Arrange each three-byte group as a little-endian `[b2, b1, b0, 0]` word.
const ENCODE_GATHER: [u8; 64] = [
    2, 1, 0, 255, 5, 4, 3, 255, 8, 7, 6, 255, 11, 10, 9, 255, 14, 13, 12, 255, 17, 16, 15, 255, 20,
    19, 18, 255, 23, 22, 21, 255, 26, 25, 24, 255, 29, 28, 27, 255, 32, 31, 30, 255, 35, 34, 33,
    255, 38, 37, 36, 255, 41, 40, 39, 255, 44, 43, 42, 255, 47, 46, 45, 255,
];

// Compact sixteen `[o0, o1, o2, 0]` words into 48 contiguous bytes.
const DECODE_COMPACT: [u8; 64] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33,
    34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

const INVALID: u8 = 0x80;

// ASCII 0..=63. `=` maps to zero after its placement has been validated.
const DECODE_LOW: [u8; 64] = [
    INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
    INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
    INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
    INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
    INVALID, INVALID, INVALID, 62, INVALID, INVALID, INVALID, 63, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, INVALID, INVALID, INVALID, 0, INVALID, INVALID,
];

// ASCII 64..=127.
const DECODE_HIGH: [u8; 64] = [
    INVALID, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, INVALID, INVALID,
    INVALID, INVALID, INVALID,
];

static SIMD_LEVEL: OnceLock<Level> = OnceLock::new();

/// An error returned when decoding malformed Base64.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// The encoded length is not divisible by four.
    InvalidLength { len: usize },
    /// A byte is not part of the standard Base64 alphabet.
    InvalidByte { index: usize, byte: u8 },
    /// Padding occurs anywhere other than the final one or two bytes.
    InvalidPadding { index: usize },
    /// Unused bits in the final Base64 quantum are non-zero.
    NonCanonicalTrailingBits { index: usize },
    /// The decoded size cannot be represented as a `usize`.
    LengthOverflow,
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::InvalidLength { len } => {
                write!(f, "Base64 input length {len} is not divisible by four")
            }
            Self::InvalidByte { index, byte } => {
                write!(f, "invalid Base64 byte 0x{byte:02x} at index {index}")
            }
            Self::InvalidPadding { index } => {
                write!(f, "invalid Base64 padding at index {index}")
            }
            Self::NonCanonicalTrailingBits { index } => {
                write!(f, "non-zero trailing bits at Base64 index {index}")
            }
            Self::LengthOverflow => f.write_str("decoded Base64 length overflows usize"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Return the encoded length, or `None` if it cannot be represented as a `usize`.
pub const fn encoded_len(input_len: usize) -> Option<usize> {
    match input_len.checked_add(2) {
        Some(rounded) => match (rounded / 3).checked_mul(4) {
            Some(len) => Some(len),
            None => None,
        },
        None => None,
    }
}

/// Encode bytes using the standard padded Base64 alphabet.
pub fn encode(input: &[u8]) -> String {
    let output_len = encoded_len(input.len()).expect("Base64 encoded length overflow");
    let mut output = vec![0; output_len];
    let level = *SIMD_LEVEL.get_or_init(Level::new);

    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    if let Some(avx512) = level.as_avx512() {
        avx512::encode(avx512, input, &mut output);
        return String::from_utf8(output).expect("the Base64 alphabet is valid UTF-8");
    }

    dispatch!(level, simd => encode_impl(simd, input, &mut output));

    String::from_utf8(output).expect("the Base64 alphabet is valid UTF-8")
}

/// Decode standard padded Base64.
///
/// The decoder is strict: the length must be divisible by four, padding is
/// required when the final quantum is short, and unused trailing bits must be
/// zero.
pub fn decode(input: &[u8]) -> Result<Vec<u8>, DecodeError> {
    let padding = padding_len(input)?;
    let maximum_len = (input.len() / 4)
        .checked_mul(3)
        .ok_or(DecodeError::LengthOverflow)?;
    let mut output = vec![0; maximum_len];
    let level = *SIMD_LEVEL.get_or_init(Level::new);

    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    if let Some(avx512) = level.as_avx512() {
        avx512::decode(avx512, input, &mut output, padding)?;
        validate_trailing_bits(input, padding)?;
        output.truncate(maximum_len - padding);
        return Ok(output);
    }

    dispatch!(level, simd => decode_impl(simd, input, &mut output, padding))?;

    validate_trailing_bits(input, padding)?;
    output.truncate(maximum_len - padding);
    Ok(output)
}

#[inline(always)]
fn encode_impl<S: Simd>(simd: S, mut input: &[u8], mut output: &mut [u8]) {
    let alphabet = u8x64::simd_from(simd, ALPHABET);
    let gather = u8x64::simd_from(simd, ENCODE_GATHER);

    while input.len() >= 48 {
        let low = u8x32::from_slice(simd, &input[..32]);
        let tail = u8x16::from_slice(simd, &input[32..48]);
        let zero = u8x16::splat(simd, 0);
        let bytes = low.combine(tail.combine(zero));
        let gathered = bytes.swizzle_dyn_precise(gather);
        let words: u32x16<S> = gathered.bitcast();

        let sextet_words = ((words >> 18) & 0x0000_003f)
            | ((words >> 4) & 0x0000_3f00)
            | ((words << 10) & 0x003f_0000)
            | ((words << 24) & 0x3f00_0000);
        let sextets: u8x64<S> = sextet_words.bitcast();
        alphabet
            .swizzle_dyn_precise(sextets)
            .store_slice(&mut output[..64]);

        input = &input[48..];
        output = &mut output[64..];
    }

    encode_tail(input, output);
}

#[inline(always)]
fn decode_impl<S: Simd>(
    simd: S,
    mut input: &[u8],
    mut output: &mut [u8],
    padding: usize,
) -> Result<(), DecodeError> {
    let low_table = u8x64::simd_from(simd, DECODE_LOW);
    let high_table = u8x64::simd_from(simd, DECODE_HIGH);
    let compact_indices = u8x64::simd_from(simd, DECODE_COMPACT);
    let input_len = input.len();
    let mut input_offset = 0;

    while input.len() >= 64 {
        let chars = u8x64::from_slice(simd, &input[..64]);
        let low = low_table.swizzle_dyn_precise(chars);
        let high = high_table.swizzle_dyn_precise(chars - 64);
        let sextets = low | high;
        let allowed_padding_bits = if input_offset + 64 == input_len {
            match padding {
                2 => (1_u64 << 62) | (1_u64 << 63),
                1 => 1_u64 << 63,
                _ => 0,
            }
        } else {
            0
        };
        let allowed_padding = mask8x64::from_bitmask(simd, allowed_padding_bits);
        let unexpected_padding = chars.simd_eq(b'=') & !allowed_padding;
        let invalid = (sextets | (chars & 0x80)).simd_gt(63) | unexpected_padding;

        if invalid.any_true() {
            let lane = invalid.to_bitmask().trailing_zeros() as usize;
            let index = input_offset + lane;
            let byte = input[lane];
            return if byte == b'=' {
                Err(DecodeError::InvalidPadding { index })
            } else {
                Err(DecodeError::InvalidByte { index, byte })
            };
        }

        let words: u32x16<S> = sextets.bitcast();
        let decoded_words = ((words << 2) & 0x0000_00fc)
            | ((words >> 12) & 0x0000_0003)
            | ((words << 4) & 0x0000_f000)
            | ((words >> 10) & 0x0000_0f00)
            | ((words << 6) & 0x00c0_0000)
            | ((words >> 8) & 0x003f_0000);
        let decoded: u8x64<S> = decoded_words.bitcast();
        let compacted = decoded.swizzle_dyn_precise(compact_indices);
        let (first, second) = compacted.split();
        let (middle, _) = second.split();
        first.store_slice(&mut output[..32]);
        middle.store_slice(&mut output[32..48]);

        input = &input[64..];
        output = &mut output[48..];
        input_offset += 64;
    }

    decode_tail(input, output, input_offset, input_len - padding)
}

fn encode_tail(input: &[u8], output: &mut [u8]) {
    let mut input_chunks = input.chunks_exact(3);
    let mut output_chunks = output.chunks_exact_mut(4);

    for (src, dst) in input_chunks.by_ref().zip(output_chunks.by_ref()) {
        encode_quantum(src[0], src[1], src[2], dst);
    }

    let remainder = input_chunks.remainder();
    let Some(dst) = output_chunks.next() else {
        debug_assert!(remainder.is_empty());
        return;
    };

    match remainder {
        [a] => {
            dst[0] = ALPHABET[(a >> 2) as usize];
            dst[1] = ALPHABET[((a & 0x03) << 4) as usize];
            dst[2] = b'=';
            dst[3] = b'=';
        }
        [a, b] => {
            dst[0] = ALPHABET[(a >> 2) as usize];
            dst[1] = ALPHABET[(((a & 0x03) << 4) | (b >> 4)) as usize];
            dst[2] = ALPHABET[((b & 0x0f) << 2) as usize];
            dst[3] = b'=';
        }
        [] => unreachable!("an output chunk cannot remain without an input remainder"),
        _ => unreachable!(),
    }
}

#[inline]
fn encode_quantum(a: u8, b: u8, c: u8, output: &mut [u8]) {
    output[0] = ALPHABET[(a >> 2) as usize];
    output[1] = ALPHABET[(((a & 0x03) << 4) | (b >> 4)) as usize];
    output[2] = ALPHABET[(((b & 0x0f) << 2) | (c >> 6)) as usize];
    output[3] = ALPHABET[(c & 0x3f) as usize];
}

fn decode_tail(
    input: &[u8],
    output: &mut [u8],
    offset: usize,
    padding_start: usize,
) -> Result<(), DecodeError> {
    for (chunk_index, (src, dst)) in input
        .chunks_exact(4)
        .zip(output.chunks_exact_mut(3))
        .enumerate()
    {
        let base = offset + chunk_index * 4;
        let a = decode_tail_byte(src[0], base, padding_start)?;
        let b = decode_tail_byte(src[1], base + 1, padding_start)?;
        let c = decode_tail_byte(src[2], base + 2, padding_start)?;
        let d = decode_tail_byte(src[3], base + 3, padding_start)?;

        dst[0] = (a << 2) | (b >> 4);
        dst[1] = (b << 4) | (c >> 2);
        dst[2] = (c << 6) | d;
    }

    Ok(())
}

fn padding_len(input: &[u8]) -> Result<usize, DecodeError> {
    if !input.len().is_multiple_of(4) {
        return Err(DecodeError::InvalidLength { len: input.len() });
    }
    if input.is_empty() {
        return Ok(0);
    }

    let padding = if input.ends_with(b"==") {
        2
    } else if input.ends_with(b"=") {
        1
    } else {
        0
    };

    Ok(padding)
}

fn validate_trailing_bits(input: &[u8], padding: usize) -> Result<(), DecodeError> {
    if padding == 2 {
        let index = input.len() - 3;
        let value = decode_byte(input[index]).ok_or(DecodeError::InvalidByte {
            index,
            byte: input[index],
        })?;
        if value & 0x0f != 0 {
            return Err(DecodeError::NonCanonicalTrailingBits { index });
        }
    } else if padding == 1 {
        let index = input.len() - 2;
        let value = decode_byte(input[index]).ok_or(DecodeError::InvalidByte {
            index,
            byte: input[index],
        })?;
        if value & 0x03 != 0 {
            return Err(DecodeError::NonCanonicalTrailingBits { index });
        }
    }

    Ok(())
}

#[inline]
fn decode_tail_byte(byte: u8, index: usize, padding_start: usize) -> Result<u8, DecodeError> {
    if byte == b'=' && index < padding_start {
        return Err(DecodeError::InvalidPadding { index });
    }

    decode_byte(byte).ok_or(DecodeError::InvalidByte { index, byte })
}

#[inline]
fn decode_byte(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        b'=' => Some(0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc_4648_examples() {
        let cases: &[(&[u8], &str)] = &[
            (b"", ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
        ];

        for &(plain, encoded) in cases {
            assert_eq!(encode(plain), encoded);
            assert_eq!(decode(encoded.as_bytes()).unwrap(), plain);
        }
    }

    #[test]
    fn round_trips_simd_boundaries() {
        for len in 0..=257 {
            let input: Vec<u8> = (0..len)
                .map(|i| (i as u8).wrapping_mul(31).wrapping_add(17))
                .collect();
            let encoded = encode(&input);
            assert_eq!(
                encoded,
                scalar_encode(&input),
                "encode failed at length {len}"
            );
            let decoded = decode(encoded.as_bytes()).unwrap();
            assert_eq!(decoded, input, "failed at input length {len}");
        }
    }

    #[test]
    fn exercises_every_alphabet_entry() {
        let input: Vec<u8> = (0..=255).collect();
        let encoded = encode(&input);
        for byte in ALPHABET {
            assert!(encoded.as_bytes().contains(&byte));
        }
        assert_eq!(decode(encoded.as_bytes()).unwrap(), input);
    }

    #[test]
    fn reports_invalid_input() {
        assert_eq!(decode(b"A"), Err(DecodeError::InvalidLength { len: 1 }));
        assert_eq!(
            decode(b"Zm=v"),
            Err(DecodeError::InvalidPadding { index: 2 })
        );
        assert_eq!(
            decode(b"=h=="),
            Err(DecodeError::InvalidPadding { index: 0 })
        );
        assert_eq!(
            decode(b"Zm9!"),
            Err(DecodeError::InvalidByte {
                index: 3,
                byte: b'!'
            })
        );
        assert_eq!(
            decode(b"Zh=="),
            Err(DecodeError::NonCanonicalTrailingBits { index: 1 })
        );
        assert_eq!(
            decode(b"Zm9="),
            Err(DecodeError::NonCanonicalTrailingBits { index: 2 })
        );
    }

    #[test]
    fn reports_invalid_byte_in_each_simd_lane() {
        let valid = encode(&[0x5a; 96]);
        assert_eq!(valid.len(), 128);

        for index in 0..128 {
            let mut corrupted = valid.clone().into_bytes();
            corrupted[index] = 0xff;
            assert_eq!(
                decode(&corrupted),
                Err(DecodeError::InvalidByte { index, byte: 0xff })
            );
        }
    }

    #[test]
    fn rejects_every_non_alphabet_byte_in_simd() {
        let valid = encode(&[0x5a; 96]);
        let index = 31;

        for byte in 0..=u8::MAX {
            if decode_byte(byte).is_some() {
                continue;
            }

            let mut corrupted = valid.clone().into_bytes();
            corrupted[index] = byte;
            assert_eq!(
                decode(&corrupted),
                Err(DecodeError::InvalidByte { index, byte })
            );
        }
    }

    #[test]
    fn reports_unexpected_padding_in_simd() {
        let valid = encode(&[0x5a; 96]);
        assert_eq!(valid.len(), 128);

        for index in 0..127 {
            let mut corrupted = valid.clone().into_bytes();
            corrupted[index] = b'=';
            assert_eq!(
                decode(&corrupted),
                Err(DecodeError::InvalidPadding { index })
            );
        }
    }

    #[test]
    fn baseline_backend_matches_scalar() {
        let input: Vec<u8> = (0..113)
            .map(|i| (i as u8).wrapping_mul(19).wrapping_add(7))
            .collect();
        let mut encoded = vec![0; encoded_len(input.len()).unwrap()];
        let level = Level::baseline();
        dispatch!(level, simd => encode_impl(simd, &input, &mut encoded));
        assert_eq!(encoded, scalar_encode(&input).into_bytes());

        let padding = padding_len(&encoded).unwrap();
        let mut decoded = vec![0; encoded.len() / 4 * 3];
        dispatch!(level, simd => decode_impl(simd, &encoded, &mut decoded, padding)).unwrap();
        validate_trailing_bits(&encoded, padding).unwrap();
        decoded.truncate(decoded.len() - padding);
        assert_eq!(decoded, input);
    }

    fn scalar_encode(input: &[u8]) -> String {
        let mut output = vec![0; encoded_len(input.len()).unwrap()];
        encode_tail(input, &mut output);
        String::from_utf8(output).unwrap()
    }
}
