use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use intrepid_base64::{decode, encode};

const SIZES: &[usize] = &[
    1,
    2,
    3,
    45,
    46,
    47,
    48,
    49,
    63,
    64,
    65,
    1024,
    16 * 1024,
    1024 * 1024,
];

fn input(size: usize) -> Vec<u8> {
    (0..size)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(17))
        .collect()
}

fn encode_benchmarks(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("encode");

    for &size in SIZES {
        let input = input(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &input,
            |bencher, input| {
                bencher.iter(|| encode(black_box(input)));
            },
        );
    }

    group.finish();
}

fn decode_benchmarks(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("decode");

    for &size in SIZES {
        let encoded = encode(&input(size));
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &encoded,
            |bencher, encoded| {
                bencher.iter(|| decode(black_box(encoded.as_bytes())).unwrap());
            },
        );
    }

    group.finish();
}

criterion_group!(benches, encode_benchmarks, decode_benchmarks);
criterion_main!(benches);
