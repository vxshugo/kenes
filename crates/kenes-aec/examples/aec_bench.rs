//! CPU cost of the echo canceller: `cargo run --release -p kenes-aec --example aec_bench`.
//!
//! Processes 60 s of far-end bursts with their echo (the worst case: the canceller is
//! active the whole time) and reports the best of 5 runs as a real-time factor on one
//! thread, plus the slowest single 10 ms frame.

use std::time::{Duration, Instant};

use kenes_aec::{AecConfig, EchoCanceller, FRAME};

fn signals(seconds: usize) -> (Vec<f32>, Vec<f32>) {
    let n = seconds * 16_000;
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    // Talk spurts: 1 s on, 0.5 s off.
    let far: Vec<f32> = (0..n).map(|i| if i % 24_000 < 16_000 { rnd() * 0.2 } else { 0.0 }).collect();
    let mut near = vec![0.0f32; n];
    for i in 1_650..n {
        near[i] = 0.3 * far[i - 1_600] + 0.1 * far[i - 1_650] + rnd() * 0.002;
    }
    (far, near)
}

fn main() {
    let seconds = 60;
    let (far, near) = signals(seconds);
    for (name, cfg) in [
        ("AEC3 + delay tracker (default)", AecConfig::default()),
        ("AEC3 only", AecConfig { track_delay: false, ..AecConfig::default() }),
    ] {
        let mut best = Duration::MAX;
        let mut worst_frame = Duration::ZERO;
        for _ in 0..5 {
            let mut aec = EchoCanceller::new(cfg.clone());
            let mut out = near.clone();
            let t = Instant::now();
            for (f, n) in far.as_chunks::<FRAME>().0.iter().zip(out.as_chunks_mut::<FRAME>().0) {
                let tf = Instant::now();
                aec.process_frame(f, n);
                worst_frame = worst_frame.max(tf.elapsed());
            }
            best = best.min(t.elapsed());
        }
        println!(
            "{name}: {:.3} s per {seconds} s of audio, RTF {:.4}; slowest frame {:.2} ms",
            best.as_secs_f64(),
            best.as_secs_f64() / seconds as f64,
            worst_frame.as_secs_f64() * 1000.0
        );
    }
}
