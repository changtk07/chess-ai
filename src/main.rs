mod bench;
mod chess;
mod engine;

use crate::{
    chess::State,
    engine::{
        mcts::MctsConfig,
        model::TransformerModelConfig,
        samples::{SampleWriter, UNTRAINED},
        self_play::{self as self_play_driver, GameResult, SelfPlayConfig, SelfPlayStats},
    },
};
use burn::{
    backend::{Cuda, NdArray},
    tensor::{backend::Backend, bf16, f16},
};
use std::{path::Path, str::FromStr, time::Instant};

const USAGE: &str = "\
Usage: cinder <command> [--backend cpu|cuda|cuda-f16|cuda-bf16]

Commands:
  self-play [--parallel 256] [--sims 800] [--games 256] [--seed 0] [--out file]
                      play games with untrained weights, reporting throughput and
                      appending each finished game's training samples to the file
  bench [max_batch]   network throughput by batch size, then search speed
  perft [depth]       count positions from the start, per root move";

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let backend = take_option(&mut args, "--backend").unwrap_or_else(|| "cpu".into());

    match backend.as_str() {
        "cpu" => run::<NdArray>(args, &Default::default()),
        "cuda" => run::<Cuda>(args, &Default::default()),
        "cuda-f16" => run::<Cuda<f16>>(args, &Default::default()),
        "cuda-bf16" => run::<Cuda<bf16>>(args, &Default::default()),
        other => usage_error(&format!("unknown backend {other}")),
    }
}

fn run<B: Backend>(mut args: Vec<String>, device: &B::Device) {
    if args.is_empty() {
        println!("{USAGE}");
        return;
    }

    match args.remove(0).as_str() {
        "perft" => {
            let depth = take_positional(&mut args, "depth", 5);
            reject_leftovers(&args);
            perft(depth);
        }
        "bench" => {
            let max_batch_size = take_positional(&mut args, "max_batch", 256);
            reject_leftovers(&args);
            bench::run::<B>(device, max_batch_size);
        }
        "self-play" => self_play::<B>(args, device),
        other => usage_error(&format!("unknown command {other}")),
    }
}

// Removes `name` and the value after it from `args`, returning the value
fn take_option(args: &mut Vec<String>, name: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == name)?;
    args.remove(index);

    match args.get(index) {
        Some(value) if !value.starts_with("--") => Some(args.remove(index)),
        _ => usage_error(&format!("{name} needs a value")),
    }
}

fn take_parsed_option<T: FromStr>(args: &mut Vec<String>, name: &str, default: T) -> T {
    take_option(args, name).map_or(default, |value| parse(name, &value))
}

// Takes the first argument if it isn't an option
fn take_positional<T: FromStr>(args: &mut Vec<String>, name: &str, default: T) -> T {
    match args.first() {
        Some(value) if !value.starts_with("--") => parse(name, &args.remove(0)),
        _ => default,
    }
}

fn parse<T: FromStr>(name: &str, value: &str) -> T {
    value
        .parse()
        .unwrap_or_else(|_| usage_error(&format!("{name} expects a number, got {value}")))
}

fn reject_leftovers(args: &[String]) {
    if !args.is_empty() {
        usage_error(&format!("unexpected arguments: {}", args.join(" ")));
    }
}

fn usage_error(message: &str) -> ! {
    eprintln!("error: {message}\n\n{USAGE}");
    std::process::exit(2);
}

fn perft(depth: u8) {
    let mut state = State::new();
    let start_time = Instant::now();

    println!("Running perft depth {} on starting position...", depth);
    state.divide(depth);

    let elapsed = start_time.elapsed();
    println!("Time taken: {:?}", elapsed);
}

fn self_play<B: Backend>(mut args: Vec<String>, device: &B::Device) {
    let out = take_option(&mut args, "--out");
    let n_parallel: usize = take_parsed_option(&mut args, "--parallel", 256);
    let n_sims: usize = take_parsed_option(&mut args, "--sims", 800);
    let n_games: usize = take_parsed_option(&mut args, "--games", 256);
    let seed: u64 = take_parsed_option(&mut args, "--seed", 0);
    reject_leftovers(&args);

    if n_parallel == 0 || n_games == 0 {
        usage_error("--parallel and --games must be at least 1");
    }
    // The first simulation only expands the root, and samples store visit counts in 16 bits
    if !(2..=u16::MAX as usize).contains(&n_sims) {
        usage_error(&format!("--sims must be from 2 to {}", u16::MAX));
    }

    let mut writer = out.as_deref().map(|path| {
        SampleWriter::open(Path::new(path))
            .unwrap_or_else(|error| panic!("can't write samples to {path}: {error}"))
    });
    let mut n_samples = 0;

    // Until weights can be loaded from a file, the seed also fixes the random initial weights
    B::seed(device, seed);
    let model = TransformerModelConfig::new().init::<B>(device);
    let mcts_config = MctsConfig::new().with_n_sims(n_sims);
    let config = SelfPlayConfig::new()
        .with_n_parallel(n_parallel)
        .with_n_games(n_games)
        .with_seed(seed);

    println!(
        "Backend: {}, {n_parallel} games at a time, {n_sims} simulations per move",
        B::name(device)
    );

    let mut previous = SelfPlayStats::default();
    let stats = self_play_driver::run(
        &model,
        &mcts_config,
        &config,
        device,
        |_, samples| {
            if let Some(writer) = writer.as_mut() {
                writer
                    .write_game(UNTRAINED, &samples)
                    .expect("writing samples");
                n_samples += samples.len();
            }
        },
        |stats| {
            print_progress(stats, &previous);
            previous = stats.clone();
        },
    );

    let count = |result| {
        stats
            .games
            .iter()
            .filter(|record| record.result == result)
            .count()
    };
    let total_plies: usize = stats.games.iter().map(|record| record.plies).sum();
    println!(
        "\n{} games: {} white wins, {} black wins, {} draws, {} stopped at the ply limit, {:.0} plies on average",
        stats.games.len(),
        count(GameResult::WhiteWins),
        count(GameResult::BlackWins),
        count(GameResult::Draw),
        count(GameResult::PlyLimit),
        total_plies as f64 / stats.games.len() as f64,
    );
    print!("Overall: ");
    print_progress(&stats, &SelfPlayStats::default());
    if let Some(path) = out {
        println!("{n_samples} samples appended to {path}");
    }
}

// Rates over the interval since `previous`, and where that time went
fn print_progress(current: &SelfPlayStats, previous: &SelfPlayStats) {
    let seconds = (current.elapsed - previous.elapsed).as_secs_f64();
    let evaluations = (current.n_evaluations - previous.n_evaluations) as f64;
    let steps = (current.n_steps - previous.n_steps) as f64;
    let share = |now: std::time::Duration, before: std::time::Duration| {
        100.0 * (now - before).as_secs_f64() / seconds
    };

    println!(
        "{:>5.0}s  games {:>4}  {:>6.0} sims/s  {:>6.0} evals/s  batch {:>5.1}  \
         time: select {:>4.1}%, network {:>4.1}%, apply {:>4.1}%",
        current.elapsed.as_secs_f64(),
        current.games.len(),
        (current.n_sims - previous.n_sims) as f64 / seconds,
        evaluations / seconds,
        evaluations / steps,
        share(current.select_time, previous.select_time),
        share(current.evaluate_time, previous.evaluate_time),
        share(current.apply_time, previous.apply_time),
    );
}
