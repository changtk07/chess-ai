# Performance plan: self-play on an RTX 4090

Status as of 2026-10-25: Phases 0–2 are done. Phases 3 and 4 are next.

The goal is self-play fast enough to train on a single RTX 4090. The network evaluates positions in large batches on the GPU, and the CPU cores do the tree work for many games at once.

## Environment

- WSL2 running Arch Linux, i7-13700KF (24 threads), 15 GB RAM visible to WSL, RTX 4090 24 GB, Windows driver 621.01.
- The wgpu backend finds no Vulkan adapter under WSL2, so the GPU path is CUDA.
- burn's CUDA backend compiles kernels at runtime with NVRTC, which comes with the CUDA toolkit. Here that's CUDA 13.4 from `pacman -S cuda`, installed to `/opt/cuda`.
- cudarc 0.18, which burn 0.20 pins, rejects toolkits newer than CUDA 13.1. `.cargo/config.toml` sets `CUDARC_CUDA_VERSION=13010` so it builds against its 13.1 bindings, which work with 13.4.
- Every command takes `--backend cpu|cuda|cuda-f16|cuda-bf16`. The default is `cpu`.

## Diagnosis

Measured before any changes, with the default network (8 blocks, width 384, about 1.3 GFLOPs per position):

- The network was effectively 100% of search time. Move generation and tree work didn't register.
- On the CPU (NdArray), one position took 35–40 ms, and throughput topped out at about 50 positions/s at any batch size. One thread was as fast as 24.
- On the GPU, every call costs at least about 7.7 ms, the same for 1 position as for 16. That's fixed per-call overhead from kernel launches, synchronization and WSL. Search sends one position per call, so it gains only about 4× on the GPU (around 130 simulations/s versus 30). Throughput needs batches of hundreds of positions.

## Current measurements

Network throughput in positions/s on an idle GPU, after Phase 2. Each figure covers input encoding, the forward pass and readback, from `cinder bench 4096 --backend <backend>`:

| batch | CPU | f32 | f16 | bf16 |
|---|---|---|---|---|
| 1 | 24 | 119 | 107 | 125 |
| 16 | 49 | 2,031 | 1,879 | 2,066 |
| 64 | 51 | 6,740 | 7,370 | 7,912 |
| 256 | 52 | 8,118 | 14,057 | 15,566 |
| 1,024 | – | 8,077 | 13,276 | **16,224** |
| 4,096 | – | 9,114 | 13,429 | 13,782 |

Search, still one position per call: about 30 simulations/s on the CPU and 115–136 on the GPU.

Largest difference from the CPU, for the same weights, in policy or value (the GPU tests in `model.rs`): f32 6e-5, f16 5e-4, bf16 3e-3.

Kernel headroom: a plain 2D matrix multiply at the network's layer size, `[batch × 65, 384] × [384, 384]`, reaches 37–43 TFLOPS in f32. burn's 3D form, which `Linear` uses on `[batch, 65, 384]` inputs, gets about half that. The whole network does about 10 TFLOPS in f32 and about 20 in bf16.

## Benchmarking pitfalls

1. **Other GPU load skews everything.** A game running on Windows kept the GPU at about 99% and made identical runs differ by up to 11×. WSL can't see Windows processes, so check `nvidia-smi` power first (about 30–70 W when idle), and use Task Manager or `nvidia-smi` in PowerShell to find the culprit.
2. **The first calls for each new tensor shape compile and autotune kernels.** Autotune results persist in `target/autotune`, but compiled kernels don't survive the process, so each run recompiles. `bench` keeps warming up until calls stop getting faster.
3. **Uploading many tiny tensors and stacking them stalls burn's fusion optimizer** on one CPU core, and the cost grows faster than linearly with the number of operations. Build each batch in one host buffer and upload it once.
4. **A scratch copy of the repo that shares `CARGO_TARGET_DIR` reuses the repo's binaries,** because cargo doesn't hash the absolute path. Rename the package in the copy's `Cargo.toml`.
5. **Stack sampling needs `sudo eu-stack -p <pid>`,** since ptrace is restricted for normal users.

## Phases

### Done

- **Phase 0: benchmark command.** `cinder bench [max_batch]` reports network throughput by batch size, then search speed. (`feat: add bench command for network and search throughput`)
- **Phase 1: CUDA backend.**
  - `perf: build legal move masks for a whole batch in one upload`
  - `feat: add CUDA backend with f16 and bf16 precision`. Includes GPU-versus-CPU tests, skipped by default; run them with `cargo test --release -- --ignored`.
  - `fix: warm up bench until GPU kernels settle`
- **Phase 2: legal moves only, one readback.**
  - Instead of a 4,672-wide mask, each position's legal-move indices are uploaded, padded to the batch's largest move count, with a padding mask.
  - The policy head picks out just those logits on the GPU, and the softmax covers only legal moves.
  - `TransformerModel::evaluate` reads priors and values back in a single transfer and returns one `Evaluation { priors, value }` per position, with priors in move order.
  - `forward` returns tensors, for training later.
  - bf16 at batch 1,024 went from 11,639 to 16,224 positions/s. (`perf: evaluate legal moves only, with a single readback`)

### Phase 3: pausable search

Split the recursive `Mcts::explore` into steps, so a search can stop at a position that needs the network and resume once its batch is evaluated.

```rust
/// Runs simulations until one reaches a position the network must evaluate, leaving `state`
/// there. Finished games are scored inside, so they never need the network.
fn next_leaf(&mut self, state: &mut State) -> bool;   // false once all n_sims are done
fn leaf_moves(&self) -> &[Move];
/// Expands the pending leaf with the network's output, backs the value up the stored path,
/// and returns `state` to the root.
fn apply(&mut self, state: &mut State, evaluation: Evaluation);
```

- Each search keeps a path stack instead of recursing. Values alternate sign on the way up, as they do now.
- Each search has at most one position waiting for the network, so virtual loss isn't needed.
- `Mcts::search()` becomes a loop over `next_leaf`, `evaluate` and `apply`. The bench, the smoke game and all existing tests, including the mate-in-1 and visit-count tests, keep working unchanged and guard the refactor.
- A small commit first: store each node's children as a contiguous index range instead of a `Vec<usize>`. They're already pushed contiguously. That removes one allocation per expansion and shrinks nodes from about 56 to 40 bytes. 256 trees × 800 simulations is about 7M nodes, so that saves roughly 100 MB.

### Phase 4: parallel self-play driver

N games in flight, 256 by default. Each step:

1. **Select, in parallel with rayon** (1.12, already in the dependency tree): each game calls `next_leaf`.
   - A game whose search is finished plays its move: sampled with temperature for the first 30 plies, then the most-visited move. It then starts a new search.
   - A finished game is replaced by a new one, so all slots stay full.
2. **Evaluate:** one `evaluate` call on the GPU for every pending position.
3. **Apply, in parallel:** each game calls `apply`.

Details:

- Each game has its own seeded RNG, so runs are reproducible whatever the thread scheduling.
- The command is `cinder self-play --parallel 256 --sims 800 --games N --backend cuda-bf16`. It replaces the current single smoke game.
- Progress is reported every few seconds: games finished, simulations/s, evaluations/s, average batch size, and how each step's time splits between select, GPU and apply. That split is the diagnostic for whatever comes next.
- Test: with a tiny model on the CPU, 8 games at once with few simulations must each end in a real outcome or reach the ply cap.
- Measure 1 thread against 24 with `RAYON_NUM_THREADS=1`, and try 256, 512 and 1,024 games. bf16 is at 15,566 positions/s at batch 256 and 16,224 at 1,024.

CPU parallelism, honestly: tree work is an estimated 3–5 µs per game per step, so about 1 ms for 256 games on one core, against about 16 ms for the GPU batch. Rayon should start out at around 5% gain. It matters for keeping up as the GPU gets faster. Overlapping CPU and GPU work waits until the measured split says it's worth it.

Expected result: around 10,000+ simulations/s with 256 games, from about 130 now.

### Planned commits

1. `refactor: store MCTS children as a contiguous range`
2. `refactor: split MCTS simulations into select and apply steps`
3. `feat: play self-play games in parallel with batched evaluation`

Measure with `cinder bench` and the self-play report before moving on.

## Throughput math

At about 12,000 simulations/s and 800 simulations per move, a 150-ply game is 120,000 evaluations, so about 360 games per hour across all slots. AlphaZero-style training wants tens of thousands of games, so the bigger levers after this plan are:

- Fewer simulations per move (100–200), with occasional full searches ("playout cap randomization", as KataGo does). At 150 simulations it would be about 2,000 games per hour.
- A smaller network for early training.
- The kernel headroom above.

## Decisions

- **bf16 for self-play.** It's about as fast as f16, can't overflow (f16's maximum is 65,504, which trained activations could exceed), and holds up better at large batches. Its accuracy is within 3e-3 of the CPU. f16 is a one-flag switch if accuracy turns out to matter.
- **256 parallel games by default.** It's a flag, to be swept in Phase 4.
- **The CPU stays the default backend,** so tests and other machines stay portable. Pass `--backend cuda-bf16` for self-play.

## After this plan

- Self-play data: Dirichlet noise at the root, recording `(position, visit distribution, result)` training samples, and writing them to disk.
- Reusing the search tree between moves.
- Kernel efficiency: flatten inputs to 2D before linear layers; look at attention.
- Overlapping CPU tree work with GPU evaluation (two groups of games taking turns).
- Training on the GPU (burn's `Autodiff<Cuda>`), then a UCI interface.
