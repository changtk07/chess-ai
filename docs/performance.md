# Performance notes

How self-play runs fast on an RTX 4090 under WSL2, with measurements to compare future changes against and the pitfalls hit along the way. Measured 2026-10-25 and 2026-10-26; the 2D layers and fused attention were measured on a laptop on 2026-10-28, and the 4090 figures predate them.

Self-play runs at about 15,500 simulations/s: 256 games at once, with their positions evaluated in batches on the GPU in bf16 and the tree work on one CPU thread. Before this work it ran about 30 simulations/s on the CPU. The network is now about 98% of every step, so further speed has to come from the network.

The work is in the commits from `feat: add bench command for network and search throughput` (826ed90) through `feat: play self-play games in parallel with batched evaluation`.

## Environment

- WSL2 running Arch Linux, i7-13700KF (24 threads), 15 GB RAM visible to WSL, RTX 4090 24 GB, Windows driver 621.01.
- Also a laptop: WSL2 running Arch Linux, 16 threads, 31 GB RAM visible to WSL, RTX 5080 Laptop GPU 16 GB (power limit up to 130 W), the same driver and CUDA 13.4. It runs the network at roughly half the 4090's speed, so compare measurements within one machine only.
- The wgpu backend finds no Vulkan adapter under WSL2, so the GPU path is CUDA.
- burn's CUDA backend compiles kernels at runtime with NVRTC, which comes with the CUDA toolkit. Here that's CUDA 13.4 from `pacman -S cuda`, installed to `/opt/cuda`.
- cudarc 0.18, which burn 0.20 pins, rejects toolkits newer than CUDA 13.1. `.cargo/config.toml` sets `CUDARC_CUDA_VERSION=13010` so it builds against its 13.1 bindings, which work with 13.4.
- Every command takes `--backend cpu|cuda|cuda-f16|cuda-bf16`. The default is `cpu`.

## What makes it fast

- **Batches across games.** The GPU costs at least about 7.7 ms per call, the same for 1 position as for 16, because of kernel launches, synchronization and WSL. A single search sends one position per call, so `self-play` keeps 256 games going and evaluates one position from each in a single call.
- **Pausable search.** `Search::next_leaf` runs simulations until one reaches a position the network must evaluate, and leaves the state there. `Search::apply` takes the network's output, backs the value up and returns the state to the root. Finished games are scored without the network.
- **Legal moves only.** Each position's legal-move indices are uploaded, padded to the batch's largest move count. The policy head picks out just those logits on the GPU, and priors and values come back in one small transfer, instead of all 4,672 moves per position.
- **64 tokens.** Castling rights, en passant file and the halfmove clock are added into every square's token rather than given a token of their own, since attention over 64 tokens runs far faster than over 65.
- **One upload per input.** A batch's inputs are built in single host buffers. Per-position tensors stall burn's fusion optimizer (see the pitfalls below).
- **2D matrix multiplies.** The blocks keep every position's tokens stacked in one `[batch × 64, 384]` matrix, because burn runs `Linear` on 3D inputs well below the speed of the 2D form. Attention reshapes into heads only around its own two matrix multiplies, and projects queries, keys and values in a single one.

## Measurements

Default network: 8 blocks, width 384, about 1.3 GFLOPs per position.

### Before

- The network was effectively 100% of search time. Move generation and tree work didn't register.
- On the CPU (NdArray), one position took 35–40 ms, and throughput topped out at about 50 positions/s at any batch size. One thread was as fast as 24.
- Single-game search: about 30 simulations/s on the CPU and 115–136 on the GPU.

### Network throughput

Positions/s on an idle GPU, covering input encoding, the forward pass and readback, from `cinder bench 4096 --backend <backend>`:

| batch | CPU | f32 | f16 | bf16 |
|---|---|---|---|---|
| 1 | 24 | 119 | 107 | 125 |
| 16 | 49 | 2,031 | 1,879 | 2,066 |
| 64 | 51 | 6,740 | 7,370 | 7,912 |
| 256 | 52 | 8,118 | 14,057 | 15,566 |
| 1,024 | – | 8,077 | 13,276 | **16,224** |
| 4,096 | – | 9,114 | 13,429 | 13,782 |

Largest difference from the CPU, for the same weights, in policy or value (the GPU tests in `model.rs`): f32 6e-5, f16 5e-4, bf16 3e-3.

Re-measured on 2026-10-26 at `418e42c`, two runs: bf16 gave 15,027 and 15,270 at batch 256, and 15,329 and 15,418 at 1,024. That's a few percent under the table, with no network code changed in between, so treat differences of that size as run-to-run variation.

### Self-play

`cinder self-play --sims 800 --backend cuda-bf16`. Steady-state simulations/s, averaged over 5-second reports after kernel compilation:

| parallel games | tree work on 24 threads (rayon) | tree work on 1 thread |
|---|---|---|
| 256 | 11,944 | **15,501** |
| 512 | 14,581 | 15,118 |
| 1,024 | 14,533 | 15,344 |

On one thread, the network takes about 98% of each step and the tree work about 2%.

Re-measured on 2026-10-26 at `418e42c`, after root noise and sample recording were added: 256 games averaged 15,331 simulations/s (14,804-15,502) with `--out` writing samples. The network took 98.2% of each step and writing samples about 0%, so neither feature costs measurable throughput.

With a fixed `--games`, batches shrink as the last games finish, and those games run at near single-position speed. Continuous generation avoids this.

### Kernel headroom

A plain 2D matrix multiply at the network's layer size, `[batch × 65, 384] × [384, 384]`, reaches 37–43 TFLOPS in f32. burn's 3D form, which `Linear` uses on `[batch, 65, 384]` inputs, gets about half that. The whole network does about 10 TFLOPS in f32 and about 20 in bf16.

### 2D layers and fused attention

Measured on the laptop with `--backend cuda-bf16`, at batch 256 unless noted.

A chain of 32 `Linear(384, 384)` layers ran at 20 TFLOPS on `[batch, 65, 384]` inputs and 29 TFLOPS on the same data as a `[batch × 65, 384]` matrix (31 against 21 at batch 1,024). In f32 the gap was wider: 17 against 9.5 TFLOPS.

Timing the network's parts before the change, attention took 22 of its 35 ms, the feed-forward layers 12.5, and input encoding, embedding, heads and readback about 1.2 together. Attention's four projections do the same work as the feed-forward layers, so about 10 ms went to the attention core: the score matrix multiplies, softmax and the reshapes between tokens and heads. Its matrix multiplies are 1,536 tiny `[65 × 64] × [64 × 65]` products per call, which run far below the GPU's speed.

Eight attention sublayers, chained:

| attention | ms |
|---|---|
| burn's `MultiHeadAttention`, 3D | 23.1 |
| written out, 2D projections | 16.7 |
| written out, one fused query/key/value projection | **15.3** |
| the same without the core (projections only) | 6.2 |

The whole network, two runs each, alternating builds:

| | positions/s, `bench` batch 256 | self-play simulations/s |
|---|---|---|
| before | 7,410 and 5,872 | 6,850 and 6,800 |
| after | 9,699 and 10,223 | 11,840 and 11,730 |

Self-play figures average the 5-second reports after the first minute. Self-play gained more than `bench` did, about 1.7×; the laptop's runs vary more than the 4090's did.

### 64 tokens instead of 65

The network used to give castling rights, en passant file and halfmove clock a token of their own, a 65th next to the squares. Adding them into every square's token instead keeps the network the same size, but attention's tiny matrix multiplies run much faster at 64 than at 65, likely because 65 spills into an extra, almost empty tile of the GPU's matrix kernels.

On the laptop, a prototype measured with `bench` at batch 256 went from 10,200 to 13,400 positions/s at 8 blocks of width 384, from 16,700 to 24,500 at 8 × 256, and from 21,300 to 32,000 at 6 × 256. Self-play at 8 × 384 with 256 games went from 11,071 to 13,889 simulations/s.

### Network size

Self-play simulations/s on the laptop with 64 tokens, `--backend cuda-bf16`, from `--blocks` and `--width`, averaging the 5-second reports from 40 s on:

| blocks × width | 256 games | 512 games |
|---|---|---|
| 6 × 256 | 26,965 | 29,061 |
| 8 × 256 | 24,316 | **27,596** |
| 10 × 256 | 20,425 | 20,864 |
| 12 × 256 | 16,516 | 17,855 |
| 6 × 384 | 16,529 | 19,209 |
| 8 × 384 | 13,889 | 12,974 |
| 10 × 384 | 10,173 | 10,188 |

`scripts/bench-sizes.sh` repeats this measurement on any machine, averaging all reports after the first two instead.

From 8 blocks up, time grows about in proportion to the number of blocks. Below that, the work outside the blocks stops shrinking: 6 blocks gained only 5–10% over 8, with the network still about 93% of each step. 512 games helped the smaller networks and made no difference to the larger ones. Runs differ by up to about 10% on the laptop, so treat smaller gaps as noise.

## CPU parallelism

Tree work on one thread takes about 0.36 ms per 256-game step, against about 17 ms for the network. With rayon over 24 threads it took about 3.6 ms, and the network call itself rose from about 17.4 to 18.9 ms. Dispatching thousands of microsecond tasks costs more than it saves, and idle workers likely compete with the thread driving the GPU. Rayon was slower at every size tried, so the tree work is sequential.

Extra cores become worth it only once the network is roughly 10× faster and the tree work's share grows. At that point, overlapping CPU and GPU work, with two groups of games taking turns, is the first thing to try.

## Throughput math

At about 15,500 simulations/s and 800 simulations per move, a 150-ply game is 120,000 evaluations, so about 465 games per hour across all slots. AlphaZero-style training wants tens of thousands of games, so the bigger levers are:

- Fewer simulations per move (100–200), with occasional full searches ("playout cap randomization", as KataGo does). At 150 simulations it would be about 2,500 games per hour.
- A smaller network for early training.
- The kernel headroom above.

## Decisions

- **bf16 for self-play.** It's about as fast as f16, can't overflow (f16's maximum is 65,504, which trained activations could exceed), and holds up better at large batches. Its accuracy is within 3e-3 of the CPU. f16 is a one-flag switch if accuracy turns out to matter.
- **256 parallel games by default.** 512 and 1,024 were no faster and use more memory.
- **Tree work on one thread,** per the measurements above.
- **The CPU stays the default backend,** so tests and other machines stay portable. Pass `--backend cuda-bf16` for self-play.
- **Attention is written out** rather than calling `burn::tensor::module::attention`. In burn 0.20 that op falls back to the same naive steps under the fusion backend, so it gains nothing, and without fusion the CUDA backend runs a flash-attention kernel with a causal mask hard-coded, which would silently stop squares from attending to later squares.

## Benchmarking pitfalls

1. **Other GPU load skews everything.** A game running on Windows kept the GPU at about 99% and made identical runs differ by up to 11×. WSL can't see Windows processes, so check `nvidia-smi` power first (about 30–70 W when idle), and use Task Manager or `nvidia-smi` in PowerShell to find the culprit.
2. **The first calls for each new tensor shape compile and autotune kernels.** Autotune results persist in `target/autotune`, but compiled kernels don't survive the process, so each run recompiles. `bench` keeps warming up until calls stop getting faster; skip the first self-play reports.
3. **Uploading many tiny tensors and stacking them stalls burn's fusion optimizer** on one CPU core, and the cost grows faster than linearly with the number of operations. Build each batch in one host buffer and upload it once.
4. **A scratch copy of the repo that shares `CARGO_TARGET_DIR` reuses the repo's binaries,** because cargo doesn't hash the absolute path. Rename the package in the copy's `Cargo.toml`.
5. **Stack sampling needs `sudo eu-stack -p <pid>`,** since ptrace is restricted for normal users.
6. **zsh doesn't split unquoted variables into words,** so `set -- $cfg` and similar loops in benchmark scripts pass one argument where bash would pass several.
7. **Laptop power modes cap the GPU.** Partway through one session the laptop's GPU limit dropped to 33 W, and the network ran 5× slower. Check `enforced.power.limit` with `nvidia-smi --query-gpu=power.draw,enforced.power.limit,clocks_event_reasons.sw_power_cap --format=csv` under load. On ASUS laptops, G-Helper's Turbo or Balanced mode on the barrel charger lifts it, and the limit rises to 103–130 W under load.
8. **Restarting WSL clears `/tmp`.** Keep baseline binaries for before/after comparisons under `target/`.
