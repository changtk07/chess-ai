# Performance notes

How self-play runs fast on an RTX 4090 under WSL2, with measurements to compare future changes against and the pitfalls hit along the way. Measured 2026-10-25 and 2026-10-26.

Self-play runs at about 15,500 simulations/s: 256 games at once, with their positions evaluated in batches on the GPU in bf16 and the tree work on one CPU thread. Before this work it ran about 30 simulations/s on the CPU. The network is now about 98% of every step, so further speed has to come from the network.

The work is in the commits from `feat: add bench command for network and search throughput` (826ed90) through `feat: play self-play games in parallel with batched evaluation`.

## Environment

- WSL2 running Arch Linux, i7-13700KF (24 threads), 15 GB RAM visible to WSL, RTX 4090 24 GB, Windows driver 621.01.
- The wgpu backend finds no Vulkan adapter under WSL2, so the GPU path is CUDA.
- burn's CUDA backend compiles kernels at runtime with NVRTC, which comes with the CUDA toolkit. Here that's CUDA 13.4 from `pacman -S cuda`, installed to `/opt/cuda`.
- cudarc 0.18, which burn 0.20 pins, rejects toolkits newer than CUDA 13.1. `.cargo/config.toml` sets `CUDARC_CUDA_VERSION=13010` so it builds against its 13.1 bindings, which work with 13.4.
- Every command takes `--backend cpu|cuda|cuda-f16|cuda-bf16`. The default is `cpu`.

## What makes it fast

- **Batches across games.** The GPU costs at least about 7.7 ms per call, the same for 1 position as for 16, because of kernel launches, synchronization and WSL. A single search sends one position per call, so `self-play` keeps 256 games going and evaluates one position from each in a single call.
- **Pausable search.** `Search::next_leaf` runs simulations until one reaches a position the network must evaluate, and leaves the state there. `Search::apply` takes the network's output, backs the value up and returns the state to the root. Finished games are scored without the network.
- **Legal moves only.** Each position's legal-move indices are uploaded, padded to the batch's largest move count. The policy head picks out just those logits on the GPU, and priors and values come back in one small transfer, instead of all 4,672 moves per position.
- **One upload per input.** A batch's inputs are built in single host buffers. Per-position tensors stall burn's fusion optimizer (see the pitfalls below).

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

## Benchmarking pitfalls

1. **Other GPU load skews everything.** A game running on Windows kept the GPU at about 99% and made identical runs differ by up to 11×. WSL can't see Windows processes, so check `nvidia-smi` power first (about 30–70 W when idle), and use Task Manager or `nvidia-smi` in PowerShell to find the culprit.
2. **The first calls for each new tensor shape compile and autotune kernels.** Autotune results persist in `target/autotune`, but compiled kernels don't survive the process, so each run recompiles. `bench` keeps warming up until calls stop getting faster; skip the first self-play reports.
3. **Uploading many tiny tensors and stacking them stalls burn's fusion optimizer** on one CPU core, and the cost grows faster than linearly with the number of operations. Build each batch in one host buffer and upload it once.
4. **A scratch copy of the repo that shares `CARGO_TARGET_DIR` reuses the repo's binaries,** because cargo doesn't hash the absolute path. Rename the package in the copy's `Cargo.toml`.
5. **Stack sampling needs `sudo eu-stack -p <pid>`,** since ptrace is restricted for normal users.
6. **zsh doesn't split unquoted variables into words,** so `set -- $cfg` and similar loops in benchmark scripts pass one argument where bash would pass several.
