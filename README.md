# OpenCL-Poker-Equity

C++ and OpenCL Texas Hold'em equity estimator.

Stage one estimates a single hero hand against one or more unknown opponent hands using
Monte Carlo simulation. The OpenCL engine runs the simulation as independent
GPU work items.

## Why Monte Carlo?

Poker equity is a search over unknown cards. If you hold `As Ah` before the
flop against one unknown opponent, there are 50 unseen cards left:

- Opponent hole cards: `C(50, 2) = 1,225`
- Five-card board runouts after that: `C(48, 5) = 1,712,304`
- Total opponent-plus-board completions: `2,097,572,400`

That is already about 2.1 billion complete deals for a single heads-up preflop
query. With five unknown opponents, the labeled opponent-hand and board search
space is:

```text
C(50,2) * C(48,2) * C(46,2) * C(44,2) * C(42,2) * C(40,5)
= 766,497,953,677,279,824,000
```

Exhaustively evaluating every completion gets expensive quickly. Monte Carlo
sampling estimates the same probability by randomly dealing many valid
completions instead. Each trial is independent, so OpenCL can run thousands or
millions of trials in parallel on the GPU.


## Build

```sh
cmake -S . -B build
cmake --build build
```

## Run

```sh
./build/poker_equity --hand "As Ks"
./build/poker_equity --hand "As Ks" --board "Qh Jh 2c" --trials 1000000
./build/poker_equity --hand "As Ah" --opponents 5 --trials 1000000
./build/poker_equity --hand "As Ah" --seed 123
```

## Profile

Profile mode compares a pure C++ CPU reference implementation against the
OpenCL engine. It builds the OpenCL program once, runs warmups, then reports
per-run timings plus min/mean/max summaries.

By default, profile mode randomizes the hero hand and public board for each
run. If you provide `--hand` or `--board`, that part stays fixed and the missing
part is randomized.

```sh
./build/poker_equity --profile --trials 1000000 --profile-runs 10 --warmup-runs 2
./build/poker_equity --profile --hand "As Ah" --opponents 5
```

Options:

- `--hand`: exactly two private hero cards.
- `--board`: zero to five public cards.
- `--opponents`: number of unknown opponents; defaults to `1`.
- `--trials`: Monte Carlo sample count; defaults to `1000000`.
- `--seed`: optional deterministic random seed.
- `--profile`: run repeated CPU reference vs OpenCL profiling.
- `--profile-runs`: measured runs for `--profile`; defaults to `5`.
- `--warmup-runs`: unmeasured warmup runs for `--profile`; defaults to `1`.
- `--list-devices`: print OpenCL platform and device diagnostics.

Cards use standard rank+suit notation:

- Ranks: `2 3 4 5 6 7 8 9 T J Q K A`
- Suits: `c d h s`

Example output:

```text
Hero hand: As Ks
Board:     Qh Jh 2c
Opponents: 1 unknown random hand
Trials:    1000000
Seed:      123
Device:    Apple M2

Win:       63.4210%
Tie:       1.1270%
Loss:      35.4520%
Equity:    63.9845%
```
