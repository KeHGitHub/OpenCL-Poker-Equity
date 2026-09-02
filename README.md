# OpenCL-Poker-Equity

C++ and OpenCL Texas Hold'em equity estimator.

Stage one estimates a single hero hand against one or more unknown opponent hands using
Monte Carlo simulation. The OpenCL engine runs the simulation as independent
GPU work items.

On native Apple Silicon builds, this targets the OpenCL GPU path. The project
does not use an OpenCL CPU device.

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

Options:

- `--hand`: exactly two private hero cards.
- `--board`: zero to five public cards.
- `--opponents`: number of unknown opponents; defaults to `1`.
- `--trials`: Monte Carlo sample count; defaults to `1000000`.
- `--seed`: optional deterministic random seed.
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
