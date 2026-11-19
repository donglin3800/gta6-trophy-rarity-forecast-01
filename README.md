# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/donglin3800/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/donglin3800/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Week-one GTA VI trophy earn-rate forecaster: robust, decelerating logit trends fitted to real Exophase reads, offline and stdlib-only. It predicts the 5th, 25th, 50th, 75th and 95th percentiles of each trophy's earn rate at the target time, for the make.fun bounty "Forecast how many GTA VI players earn each trophy in week one".

## Run it

```sh
MAKEFUN_INPUT_DIR=./input MAKEFUN_OUTPUT_DIR=./output python3 main.py
```

The program reads every `*.in` case in the input directory (default `/input`) and writes one `<id>.out` per case to the output directory (default `/output`). Each output is a JSON object with one key, `trophies`, holding one row per trophy: `{"id": n, "q": [q05, q25, q50, q75, q95]}`, each value a percentage between 0 and 100.

## How it works

- Each trophy's earn rate is tracked on the logit scale against log time since launch, with a recency-weighted, Huber-robust straight-line fit. Early reads from a few hundred tracked players are downweighted (they are heavily quantised), and Exophase refresh steps are treated as outliers.
- The fitted slope is shrunk by its own standard error, and the forecast is re-anchored on the last read, so a flat or noisy history means persistence, not a fall back to the historical mean.
- Growth is damped by how much it has already slowed: the slope of the last 24 hours is compared with the whole-history slope. A trophy still climbing at its old pace keeps its full slope; one whose recent slope has fallen keeps less than half of it. Launch-week adoption bends over as the first cohort finishes the early chapters and new buyers dilute the rates, and the damping is read off each trophy's own history.
- Story-chain coherence is enforced per quantile (a later chapter trophy cannot sit above the earlier one), a near-100% trophy is held near 100% instead of extrapolated, and per-trophy ceilings guard the platinum, the 100% trophy, the final mission and the chapter trophies against runaway extrapolation.
- Quantile spread combines a random-walk term over the horizon, slope uncertainty, fit residuals and a refresh allowance. The program is deterministic, uses only the Python standard library, and needs no network.

---

<sub>Entered on [make.fun](https://make.fun/?ref=frrbny3y&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
