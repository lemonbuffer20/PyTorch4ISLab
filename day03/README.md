# Day 3: Prepare training

## What to reproduce:
* `torch4is/utils.py` - `wandb_setup`
* `my_optim/sgd.py`
* `my_optim/build.py`
* `my_optim/my_sched/cosine.py`
* `my_optim/my_sched/build.py`
* `day03/config.json`
* `day03/wandb_logging.py`

## Key

1. You should know how to make optimizer and scheduler.
2. You should be familiar with configuration.
3. You should be familiar with logging to WandB.

## Configuration

1. Every experiment should be easily TUNABLE and REPRODUCIBLE.
2. Best is to separate configuration to JSON/YAML configuration file.
3. Although additional effort is required, it is really important to write papers.

## WandB

1. There are 3 modes in WandB.
    * Online - track online
    * Offline - do not track, don't use.
    * Disabled - track offline
2. `wandb.log` can log any values.
3. WandB tracks config file, console output, and files (if exist).
