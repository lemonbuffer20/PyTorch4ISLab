from torch4is.my_optim.my_sched.cosine import MyCosineLRScheduler


def build_scheduler(cfg: dict, optimizer):
    try:
        name = cfg["name"].lower()
    except KeyError:
        raise KeyError(f"[ERROR:SCHED] Scheduler config don't have name inside.")

    if (name == "cos") or (name == "cosine"):
        scheduler = MyCosineLRScheduler(optimizer,
                                        max_steps=cfg.get("max_steps"),
                                        warmup_steps=cfg.get("warmup_steps", 0),
                                        min_lr=cfg.get("min_lr", 1e-8),
                                        mode=cfg.get("mode", "min"))
    else:
        raise ValueError(f"[ERROR:SCHED] Unexpected scheduler type {cfg['name']}.")
    return scheduler
