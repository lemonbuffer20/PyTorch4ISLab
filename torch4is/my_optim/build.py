from torch4is.my_optim.sgd import MySGD
from torch4is.my_optim.adam import MyAdam


def build_optimizer(cfg: dict, parameters):
    try:
        name = cfg["name"].lower()
    except KeyError:
        raise KeyError(f"[ERROR:OPTIM] Optimizer config don't have name inside.")

    if name == "sgd":
        optimizer = MySGD(parameters,
                          lr=cfg.get("lr"),
                          momentum=cfg.get("momentum", 0.0),
                          weight_decay=cfg.get("weight_decay", 0.0),
                          nesterov=cfg.get("nesterov", False))
    elif name == "adam":
        optimizer = MyAdam(parameters,
                           lr=cfg.get("lr"),
                           betas=tuple(cfg.get("betas", (0.9, 0.999))),
                           eps=cfg.get("eps", 1e-8),
                           weight_decay=cfg.get("weight_decay", 0.0))
    else:
        raise ValueError(f"[ERROR:OPTIM] Unexpected optimizer type {cfg['name']}.")
    return optimizer
