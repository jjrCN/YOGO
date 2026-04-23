

def parse_cfg(cfg):
    lp = SimpleNamespace(**cfg.get('model_params', {}))
    op = SimpleNamespace(**cfg.get('optim_params', {}))
    pp = SimpleNamespace(**cfg.get('pipeline_params', {}))
    return lp, op, pp