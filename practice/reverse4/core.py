from config import reversConfig

def do_reverse(cfg : reversConfig) -> str:
    result = cfg.input[::-1]
    if cfg.upper:
        result = result.upper()
    return result