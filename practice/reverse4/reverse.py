import argparse
from config import reversConfig
from core import do_reverse
def main():
    p  = argparse.ArgumentParser(description='Reverse a string')

    p.add_argument("--input", type=str)
    p.add_argument("--upper", action="store_true", default=None)
    p.add_argument("--repeat", type=int)

    args = p.parse_args()

    cfg = reversConfig()

    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    result = do_reverse(cfg )
    for _ in range(cfg.repeat):
        print(result)

if __name__ == '__main__':
    main()