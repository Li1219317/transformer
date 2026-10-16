import argparse
from config import reversConfig

p = argparse.ArgumentParser()
p.add_argument( "--input", type = str  )
p.add_argument( "--upper", action = "store_true" )
p.add_argument( "--repeat", type = int )

cfg = reverseConfig()
for k, v in vars(args).items():
    if v is not None and hasattr(cfg, k):
        setattr(cfg, k, v)
