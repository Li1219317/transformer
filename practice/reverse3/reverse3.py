import argparse
from config import reversConfig
# 初始化参数解释器
p = argparse.ArgumentParser()
#添加解析器参数
args = p.parse_args()
# 添加解析器参数
p.add_argument( "--input", type = str  )
p.add_argument( "--upper", action = "store_true" )
p.add_argument( "--repeat", type = int )
#调用一个名为 reversConfig 的函数（或类构造器），返回一个配置对象，赋值给 cfg。
# 就是一个对象，把上面那些参数变成它的属性：
cfg = reversConfig()
# 对于一个在参数args的键值对。先将参数转变为字典，再将字典转变为键值对？
for k, v in vars(args).items():
    if v is not None and hasattr(cfg, k):
        setattr(cfg, k, v)

result = cfg.input[::-1]
if args.upper:
    result = result.upper()
for _ in range(args.repeat):
    print(result)