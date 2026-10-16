import argparse

from practice.reverse import result
# 创建参数解析器
p = argparse.ArgumentParser()
# 添加解析器参数
p.add_argument("--input",action="store",type=str,required=True,help="要倒叙的文本")
p.add_argument("--upper",action="store_true", help="是否把结果转成大写")
# 解析
args = p.parse_args()

# 真正要执行的函数
# 读取输入的参数
text = args.input
# 逻辑
result = text[::-1]
if args.upper:
    result = result.upper()
print(result)