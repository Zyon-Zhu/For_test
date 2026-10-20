# -*- coding: utf-8 -*-
"""
通用数据分析脚本骨架。

用法:
    python analyze.py data/xxx.csv [--out outputs/summary.txt]

开始分析后只改两处:
    1. COL_* 常量 —— 换成你数据里的真实列名
    2. main() 里标注了 [业务逻辑] 的段落 —— 换成你自己的指标计算

铁律: README 里写的每一个数字, 都必须能由本脚本的输出复算出来。
"""
import argparse
import os
import sys

import pandas as pd

# ---- 列名映射: 换成真实数据的列名, 其余代码不动 ----
COL_DATE = None   # 例: "日期"
COL_VALUE = None  # 例: "销售额"
COL_GROUP = None  # 例: "渠道"


def load(path: str) -> pd.DataFrame:
    """读取数据。CSV 编码不对时按需加 encoding='gbk' 或 sep='\t'。"""
    df = pd.read_csv(path)
    return df


def overview(df: pd.DataFrame) -> str:
    """基础概况: 形状 / 列名 / 缺失 / 数值摘要。任何数据集都能直接跑。"""
    lines = []
    lines.append(f"行数: {len(df)}")
    lines.append(f"列数: {df.shape[1]}")
    lines.append(f"列名: {list(df.columns)}")

    dup = df.duplicated().sum()
    lines.append(f"完全重复行: {dup}")

    na = df.isna().sum()
    na = na[na > 0]
    if len(na) == 0:
        lines.append("缺失值: 无")
    else:
        lines.append("缺失值:")
        for col, n in na.items():
            lines.append(f"  {col}: {n} ({n / len(df):.1%})")

    lines.append("数值列摘要:")
    lines.append(df.describe(include="all").T.to_string())
    return "\n".join(lines)


def metrics(df: pd.DataFrame) -> str:
    """[业务逻辑] 关键指标计算 —— 按你的分析目标改写这一段。"""
    lines = []
    lines.append("关键指标:")
    # 示例指标(所有数值列的总和与均值), 替换成真实指标:
    for col in df.select_dtypes("number").columns:
        lines.append(f"  {col} | 总和={df[col].sum()} | 均值={df[col].mean():.4g} | 中位数={df[col].median():.4g}")

    # 示例: 按 COL_GROUP 分组求和, 用到时取消注释并改列名:
    # g = df.groupby(COL_GROUP)[COL_VALUE].agg(["count", "sum", "mean"])
    # lines.append("分组汇总:")
    # lines.append(g.to_string())
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="通用分析脚本骨架")
    ap.add_argument("csv", help="输入 CSV 路径, 例: data/xxx.csv")
    ap.add_argument("--out", default=None, help="同时把结果写入该文件, 例: outputs/summary.txt")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"错误: 找不到输入文件 {args.csv}", file=sys.stderr)
        return 1

    df = load(args.csv)
    report = "\n\n".join([overview(df), metrics(df)])
    print(report)

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n已写入: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
