# -*- coding: utf-8 -*-
"""
signal_records.csv 专项分析：
1. 统一时区，计算行情滞后 quote_lag_min = signal_time(UTC) - quote_synced_at(UTC)
2. 验证滞后与信号失效(is_valid=0)的关系，寻找拦截阈值
3. 用成本口径(失效75元/次, 拦截20元/次)估算拦截策略的净收益
输出: outputs/analysis_report.txt
"""
import os
import pandas as pd

COST_FAIL = 75        # 每次信号失效损失
COST_INTERCEPT = 20   # 每次陈旧行情拦截成本

df = pd.read_csv("signal_records.csv")
df = df.drop_duplicates()  # 57 行全列重复, 去重后 1133 行, record_id 唯一
out = []

def log(s=""):
    out.append(str(s))

log("=" * 70)
log("一、数据质量")
log("=" * 70)
log(f"原始行数: 1190, 去重后: {len(df)}")
dup_id = df["record_id"].duplicated().sum()
log(f"record_id 重复: {dup_id}")
full_dup = df.duplicated().sum()
log(f"完全重复行(全列): {full_dup}")
log(f"is_valid 缺失: {df['is_valid'].isna().sum()} ({df['is_valid'].isna().mean():.1%})")
log(f"is_valid 分布: 有效={int(df['is_valid'].sum())}, 失效={int((df['is_valid']==0).sum())}, "
    f"有效率(已知结果中)={df['is_valid'].mean():.2%}")

# ---- 时区统一 ----
df["signal_utc"] = pd.to_datetime(df["signal_time"], utc=True)
df["quote_utc"] = pd.to_datetime(df["quote_synced_at"], utc=True)
df["quote_lag_min"] = (df["signal_utc"] - df["quote_utc"]).dt.total_seconds() / 60

log("")
log("=" * 70)
log("二、行情滞后 quote_lag_min 分布（分钟, 已统一时区）")
log("=" * 70)
desc = df["quote_lag_min"].describe(percentiles=[.5, .75, .9, .95, .99])
for k, v in desc.items():
    log(f"  {k}: {v:.2f}")
neg = (df["quote_lag_min"] < 0).sum()
log(f"  滞后<0 (行情在信号之后, 异常): {neg}")

log("")
log("=" * 70)
log("三、滞后分箱 vs 信号有效率")
log("=" * 70)
known = df[df["is_valid"].notna()].copy()
bins = [-1, 1, 2, 3, 5, 10, 20, 60, 1e9]
labels = ["(0,1]", "(1,2]", "(2,3]", "(3,5]", "(5,10]", "(10,20]", "(20,60]", ">60"]
known["lag_bin"] = pd.cut(known["quote_lag_min"], bins=bins, labels=labels)
g = known.groupby("lag_bin", observed=True).agg(
    n=("is_valid", "size"),
    valid_rate=("is_valid", "mean"),
    fail_n=("is_valid", lambda s: int((s == 0).sum())),
)
log(g.to_string())

log("")
log("=" * 70)
log("四、阈值扫描: 若拦截 lag>阈值 的信号, 成本收益(仅对已知结果行估算)")
log("=" * 70)
log(f"口径: 失效损失 {COST_FAIL} 元/次, 拦截成本 {COST_INTERCEPT} 元/次")
log(f"{'阈值':>6} | {'拦截数':>6} | {'拦截中失效':>8} | {'挽回损失':>8} | {'拦截成本':>8} | {'净收益':>8}")
log("-" * 60)
for th in [1, 2, 3, 5, 10, 15, 20, 30]:
    sel = known[known["quote_lag_min"] > th]
    intercepted = len(sel)
    fails_avoided = int((sel["is_valid"] == 0).sum())
    saved = fails_avoided * COST_FAIL
    cost = intercepted * COST_INTERCEPT
    log(f"{th:>6} | {intercepted:>6} | {fails_avoided:>8} | {saved:>8} | {cost:>8} | {saved-cost:>8}")

log("")
log("=" * 70)
log("五、其他维度 vs 失效(已知结果行)")
log("=" * 70)
for col in ["cluster_node", "asset_class"]:
    log(f"--- {col} ---")
    g2 = known.groupby(col).agg(n=("is_valid", "size"), fail_rate=("is_valid", lambda s: 1 - s.mean()))
    g2 = g2.sort_values("fail_rate", ascending=False)
    log(g2.to_string())
    log("")

log("--- order_depth 分箱 ---")
known["depth_bin"] = pd.cut(known["order_depth"], [-1, 0, 3, 5, 10, 20, 100])
g3 = known.groupby("depth_bin", observed=True).agg(n=("is_valid", "size"), fail_rate=("is_valid", lambda s: 1 - s.mean()))
log(g3.to_string())
log("")
log("--- volume_24h 分箱 ---")
known["vol_bin"] = pd.cut(known["volume_24h"], [-1, 10, 30, 50, 70, 90, 200])
g4 = known.groupby("vol_bin", observed=True).agg(n=("is_valid", "size"), fail_rate=("is_valid", lambda s: 1 - s.mean()))
log(g4.to_string())
log("")
log("--- 失效 vs 有效两组的 lag 对比 ---")
log(known.groupby("is_valid")["quote_lag_min"].describe().to_string())

log("")
log("=" * 70)
log("六、缺失 is_valid 的 58 行画像")
log("=" * 70)
na = df[df["is_valid"].isna()]
log(f"滞后中位数: {na['quote_lag_min'].median():.1f} 分钟 (全体 {df['quote_lag_min'].median():.1f})")
log(f"lag>10min 占比: {(na['quote_lag_min']>10).mean():.1%} (全体 {(df['quote_lag_min']>10).mean():.1%})")
log(na["cluster_node"].value_counts().to_string())

report = "\n".join(out)
os.makedirs("outputs", exist_ok=True)
with open("outputs/analysis_report.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n")
print(report)

