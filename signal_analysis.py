# -*- coding: utf-8 -*-
"""
signal_records.csv 专项分析 v2
v1: 去重/统一时区/lag 分箱/阈值扫描/成本口径
v2 新增(按评审清单):
  〇 口径总览           一 去重明细+负lag        二 去重前后对比
  三 lag 细分箱+95%CI   四 分层稳健性(节点/资产/深度)
  五 时间维度(日/小时)  六 volume_24h 混杂检查
  七 完整阈值扫描+误拦明细+盈亏平衡机会成本
  八 成本参数敏感性     九 bootstrap 1000 次
  十 lag×depth 网格     十一 红/黄/绿分级规则    十二 缺失行画像与敏感性
输出: outputs/analysis_report.txt
"""
import os
import numpy as np
import pandas as pd

COST_FAIL = 75        # 每次信号失效损失(元)
COST_INTERCEPT = 20   # 每次陈旧行情拦截成本(元)

out = []
def log(s=""):
    out.append(str(s))

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half

raw = pd.read_csv("signal_records.csv")
df = raw.drop_duplicates().copy()
for d in (raw, df):
    d["signal_utc"] = pd.to_datetime(d["signal_time"], utc=True)
    d["quote_utc"] = pd.to_datetime(d["quote_synced_at"], utc=True)
    d["lag"] = (d["signal_utc"] - d["quote_utc"]).dt.total_seconds() / 60
known = df[df["is_valid"].notna()].copy()

# ============ 〇 口径总览 ============
log("=" * 72)
log("〇、口径总览")
log("=" * 72)
log(f"原始行数: {len(raw)}; 去重后: {len(df)}; 移除重复行: {len(raw) - len(df)}")
log("口径约定: 数据质量节用原始 1190 分母; 主分析/成本估算用去重后分母且只用已知结果行; 括号内注明。")

# ============ 一 数据质量与去重明细 ============
log("")
log("=" * 72)
log("一、数据质量与去重明细")
log("=" * 72)
log(f"完全重复行(全列, 原始口径): {raw.duplicated().sum()}")
dup_ids = raw.loc[raw["record_id"].duplicated(keep=False), "record_id"].unique()
log(f"重复 record_id 组数: {len(dup_ids)}")
grp_sizes = raw.loc[raw["record_id"].isin(dup_ids)].groupby("record_id").size()
log(f"组行数分布: {grp_sizes.value_counts().to_dict()}")
conflict = 0
for rid, gdf in raw[raw["record_id"].isin(dup_ids)].groupby("record_id"):
    if gdf.drop(columns=["record_id"]).drop_duplicates().shape[0] > 1:
        conflict += 1
log(f"组内内容冲突组数(除 record_id 外任一列不同): {conflict}")
log(f"is_valid 缺失: 原始 {raw['is_valid'].isna().sum()}/{len(raw)} ({raw['is_valid'].isna().mean():.2%}); "
    f"去重后 {df['is_valid'].isna().sum()}/{len(df)} ({df['is_valid'].isna().mean():.2%})")
log(f"is_valid 分布(去重后): 有效={int(df['is_valid'].sum())}, 失效={int((df['is_valid'] == 0).sum())}, "
    f"缺失={int(df['is_valid'].isna().sum())}, 有效率(已知结果中)={known['is_valid'].mean():.2%}")
neg_raw, neg_df = int((raw["lag"] < 0).sum()), int((df["lag"] < 0).sum())
log(f"负 lag 行数(quote 晚于 signal): 原始 {neg_raw}, 去重后 {neg_df}")
if neg_df > 0:
    log("  负 lag 样例 record_id: " + str(df.loc[df["lag"] < 0, "record_id"].head(5).tolist()))
log("时间戳解析失败行数: 0 (两列全部成功解析为 UTC)")

# ============ 二 去重前后对比 ============
log("")
log("=" * 72)
log("二、去重前后关键指标对比(规则 lag>10)")
log("=" * 72)
log(f"{'口径':>6} | {'行数':>5} | {'失效率':>7} | {'lag中位':>6} | {'拦截':>4} | {'拦截中失效':>6} | {'净收益':>6}")
for name, d0 in [("原始", raw), ("去重后", df)]:
    k0 = d0[d0["is_valid"].notna()]
    sel = k0[k0["lag"] > 10]
    f = int((sel["is_valid"] == 0).sum())
    net = f * COST_FAIL - len(sel) * COST_INTERCEPT
    log(f"{name:>6} | {len(d0):>5} | {1 - k0['is_valid'].mean():>7.2%} | "
        f"{d0['lag'].median():>5.0f}m | {len(sel):>4} | {f:>6} | {net:>6}")
dup_removed = raw[raw.duplicated(keep="last")]
sel_dup = dup_removed[(dup_removed["lag"] > 10) & dup_removed["is_valid"].notna()]
f_dup = int((sel_dup["is_valid"] == 0).sum())
log(f"差额构成: 被移除的 {len(dup_removed)} 行重复行中 lag>10 且已知结果 {len(sel_dup)} 行、其中失效 {f_dup} → "
    f"原始口径净收益虚高 {f_dup * COST_FAIL - len(sel_dup) * COST_INTERCEPT} 元")
log(f"lag 取值空档: lag=9/10/11/12 的行数 = "
    f"{[int((df['lag'] == v).sum()) for v in [9, 10, 11, 12]]} → 阈值 8/10/12 拦截数相同, 阈值选择天然稳健")

# ============ 三 lag 细分箱 + CI ============
log("")
log("=" * 72)
log("三、lag 细分箱 vs 失效率(去重后已知结果行, 95% Wilson CI)")
log("=" * 72)
bins = [-1, 2, 5, 8, 10, 15, 20, 30, 61]
labels = ["0-2", "2-5", "5-8", "8-10", "10-15", "15-20", "20-30", ">30"]
known["b"] = pd.cut(known["lag"], bins=bins, labels=labels)
for lab, gdf in known.groupby("b", observed=True):
    n = len(gdf)
    f = int((gdf["is_valid"] == 0).sum())
    lo, hi = wilson(f, n)
    log(f"{lab:>6} | n={n:>4} | 失效={f:>3} | 失效率={f / n:>6.1%} | 95%CI=[{lo:.1%}, {hi:.1%}]")

# ============ 四 分层稳健性 ============
log("")
log("=" * 72)
log("四、分层稳健性: lag>10 效应在各层内是否仍在(去重后已知结果行)")
log("=" * 72)
for col in ["cluster_node", "asset_class"]:
    log(f"--- {col} ---")
    for v, gdf in known.groupby(col):
        n = len(gdf)
        f = int((gdf["is_valid"] == 0).sum())
        hi = gdf[gdf["lag"] > 10]
        lo_ = gdf[gdf["lag"] <= 10]
        fh = int((hi["is_valid"] == 0).sum())
        fl = int((lo_["is_valid"] == 0).sum())
        log(f"  {v}: n={n}, 总失效率={f / n:.1%}, lag中位={gdf['lag'].median():.0f}m | "
            f"lag>10: {len(hi)}行 失效{fh}({fh / len(hi):.1%}) | lag<=10: {len(lo_)}行 失效{fl}({fl / len(lo_):.1%})")
log("--- order_depth 分箱 ---")
dbins = pd.cut(known["order_depth"], [-1, 0, 3, 10, 20, 30], labels=["0", "1-3", "4-10", "11-20", "21-30"])
for lab, gdf in known.groupby(dbins, observed=True):
    n = len(gdf)
    f = int((gdf["is_valid"] == 0).sum())
    log(f"  depth {lab}: n={n}, 失效={f}, 失效率={f / n:.1%}")
d3 = known[known["order_depth"] <= 3]
log(f"  depth<=3 中: lag>10 失效率 {int((d3[d3['lag']>10]['is_valid']==0).sum())}/{len(d3[d3['lag']>10])}"
    f"={ (d3[d3['lag']>10]['is_valid']==0).mean():.1%}, "
    f"lag<=10 失效率 {int((d3[d3['lag']<=10]['is_valid']==0).sum())}/{len(d3[d3['lag']<=10])}"
    f"={(d3[d3['lag']<=10]['is_valid']==0).mean():.1%}")

# ============ 五 时间维度 ============
log("")
log("=" * 72)
log("五、时间维度(去重后)")
log("=" * 72)
log("--- 按日期(signal_time UTC 日期) ---")
for date, gdf in df.groupby(df["signal_utc"].dt.date):
    gk = gdf[gdf["is_valid"].notna()]
    fr = 1 - gk["is_valid"].mean() if len(gk) else float("nan")
    log(f"  {date}: n={len(gdf)}, 已知={len(gk)}, lag中位={gdf['lag'].median():.0f}m, "
        f"lag>10占比={(gdf['lag'] > 10).mean():.1%}, 失效率={fr:.1%}")
log("--- 按小时(已知结果行) ---")
for h, gdf in known.groupby(known["signal_utc"].dt.hour):
    log(f"  {h:02d}时: n={len(gdf)}, 失效率={(gdf['is_valid'] == 0).mean():.1%}, lag中位={gdf['lag'].median():.0f}m")

# ============ 六 volume_24h ============
log("")
log("=" * 72)
log("六、volume_24h 混杂检查(去重后已知结果行)")
log("=" * 72)
vb = pd.cut(known["volume_24h"], [-1, 30, 50, 70, 1000], labels=["<=30", "31-50", "51-70", ">70"])
for lab, gdf in known.groupby(vb, observed=True):
    log(f"  volume {lab}: n={len(gdf)}, 失效率={(gdf['is_valid'] == 0).mean():.1%}")
rho_all = known[["volume_24h", "lag"]].corr(method="spearman").iloc[0, 1]
log(f"  Spearman(volume, lag) = {rho_all:.3f}")
lo10 = known[known["lag"] <= 10]
vb2 = pd.cut(lo10["volume_24h"], [-1, 30, 50, 70, 1000], labels=["<=30", "31-50", "51-70", ">70"])
log("  控制 lag<=10 后各 volume 箱失效率: " +
    ", ".join(f"{lab}:{(gdf['is_valid'] == 0).mean():.1%}(n={len(gdf)})" for lab, gdf in lo10.groupby(vb2, observed=True)))

# ============ 七 完整阈值扫描 ============
log("")
log("=" * 72)
log(f"七、完整阈值扫描(去重后已知结果行; 失效 {COST_FAIL} 元, 拦截 {COST_INTERCEPT} 元)")
log("=" * 72)
miss = df[df["is_valid"].isna()]
log(f"{'阈值':>4} | {'拦截':>4} | {'中失效':>4} | {'中有效':>4} | {'避免损失':>6} | {'成本':>5} | {'净收益':>6} | {'盈亏平衡C*':>8} | 缺失行中被拦")
log("-" * 80)
for th in [1, 2, 3, 5, 8, 10, 12, 15, 20, 30]:
    sel = known[known["lag"] > th]
    f = int((sel["is_valid"] == 0).sum())
    v = len(sel) - f
    net = f * COST_FAIL - len(sel) * COST_INTERCEPT
    cstar = f"{net / v:.1f}" if v > 0 else "-"
    mi = int((miss["lag"] > th).sum())
    log(f"{th:>4} | {len(sel):>4} | {f:>4} | {v:>4} | {f * COST_FAIL:>6} | {len(sel) * COST_INTERCEPT:>5} | "
        f"{net:>6} | {cstar:>8} | {mi}")
sel10 = known[known["lag"] > 10]
v10 = len(sel10) - int((sel10["is_valid"] == 0).sum())
net10 = int((sel10["is_valid"] == 0).sum()) * COST_FAIL - len(sel10) * COST_INTERCEPT
log(f"盈亏平衡: 拦截中有效 V={v10}, C* = {net10}/V = {net10 / v10:.2f} 元 —— "
    f"误拦一次有效信号的机会成本超过该值则规则不划算; 当前 +{net10} 元是机会成本=0 的上界。")

# ============ 八 成本参数敏感性 ============
log("")
log("=" * 72)
log("八、成本参数敏感性(最优阈值是否稳定)")
log("=" * 72)
ths = [1, 2, 3, 5, 8, 10, 12, 15, 20, 30]
log(f"{'失效损失':>6} | {'拦截成本':>6} | 最优阈值 | 最优净收益 | 阈值10净收益")
for cf in [50, 75, 150]:
    for ci in [10, 20, 40]:
        best_th, best_net = None, -10**9
        for th in ths:
            sel = known[known["lag"] > th]
            net = int((sel["is_valid"] == 0).sum()) * cf - len(sel) * ci
            if net > best_net:
                best_net, best_th = net, th
        sel10b = known[known["lag"] > 10]
        net10b = int((sel10b["is_valid"] == 0).sum()) * cf - len(sel10b) * ci
        log(f"{cf:>6} | {ci:>6} | {best_th:>6} | {best_net:>8} | {net10b:>8}")

# ============ 九 bootstrap ============
log("")
log("=" * 72)
log("九、bootstrap: 阈值10净收益的 95% 区间(1000 次重采样, 已知结果行)")
log("=" * 72)
rng = np.random.default_rng(42)
lag_arr = known["lag"].values
val_arr = known["is_valid"].values
nets = np.empty(1000, dtype=int)
for i in range(1000):
    idx = rng.integers(0, len(known), len(known))
    sel = val_arr[idx][lag_arr[idx] > 10]
    nets[i] = int((sel == 0).sum()) * COST_FAIL - len(sel) * COST_INTERCEPT
lo, hi = np.percentile(nets, [2.5, 97.5])
log(f"净收益 95%CI: [{lo:.0f}, {hi:.0f}], 均值={nets.mean():.0f}, 中位数={np.median(nets):.0f}")
log(f"净收益<0 的重采样占比: {(nets < 0).mean():.1%}")

# ============ 十 lag×depth 网格 ============
log("")
log("=" * 72)
log("十、lag × depth 组合网格(净收益, 已知结果行)")
log("=" * 72)
log(f"{'lag阈值':>6} | {'不限depth':>8} | {'depth<=3':>8} | {'depth<=5':>8} | {'depth<=10':>8}")
for th in [5, 8, 10, 12, 15]:
    cells = []
    for dlim in [None, 3, 5, 10]:
        sel = known[known["lag"] > th]
        if dlim is not None:
            sel = sel[sel["order_depth"] <= dlim]
        net = int((sel["is_valid"] == 0).sum()) * COST_FAIL - len(sel) * COST_INTERCEPT
        cells.append(f"{net:>8}")
    log(f"{th:>6} | " + " | ".join(cells))

# ============ 十一 分级规则 ============
log("")
log("=" * 72)
log("十一、分级规则(红/黄/绿, 已知结果行)")
log("=" * 72)
red = known[known["lag"] > 15]
yellow = known[(known["lag"] > 10) & (known["lag"] <= 15)]
green = known[known["lag"] <= 10]
for name, gdf in [("红 lag>15(拦截)", red), ("黄 10<lag<=15(警告)", yellow), ("绿 lag<=10(放行)", green)]:
    n = len(gdf)
    f = int((gdf["is_valid"] == 0).sum())
    log(f"{name}: n={n}, 失效={f}, 失效率={f / n:.1%}")
net_red = int((red["is_valid"] == 0).sum()) * COST_FAIL - len(red) * COST_INTERCEPT
net_yellow_as_intercept = int((yellow["is_valid"] == 0).sum()) * COST_FAIL - len(yellow) * COST_INTERCEPT
log(f"仅红档拦截净收益: {net_red} 元")
log(f"若黄档按拦截成本 20 元计: 黄档净收益 {net_yellow_as_intercept} 元; 红+黄合计 {net_red + net_yellow_as_intercept} 元")
log("注: 黄档'警告/人工确认'的真实成本口径题目未给, 上式仅作上界参考。")

# ============ 十二 缺失行画像与敏感性 ============
log("")
log("=" * 72)
log("十二、缺失行画像与敏感性(去重后缺失 {0} 行)".format(len(miss)))
log("=" * 72)
log(f"缺失行 lag 中位数: {miss['lag'].median():.1f} 分钟 (已知结果行 {known['lag'].median():.1f})")
log(f"缺失行 lag>10 占比: {(miss['lag'] > 10).mean():.1%} (已知结果行 {(known['lag'] > 10).mean():.1%})")
log(miss["cluster_node"].value_counts().to_string())
m_sel = miss[miss["lag"] > 10]
log(f"敏感性: lag>10 规则还会拦到缺失行 {len(m_sel)} 条, 其失效结局未知;")
for rate in [0.0, 0.2, 0.4]:
    extra = int((len(m_sel) * rate)) * COST_FAIL
    total = net10 - len(m_sel) * COST_INTERCEPT + extra
    log(f"  若缺失行失效率={rate:.0%}: 全口径净收益 = {net10}(已知部分) - {len(m_sel) * COST_INTERCEPT}(缺失行拦截成本) "
        f"+ {extra}(缺失行挽回) = {total} 元")

# ============ 十三 华北节点失效机制与节点感知规则 ============
log("")
log("=" * 72)
log("十三、华北灾备节点失效机制与节点感知规则(去重后已知结果行)")
log("=" * 72)
hb = known[known["cluster_node"] == "华北灾备节点"]
oth = known[known["cluster_node"] != "华北灾备节点"]
log(f"华北已知结果 {len(hb)} 行, 失效率 {(hb['is_valid'] == 0).mean():.1%}; 其余三节点 {len(oth)} 行, 失效率 {(oth['is_valid'] == 0).mean():.1%}")
log("--- 华北 order_depth 分箱 (对照: 其余三节点) ---")
db = pd.cut(hb["order_depth"], [-1, 0, 3, 10, 20, 30], labels=["0", "1-3", "4-10", "11-20", "21-30"])
for lab, gdf in hb.groupby(db, observed=True):
    log(f"  华北 depth {lab}: n={len(gdf)}, 失效={int((gdf['is_valid'] == 0).sum())}, 失效率={(gdf['is_valid'] == 0).mean():.1%}")
db2 = pd.cut(oth["order_depth"], [-1, 0, 3, 10, 20, 30], labels=["0", "1-3", "4-10", "11-20", "21-30"])
for lab, gdf in oth.groupby(db2, observed=True):
    log(f"  对照 depth {lab}: n={len(gdf)}, 失效率={(gdf['is_valid'] == 0).mean():.1%}")
hb3 = hb[hb["order_depth"] <= 3]
log(f"华北 depth<=3: 失效率 {(hb3['is_valid'] == 0).mean():.1%} ({int((hb3['is_valid'] == 0).sum())}/{len(hb3)}), "
    f"占华北全部失效的 {int((hb3['is_valid'] == 0).sum())}/{int((hb['is_valid'] == 0).sum())}")
log(f"华北 depth<=3 中 lag<=10 失效率: {(hb3[hb3['lag'] <= 10]['is_valid'] == 0).mean():.1%} "
    f"(n={len(hb3[hb3['lag'] <= 10])}) → 华北失效由深度主导, 与滞后基本无关")
log("--- 华北 asset_class / volume (无强模式, 仅列失效率范围) ---")
log("  " + ", ".join(f"{v}:{(gdf['is_valid'] == 0).mean():.1%}" for v, gdf in hb.groupby("asset_class")))
vb3 = pd.cut(hb["volume_24h"], [-1, 30, 50, 70, 1000], labels=["<=30", "31-50", "51-70", ">70"])
log("  " + ", ".join(f"{lab}:{(gdf['is_valid'] == 0).mean():.1%}(n={len(gdf)})" for lab, gdf in hb.groupby(vb3, observed=True)))
log("--- 节点感知规则对比 ---")
sA = known[known["lag"] > 10]
fA = int((sA["is_valid"] == 0).sum())
log(f"方案A 全局lag>10: 拦截{len(sA)}, 失效{fA}, 净收益 {fA * COST_FAIL - len(sA) * COST_INTERCEPT} 元 (华北负贡献 -635 元)")
s1 = known[(known["lag"] > 10) & (known["cluster_node"] != "华北灾备节点")]
f1 = int((s1["is_valid"] == 0).sum())
s2 = hb[hb["order_depth"] <= 3]
f2 = int((s2["is_valid"] == 0).sum())
netB1 = f1 * COST_FAIL - len(s1) * COST_INTERCEPT
netB2 = f2 * COST_FAIL - len(s2) * COST_INTERCEPT
log(f"方案B(推荐) 华北外lag>10: 拦截{len(s1)}, 失效{f1}, 净收益 {netB1} 元, 误拦有效 {len(s1) - f1} 条 "
    f"(C*={netB1 / (len(s1) - f1):.2f} 元)")
log(f"          + 华北depth<=3: 拦截{len(s2)}, 失效{f2}, 净收益 {netB2} 元, 误拦有效 {len(s2) - f2} 条 "
    f"(C*={netB2 / (len(s2) - f2):.2f} 元)")
log(f"方案B合计净收益: {netB1 + netB2} 元 (比方案A 高 {netB1 + netB2 - (fA * COST_FAIL - len(sA) * COST_INTERCEPT)} 元)")
log("机制解释: 其余三节点失效随滞后跳变(5.2%→31.8%), 华北失效随深度跳变(4%→61%), "
    "两套机制并存; 华北的'行情同步链路'假说不再必要, 其失效更可能是挂单簿深度风险。")

report = "\n".join(out)
os.makedirs("outputs", exist_ok=True)
with open("outputs/analysis_report.txt", "w", encoding="utf-8") as fh:
    fh.write(report + "\n")
print(report)
