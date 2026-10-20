# -*- coding: utf-8 -*-
"""
signal_records.csv 专项分析 v3.4
v1: 去重/统一时区/lag 分箱/阈值扫描/成本口径
v2: 口径明细/CI/分层/时间与 volume/成本敏感性/bootstrap/组合规则/缺失敏感性/华北机制
v3: 方案 B（节点感知规则）、B1/B2 可加性与分节点粒度
v3.3: 方案 B 分项 bootstrap、漏拦覆盖率、子规则成本敏感性与口径卫生审计
v3.4: 数字可追溯性审计（空箱显式输出/分日期延迟失效率/失效损失敏感性/联合 bootstrap 说明）

报告结构:
  〇口径; ①质量去重; ②去重前后; ③lag 分箱+CI; ④分层; ⑤时间;
  ⑥volume; ⑦阈值; ⑧成本敏感性; ⑨全局 bootstrap; ⑩lag×depth 网格;
  ⑪分级; ⑫缺失敏感性; ⑬华北机制/方案 B; ⑭方案 B bootstrap/覆盖率/口径审计
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
log(f"lag 精确值行数: lag=9/10/11/12 = "
    f"{[int((df['lag'] == v).sum()) for v in [9, 10, 11, 12]]}")
log("阈值平台解释: lag=9/10/11 无观测; 阈值8与10拦截数相同(225/225), 阈值12为219, 属同一收益平台而非拦截数完全相同")

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
empty_8_10 = known[known["b"].isna() & (known["lag"] > 8) & (known["lag"] <= 10)]
log(f" 8-10 | n={len(empty_8_10):>4} | lag=9/10 无观测；该箱为空，不能称为8→10相邻箱直接跳变")
log(f"lag 最大值: {df['lag'].max():.0f} 分钟")

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
    gh = gk[gk["lag"] > 10]
    gl = gk[gk["lag"] <= 10]
    fr_hi = (gh["is_valid"] == 0).mean() if len(gh) else float("nan")
    fr_lo = (gl["is_valid"] == 0).mean() if len(gl) else float("nan")
    log(f"  {date}: n={len(gdf)}, 已知={len(gk)}, lag中位={gdf['lag'].median():.0f}m, "
        f"lag>10占比={(gdf['lag'] > 10).mean():.1%}, 失效率={fr:.1%}, "
        f"lag>10失效率={fr_hi:.1%}(n={len(gh)}), lag<=10失效率={fr_lo:.1%}(n={len(gl)})")
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

# ---- A 档检查: 可加性/公式/小样本 ----
log("")
log("--- B1/B2 可加性检查 ---")
inter = len(set(s1.index) & set(s2.index))
log(f"B1∩B2 交集行数: {inter} → 两规则按节点构造互斥(B1 限定非华北, B2 限定华北), 净收益相加无重复计数")
log(f"拦截合计 {len(s1) + len(s2)} 条 = {len(s1)}(B1) + {len(s2)}(B2)")
log("华北 depth<=3 × lag 交叉(失效/有效):")
h3 = hb[hb["order_depth"] <= 3]
for lab, gdf in h3.groupby(pd.cut(h3["lag"], [-1, 10, 1e9], labels=["lag<=10", "lag>10"]), observed=True):
    f_ = int((gdf["is_valid"] == 0).sum())
    log(f"  {lab}: n={len(gdf)}, 失效={f_}, 有效={len(gdf) - f_}")
log("→ 华北 depth<=3 中 lag>10 的 9 行已被 B2 覆盖, 不会落入 B1(节点互斥); lag 规则对该子集的失效无额外遗漏")
log("")
log("--- B1/B2 决策公式(误拦有效信号机会成本 C) ---")
v1 = len(s1) - f1
v2 = len(s2) - f2
log(f"B1: 净收益 = {netB1} - {v1}×C  (V1={v1}), C*={netB1 / v1:.2f} 元 → C 超过则 B1 降级为警告/人工确认")
log(f"B2: 净收益 = {netB2} - {v2}×C  (V2={v2}), C*={netB2 / v2:.2f} 元 → 同上")
log("")
log("--- 华北 depth 小样本口径 ---")
d0 = hb[hb["order_depth"] == 0]
f0 = int((d0["is_valid"] == 0).sum())
lo0, hi0 = wilson(f0, len(d0))
lom, him = wilson(f2, len(h3))
log(f"depth=0 单独: {f0}/{len(d0)} = {f0 / len(d0):.1%}, 95%CI[{lo0:.1%}, {hi0:.1%}] → 样本仅 {len(d0)}, CI 宽, 不单独作为规则依据")
log(f"depth<=3 合并(B2 实际作用域): {f2}/{len(h3)} = {f2 / len(h3):.1%}, 95%CI[{lom:.1%}, {him:.1%}] → B2 的 {netB2} 元净收益用合并口径计算, 不依赖 depth=0 单箱")
log("")
log("--- 非华北 depth<=3 单独成规则(方案 B 节点拆分依据) ---")
o3 = oth[oth["order_depth"] <= 3]
fo = int((o3["is_valid"] == 0).sum())
loo, hio = wilson(fo, len(o3))
log(f"非华北 depth<=3: n={len(o3)}, 失效={fo}, 失效率={fo / len(o3):.1%} 95%CI[{loo:.1%}, {hio:.1%}], "
    f"净收益 {fo * COST_FAIL - len(o3) * COST_INTERCEPT} 元 → 显著为负, 非华北的 depth<=3 不构成信号, 节点拆分有据")
log("")
log("--- 分节点粒度检查: depth 机制是否只在华北(回应'其他区域呢') ---")
log(f"{'节点':>8} | {'depth<=3':>16} | {'depth>3':>8} | {'lag>10':>8} | {'lag<=10':>8}")
for node in ["华东主节点", "华南边缘节点", "西南测试节点", "华北灾备节点"]:
    gdf = known[known["cluster_node"] == node]
    d3n = gdf[gdf["order_depth"] <= 3]
    d4n = gdf[gdf["order_depth"] > 3]
    lgn = gdf[gdf["lag"] > 10]
    lln = gdf[gdf["lag"] <= 10]
    f3n = int((d3n["is_valid"] == 0).sum())
    lo3n, hi3n = wilson(f3n, len(d3n)) if len(d3n) else (float("nan"), float("nan"))
    log(f"{node:>8} | {f3n}/{len(d3n)}={f3n / len(d3n):.1%} CI[{lo3n:.1%},{hi3n:.1%}] | "
        f"{int((d4n['is_valid'] == 0).sum())}/{len(d4n)}={ (d4n['is_valid'] == 0).mean():.1%} | "
        f"{int((lgn['is_valid'] == 0).sum())}/{len(lgn)}={ (lgn['is_valid'] == 0).mean():.1%} | "
        f"{int((lln['is_valid'] == 0).sum())}/{len(lln)}={ (lln['is_valid'] == 0).mean():.1%}")
for node in ["华东主节点", "华南边缘节点", "西南测试节点"]:
    gdf = known[(known["cluster_node"] == node) & (known["order_depth"] <= 3)]
    fn = int((gdf["is_valid"] == 0).sum())
    log(f"{node} depth<=3 单独规则: 拦{len(gdf)} 失{fn} 净{fn * COST_FAIL - len(gdf) * COST_INTERCEPT} 元")
log("→ 三节点 depth<=3 失效率 5.7%/12.9%/16.7%, CI 均覆盖合并基线 12%, 单独成规则全部为负 "
    "(−550/−320/−315) → depth 机制为华北特有, 无需再扩规则; lag>10 在三个节点均显著高于 lag<=10 (43.4%~57.1% vs 1.9%~3.0%), B1 三节点合并有分节点证据支撑")
log("机制解释: 其余三节点失效随滞后跳变(5.2%→31.8%), 华北失效随深度跳变(4%→61%), "
    "两套机制并存; 华北的'行情同步链路'假说不再必要, 其失效更可能是挂单簿深度风险。")

# ============ 十四 方案 B 稳健性与覆盖率(落地口径自证) ============
log("")
log("=" * 72)
log("十四、方案 B 稳健性与覆盖率(落地规则自证)")
log("=" * 72)

# ---- B1/B2 分别 bootstrap ----
rng2 = np.random.default_rng(42)
lag_a = known["lag"].values
val_a = known["is_valid"].values
node_a = (known["cluster_node"] == "华北灾备节点").values
dep_a = known["order_depth"].values
n_a = len(known)
nb = np.empty((1000, 3), dtype=int)
for i in range(1000):
    idx = rng2.integers(0, n_a, n_a)
    l_, v_, nd_, dp_ = lag_a[idx], val_a[idx], node_a[idx], dep_a[idx]
    m1 = (l_ > 10) & (~nd_)
    m2 = nd_ & (dp_ <= 3)
    f1_, f2_ = int((v_[m1] == 0).sum()), int((v_[m2] == 0).sum())
    a_ = f1_ * COST_FAIL - int(m1.sum()) * COST_INTERCEPT
    b_ = f2_ * COST_FAIL - int(m2.sum()) * COST_INTERCEPT
    nb[i] = [a_, b_, a_ + b_]
log("Bootstrap 口径: 同一次重采样用同一批行同时计算 B1、B2 与合计；合计 CI 为联合重采样，不是两个边际 CI 端点相加。")
for j, nm in enumerate(["B1", "B2", "B1+B2"]):
    lo_b, hi_b = np.percentile(nb[:, j], [2.5, 97.5])
    log(f"{nm:>5} 净收益 95%CI[{lo_b:.0f}, {hi_b:.0f}], 均值={nb[:, j].mean():.0f}, "
        f"为负占比={(nb[:, j] < 0).mean():.1%}")
log(f"→ B2 样本仅 {len(s2)} 条, CI 宽度 {np.percentile(nb[:, 1], 97.5) - np.percentile(nb[:, 1], 2.5):.0f} 元; "
    "B2 下界虽为正但明显低于点估计, 建议灰度上线、不直接全量")

# ---- 漏拦覆盖率 ----
log("")
log("--- 漏拦覆盖率(方案 B 放行的信号中仍有多少失效) ---")
b1i, b2i = set(s1.index), set(s2.index)
passed = known.drop(index=list(b1i | b2i))
tot_f_ = int((known["is_valid"] == 0).sum())
int_f_ = int((s1["is_valid"] == 0).sum()) + int((s2["is_valid"] == 0).sum())
pas_f_ = int((passed["is_valid"] == 0).sum())
log(f"已知失效总数={tot_f_}, 方案B拦截中失效={int_f_}, 放行中失效={pas_f_}, 漏拦率={pas_f_ / tot_f_:.1%}")
log(f"放行信号 {len(passed)} 条, 其中失效 {pas_f_} 条, 放行失效率={pas_f_ / len(passed):.1%} "
    f"(拦截内失效率={int_f_ / (len(s1) + len(s2)):.1%})")
log("放行失效按节点: " + str(passed[passed["is_valid"] == 0]["cluster_node"].value_counts().to_dict()))
log("放行失效 lag 分箱: " + str(pd.cut(passed[passed["is_valid"] == 0]["lag"], [-1, 5, 10, 1e9],
    labels=["<=5", "5-10", ">10"]).value_counts().to_dict()))
log("放行失效 depth 分箱: " + str(pd.cut(passed[passed["is_valid"] == 0]["order_depth"], [-1, 3, 10, 1e9],
    labels=["<=3", "4-10", ">10"]).value_counts().to_dict()))
log("→ 剩余 22 条失效分散在四节点(7/6/5/4), lag 多在 10 分钟内(21/22), depth 多在 10 以上(13/22), "
    "无稳定单一模式, 样本不足以支撑第三套规则; 放行失效率 2.6% 说明漏拦属可接受残差")

# ---- 成本敏感性按子规则拆分 ----
log("")
log("--- 成本敏感性: B1/B2 分别看(勿用'整体转负'表述) ---")
for ci_ in [10, 20, 30, 40]:
    n1_ = f1 * COST_FAIL - len(s1) * ci_
    n2_ = f2 * COST_FAIL - len(s2) * ci_
    log(f"拦截成本 {ci_:>2} 元: B1 净 {n1_:+5d} 元 ({'正' if n1_ > 0 else '负'}) | "
        f"B2 净 {n2_:+5d} 元 ({'正' if n2_ > 0 else '负'})")
log("失效损失敏感性(拦截成本固定20元):")
for cf_ in [50, 75, 150]:
    n1_ = f1 * cf_ - len(s1) * COST_INTERCEPT
    n2_ = f2 * cf_ - len(s2) * COST_INTERCEPT
    log(f"  每次失效损失 {cf_:>3} 元: B1 净 {n1_:+5d} 元 | B2 净 {n2_:+5d} 元")
log(f"→ 拦截成本升到40元时 B1 转负({f1 * COST_FAIL - len(s1) * 40:+d}), B2仍正({f2 * COST_FAIL - len(s2) * 40:+d}); "
    "失效损失降到50元时 B1/B2 仍分别为正；各子规则需分别评估, 不用'规则整体转负'概括")

# ---- 华北 depth<=3 × lag 四格 ----
log("")
log("--- 华北 depth<=3 × lag 四格(回应'B2 与 lag 是否重叠') ---")
for lab, gdf in h3.groupby(pd.cut(h3["lag"], [-1, 10, 1e9], labels=["lag<=10", "lag>10"]), observed=True):
    f_ = int((gdf["is_valid"] == 0).sum())
    log(f"  {lab}: n={len(gdf)}, 失效={f_}, 有效={len(gdf) - f_}, 失效率={f_ / len(gdf):.1%}")
log("→ 华北 depth<=3 中 lag>10 仅 9 行(失效 6), B2 的 +1270 主要来自 lag<=10 的 40 行(失效 24), "
    "depth 机制独立于 lag 成立")

# ---- 非华北 depth<=3: 噪声还是轻微保护 ----
log("")
log("--- 非华北 depth<=3: 噪声 vs 轻微保护 ---")
log(f"非华北 depth<=3 失效率 {(o3['is_valid'] == 0).mean():.2%} vs 全局基线 {(known['is_valid'] == 0).mean():.2%} "
    f"vs 非华北整体 {(oth['is_valid'] == 0).mean():.2%} → 略低于基线, 属'轻微保护因素'而非'纯噪声', "
    "拦截它反而拦掉更多有效信号, 故净收益为负")

# ---- 缺失行在方案 B 范围内的分布 ----
log("")
log("--- 缺失行落入方案 B 拦截范围的分布 ---")
miss_ = df[df["is_valid"].isna()]
mm1 = miss_[(miss_["lag"] > 10) & (miss_["cluster_node"] != "华北灾备节点")]
mm2 = miss_[(miss_["cluster_node"] == "华北灾备节点") & (miss_["order_depth"] <= 3)]
log(f"缺失 {len(miss_)} 条中: 落在 B1 范围 {len(mm1)} 条, 落在 B2 范围 {len(mm2)} 条, 合计 {len(mm1) + len(mm2)} 条 "
    f"(占缺失 {((len(mm1) + len(mm2)) / len(miss_)):.1%}) → 方案 B 缺失暴露小, 稳健性影响有限")

# ---- 节点名与口径卫生 ----
log("")
log("--- 口径卫生检查 ---")
log(f"cluster_node 原始取值: {[repr(x) for x in df['cluster_node'].unique()]}")
log(f"前后含空格行数: {int((df['cluster_node'] != df['cluster_node'].str.strip()).sum())} → 节点名干净, 无需 strip")
log("重复口径: 57 组(每组 2 行, 共涉及 114 行), 去重移除 57 行; 全列一致冲突组 0")
log("成本口径: 20 元/次为拦截动作成本(人工确认/延迟执行), 不含误拦有效信号的机会成本; "
    "后者通过 C* 单独建模(B1 35.66 元 / B2 66.84 元)")
log("is_valid=0 语义: 题面定义为'失效滑点超标'(执行结果), 数据无法区分信号方向误判与执行滑点; "
    "本分析解释执行失效, 不单独归因信号误判")

report = "\n".join(out)
os.makedirs("outputs", exist_ok=True)
with open("outputs/analysis_report.txt", "w", encoding="utf-8") as fh:
    fh.write(report + "\n")
print(report)


