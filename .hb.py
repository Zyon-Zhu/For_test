import pandas as pd
df = pd.read_csv("signal_records.csv").drop_duplicates()
df["sig"]=pd.to_datetime(df.signal_time,utc=True); df["qt"]=pd.to_datetime(df.quote_synced_at,utc=True)
df["lag"]=(df.sig-df.qt).dt.total_seconds()/60
k = df[df.is_valid.notna()]
hb = k[k.cluster_node=="华北灾备节点"]
oth = k[k.cluster_node!="华北灾备节点"]
print(f"华北已知结果行: {len(hb)}, 失效 {int((hb.is_valid==0).sum())}, 失效率 {(hb.is_valid==0).mean():.1%}")
print(f"其余三节点: {len(oth)}, 失效率 {(oth.is_valid==0).mean():.1%}")
print()
print("=== 华北 order_depth 分箱 ===")
b = pd.cut(hb.order_depth, [-1,0,3,10,20,30], labels=["0","1-3","4-10","11-20","21-30"])
for lab,g in hb.groupby(b, observed=True):
    print(f"  depth {lab}: n={len(g)}, 失效={int((g.is_valid==0).sum())}, 失效率={(g.is_valid==0).mean():.1%}")
print("对照-其余三节点:")
b2 = pd.cut(oth.order_depth, [-1,0,3,10,20,30], labels=["0","1-3","4-10","11-20","21-30"])
for lab,g in oth.groupby(b2, observed=True):
    print(f"  depth {lab}: n={len(g)}, 失效={int((g.is_valid==0).sum())}, 失效率={(g.is_valid==0).mean():.1%}")
print()
print("=== 华北 asset_class ===")
for v,g in hb.groupby("asset_class"):
    print(f"  {v}: n={len(g)}, 失效={int((g.is_valid==0).sum())}, 失效率={(g.is_valid==0).mean():.1%}")
print("对照-其余三节点最差三档:")
g2 = oth.groupby("asset_class").agg(n=("is_valid","size"), fr=("is_valid", lambda s:(s==0).mean())).sort_values("fr",ascending=False)
print(g2.head(3).to_string())
print()
print("=== 华北 volume_24h 分箱 ===")
vb = pd.cut(hb.volume_24h, [-1,30,50,70,1000], labels=["<=30","31-50","51-70",">70"])
for lab,g in hb.groupby(vb, observed=True):
    print(f"  vol {lab}: n={len(g)}, 失效={int((g.is_valid==0).sum())}, 失效率={(g.is_valid==0).mean():.1%}")
print()
print("=== 华北 lag 分布 vs 其余 ===")
print(f"华北 lag 中位 {hb.lag.median():.0f}m, lag>10 占比 {(hb.lag>10).mean():.1%} | 其余 {oth.lag.median():.0f}m, {(oth.lag>10).mean():.1%}")
# 华北: depth<=3 内 lag 关系
hb3 = hb[hb.order_depth<=3]
print(f"华北 depth<=3: n={len(hb3)}, 失效率 {(hb3.is_valid==0).mean():.1%}; 其中 lag>10: n={len(hb3[hb3.lag>10])}, 失效率 {(hb3[hb3.lag>10].is_valid==0).mean():.1%}" if len(hb3) else "华北 depth<=3 无样本")
# 华北按天
print()
print("=== 华北按天 ===")
for d,g in hb.groupby(hb.sig.dt.date):
    print(f"  {d}: n={len(g)}, 失效率={(g.is_valid==0).mean():.1%}, lag>10占比={(g.lag>10).mean():.1%}")
