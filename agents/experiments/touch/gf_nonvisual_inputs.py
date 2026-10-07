"""Non-visual inputs to the giant fibers (DNp01) in the MaleCNS v1.0 connectome: candidates for a "touch" channel.

Slow balls near a flipper tip do not loom, so the visual LC4/LPLC2 drive stays below the giant fibers' threshold.
This lists every direct input to each DNp01 grouped by superclass/class/type (excluding LC4/LPLC2), and the
sensory neurons (any superclass containing "sensory") that reach DNp01 directly or through one intermediate, with
their side, entry nerve, receptor type and path weight, to pick a mechanosensory channel to add to the body.

Usage: python agents/experiments/touch/gf_nonvisual_inputs.py   (needs data/*.feather, agents/download_malecns.sh)
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = os.path.join(ROOT, "data")
ann = pd.read_feather(os.path.join(D, "body-annotations.feather")).set_index("bodyId")
w = pd.read_feather(os.path.join(D, "connectome-weights.feather"))
cols = {c.lower(): c for c in w.columns}
pre, post, wt = cols.get("body_pre", w.columns[0]), cols.get("body_post", w.columns[1]), cols.get("weight", w.columns[2])
gf = ann[ann["type"] == "DNp01"]
gf_ids = {row["somaSide"]: bid for bid, row in gf.iterrows()}
print("giant fibers:", gf_ids)
info = ann[["type", "superclass", "class", "somaSide", "rootSide", "entryNerve", "receptorType", "subclass"]]
into = w[w[post].isin(gf_ids.values())].join(info, on=pre)
total = into.groupby(post)[wt].sum()
vis = into["type"].isin(["LC4", "LPLC2"])
print(f"\n== direct inputs (synapses); visual LC4/LPLC2 share: "
      + ", ".join(f"GF_{s} {into[(into[post]==g)&vis][wt].sum()/total[g]:.1%}" for s, g in gf_ids.items()))
for s, g in gf_ids.items():
    e = into[(into[post] == g) & ~vis]
    by = e.groupby(["superclass", "class"], dropna=False)[wt].sum().sort_values(ascending=False).head(12)
    print(f"\n-- GF_{s} non-visual inputs by superclass/class (share of all its input):")
    for (sc, cl), v in by.items():
        print(f"   {str(sc):28s} {str(cl):30s} {int(v):6d}  {v/total[g]:5.1%}")
    tp = e.groupby("type")[wt].sum().sort_values(ascending=False).head(15)
    print(f"-- GF_{s} top non-visual input types: " + ", ".join(f"{t} {int(v)}" for t, v in tp.items()))

# sensory neurons reaching the GF directly or via one intermediate
sens_ids = set(ann.index[ann["superclass"].fillna("").str.contains("sensory")])
direct = into[into[pre].isin(sens_ids)]
print(f"\n== sensory -> GF direct: {len(direct)} edges, {int(direct[wt].sum())} synapses")
if len(direct):
    print(direct.groupby(["type", "class", "entryNerve", "somaSide"], dropna=False)[wt].sum().sort_values(ascending=False).head(15).to_string())
mid = into[~into[pre].isin(sens_ids)][[pre, post, wt]].rename(columns={pre: "mid", post: "gf", wt: "w2"})
from_s = w[w[pre].isin(sens_ids) & w[post].isin(set(mid["mid"]))]
two = from_s.merge(mid, left_on=post, right_on="mid")
two["path"] = two[wt] * two["w2"]
two = two.join(info.add_prefix("s_"), on=pre).join(info[["type"]].add_prefix("m_"), on="mid")
two["gf_side"] = two["gf"].map({g: s for s, g in gf_ids.items()})
print("\n== sensory -> X -> GF (two-hop), top sensory types by path weight:")
print(two.groupby(["s_type", "s_class", "s_entryNerve", "s_receptorType", "s_somaSide", "gf_side"], dropna=False)["path"]
      .sum().sort_values(ascending=False).head(20).to_string())
print("\n== top intermediates on those two-hop paths:")
print(two.groupby(["m_type", "gf_side"])["path"].sum().sort_values(ascending=False).head(12).to_string())
