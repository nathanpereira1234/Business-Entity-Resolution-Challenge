"""
Business Entity Resolution pipeline.
data -> normalisation -> blocking (TF-IDF char/word kNN) -> pairwise features
     -> LightGBM classifier -> threshold + one-to-one decoding -> TSV outputs

Usage:
  python src/pipeline.py --data dataset --out output
"""
import argparse
import os
import re
import unicodedata

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

SEED = 42

# ---------------------------------------------------------------- normalisation
NAME_ABBR = {
    "corp": "corporation", "co": "company", "cos": "company", "inc": "incorporated",
    "ltd": "limited", "lt": "limited", "pvt": "private", "pte": "private",
    "intl": "international", "int": "international", "mfg": "manufacturing",
    "svc": "services", "svcs": "services", "bros": "brothers", "assoc": "associates",
    "grp": "group", "ent": "enterprises", "entp": "enterprises", "mgmt": "management",
    "natl": "national", "dept": "department", "ind": "industries", "inds": "industries",
    "n": "and", "et": "and", "cie": "company", "ste": "societe", "soc": "societe",
}
ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "ln": "lane",
    "dr": "drive", "hwy": "highway", "apt": "apartment", "ste": "suite", "fl": "floor",
    "flr": "floor", "nr": "near", "opp": "opposite", "ctr": "center", "centre": "center",
    "sq": "square", "pl": "place", "pkwy": "parkway", "ct": "court", "bldg": "building",
    "n": "north", "s": "south", "e": "east", "w": "west", "no": "number", "chs": "society",
    "soc": "society", "mkt": "market", "stn": "station", "extn": "extension", "ext": "extension",
    "mg": "mahatma gandhi", "sect": "sector", "sec": "sector", "ph": "phase", "fbg": "faubourg",
    "che": "chemin", "rte": "route", "imp": "impasse", "all": "allee", "qu": "quai",
}
LEGAL = {
    "incorporated", "corporation", "company", "limited", "private", "llc", "llp", "plc",
    "lp", "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "gmbh", "the", "and", "of",
    "dba", "india", "societe", "opc", "le", "la", "les", "de", "du", "des",
}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm(s, abbr):
    if not isinstance(s, str):
        return ""
    s = strip_accents(s).lower().replace("&", " and ").replace("'", "")
    s = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", s)  # 12b -> 12 b
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(abbr.get(t, t) for t in s.split())


def core_name(n):
    toks = [t for t in n.split() if t not in LEGAL]
    return " ".join(toks) if toks else n


def acronym(n):
    return "".join(t[0] for t in n.split() if t)


def numbers(a):
    return set(re.findall(r"\d+", a))


def prep(df):
    df = df.copy()
    df["name_n"] = df["business_name"].map(lambda s: norm(s, NAME_ABBR))
    df["addr_n"] = df["business_address"].map(lambda s: norm(s, ADDR_ABBR))
    df["core"] = df["name_n"].map(core_name)
    df["acr"] = df["core"].map(acronym)
    df["full"] = df["core"] + " | " + df["addr_n"]
    df["nums"] = df["addr_n"].map(numbers)
    df["longnum"] = df["nums"].map(lambda s: {x for x in s if len(x) >= 4})
    df["cty"] = df["country"].fillna("").astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


def load_split(data_dir, split):
    rd = lambda f: pd.read_csv(os.path.join(data_dir, split, f), sep="\t", dtype=str,
                               keep_default_na=False, na_values=[""])
    s1 = prep(rd(f"{split}_source1.tsv"))
    tg = prep(pd.concat([rd(f"{split}_source2.tsv"), rd(f"{split}_source3.tsv")],
                        ignore_index=True))
    return s1, tg


# ---------------------------------------------------------------- vectorising
def vectorise(s1, tg):
    """Fit TF-IDF on the split's own text only (unsupervised, no external data)."""
    specs = {
        "name_char": ("core", dict(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
        "name_word": ("core", dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)),
        "addr_char": ("addr_n", dict(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True)),
        "addr_word": ("addr_n", dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)),
        "full_char": ("full", dict(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True)),
    }
    out = {}
    for key, (col, kw) in specs.items():
        v = TfidfVectorizer(dtype=np.float32, **kw).fit(pd.concat([s1[col], tg[col]]))
        out[key] = (v.transform(s1[col]).tocsr(), v.transform(tg[col]).tocsr())
    return out


# ---------------------------------------------------------------- blocking
def topk(Q, T, k, chunk=512):
    k = min(k, T.shape[0])
    if k == 0 or Q.shape[0] == 0:
        return np.array([], int), np.array([], int)
    TT = T.T.tocsc()
    I, J = [], []
    for s in range(0, Q.shape[0], chunk):
        S = (Q[s:s + chunk] @ TT).toarray()
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        val = np.take_along_axis(S, idx, axis=1)
        rows = np.repeat(np.arange(s, s + S.shape[0]), k)
        keep = val.ravel() > 0
        I.append(rows[keep]); J.append(idx.ravel()[keep])
    return np.concatenate(I), np.concatenate(J)


BLOCK_K = {"name_char": 15, "addr_char": 10, "full_char": 15, "name_word": 5}


def block(s1, tg, V):
    """Country-scoped kNN union. Country is an open label set; if a Source 1
    country has no target records we fall back to all targets."""
    pairs = []
    tg_cty = tg["cty"].values
    for c, q_idx in s1.groupby("cty").indices.items():
        t_idx = np.where(tg_cty == c)[0]
        if len(t_idx) == 0:
            t_idx = np.arange(len(tg))
        for key, k in BLOCK_K.items():
            Q, T = V[key][0][q_idx], V[key][1][t_idx]
            i, j = topk(Q, T, k)
            pairs.append(np.stack([q_idx[i], t_idx[j]], 1))
    p = np.unique(np.concatenate(pairs), axis=0)
    return pd.DataFrame({"i": p[:, 0], "j": p[:, 1]})


# ---------------------------------------------------------------- features
def rowcos(A, B, i, j):
    return np.asarray(A[i].multiply(B[j]).sum(1)).ravel()


def jacc(a, b):
    return len(a & b) / len(a | b) if (a or b) else -1.0


def features(s1, tg, V, P):
    i, j = P["i"].values, P["j"].values
    F = pd.DataFrame(index=P.index)
    for key in V:
        F[key] = rowcos(V[key][0], V[key][1], i, j)

    a_core, b_core = s1["core"].values[i], tg["core"].values[j]
    a_name, b_name = s1["name_n"].values[i], tg["name_n"].values[j]
    a_addr, b_addr = s1["addr_n"].values[i], tg["addr_n"].values[j]
    for nm, sc in [("ratio", fuzz.ratio), ("tsort", fuzz.token_sort_ratio),
                   ("tset", fuzz.token_set_ratio), ("partial", fuzz.partial_ratio),
                   ("jw", JaroWinkler.normalized_similarity)]:
        F[f"core_{nm}"] = cpdist(list(a_core), list(b_core), scorer=sc, workers=-1)
    F["name_tset"] = cpdist(list(a_name), list(b_name), scorer=fuzz.token_set_ratio, workers=-1)
    for nm, sc in [("tset", fuzz.token_set_ratio), ("partial", fuzz.partial_ratio),
                   ("tsort", fuzz.token_sort_ratio)]:
        F[f"addr_{nm}"] = cpdist(list(a_addr), list(b_addr), scorer=sc, workers=-1)

    an, bn = s1["nums"].values[i], tg["nums"].values[j]
    al, bl = s1["longnum"].values[i], tg["longnum"].values[j]
    F["num_jacc"] = [jacc(x, y) for x, y in zip(an, bn)]
    F["longnum_match"] = [(-1 if not (x and y) else float(bool(x & y))) for x, y in zip(al, bl)]
    F["longnum_conflict"] = [float(bool(x and y and not (x & y))) for x, y in zip(al, bl)]
    F["first_tok_eq"] = [float(x.split()[:1] == y.split()[:1]) for x, y in zip(a_core, b_core)]
    acr_a, acr_b = s1["acr"].values[i], tg["acr"].values[j]
    F["acronym"] = [float(x.replace(" ", "") == ab or y.replace(" ", "") == aa)
                    for x, y, aa, ab in zip(a_core, b_core, acr_a, acr_b)]
    la = np.array([len(x) for x in a_core]); lb = np.array([len(x) for x in b_core])
    F["len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
    F["addr_missing"] = [float(not x or not y) for x, y in zip(a_addr, b_addr)]
    F["is_s3"] = tg["entity_id"].str.startswith("S3-").values[j].astype(float)

    # context features: how this pair compares with competitors
    base = 0.5 * F["name_char"] + 0.3 * F["addr_char"] + 0.2 * F["full_char"]
    g_i, g_j = P["i"], P["j"]
    F["base"] = base
    F["rank_in_s1"] = base.groupby(g_i).rank(ascending=False)
    F["gap_s1_max"] = base.groupby(g_i).transform("max") - base
    F["rank_in_tgt"] = base.groupby(g_j).rank(ascending=False)
    F["gap_tgt_max"] = base.groupby(g_j).transform("max") - base
    F["n_cand_s1"] = g_i.map(g_i.value_counts()).values
    F["n_cand_tgt"] = g_j.map(g_j.value_counts()).values
    for key in ["name_char", "addr_char"]:
        F[f"{key}_rank_s1"] = F[key].groupby(g_i).rank(ascending=False)
    return F


# ---------------------------------------------------------------- decoding / scoring
def decode(P, prob, t, r=0.0):
    """Threshold, keep within r of the S1's best, then enforce one-to-one:
    each S2/S3 record can belong to at most one Source 1 entity (S1 is deduplicated)."""
    d = P.assign(p=prob)
    d = d[d["p"] >= t]
    if r > 0:
        d = d[d["p"] >= r * d.groupby("i")["p"].transform("max")]
    d = d.sort_values("p", ascending=False).drop_duplicates("j")
    return d.groupby("i")["j"].apply(set).to_dict()


def macro_f05(pred, gt, n_s1):
    tot = 0.0
    for i in range(n_s1):
        p, g = pred.get(i, set()), gt.get(i, set())
        if not g:
            tot += 1.0 if not p else 0.0
            continue
        tp = len(p & g)
        if tp == 0:
            continue
        pr, rc = tp / len(p), tp / len(g)
        tot += 1.25 * pr * rc / (0.25 * pr + rc)
    return tot / n_s1


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=20,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=SEED)


# ---------------------------------------------------------------- main
def write_tsv(path, s1, tg, groups, col):
    ids = tg["entity_id"].values
    rows = [(s1.at[i, "entity_id"], ",".join(sorted(ids[list(groups.get(i, ()))])))
            for i in range(len(s1))]
    pd.DataFrame(rows, columns=["source1_entity_id", col]).to_csv(path, sep="\t", index=False)


def main(a):
    # ---------- train
    s1, tg = load_split(a.data, "train")
    V = vectorise(s1, tg)
    P = block(s1, tg, V)
    X = features(s1, tg, V, P)

    gt_df = pd.read_csv(os.path.join(a.data, "train", "train_ground_truth.tsv"), sep="\t",
                        dtype=str, keep_default_na=False)
    s1_pos = {e: k for k, e in enumerate(s1["entity_id"])}
    tg_pos = {e: k for k, e in enumerate(tg["entity_id"])}
    gt = {}
    for e, m in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        ms = {tg_pos[x] for x in m.split(",") if x.strip() in tg_pos}
        if ms:
            gt[s1_pos[e]] = ms
    y = np.array([j in gt.get(i, ()) for i, j in zip(P["i"], P["j"])], dtype=int)
    n_gt = sum(len(v) for v in gt.values())
    print(f"train: {len(s1)} S1, {len(tg)} targets, {len(P)} candidate pairs "
          f"({len(P)/len(s1):.1f}/S1), blocking recall ceiling {y.sum()/n_gt:.4f}")

    # ---------- grouped CV (group = Source 1 entity) for OOF probabilities
    oof, iters = np.zeros(len(P)), []
    for f, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, P["i"])):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print(f"fold {f}: best_iter {m.best_iteration}")

    best = (-1, 0.5, 0.0)
    for t in np.arange(0.2, 0.96, 0.025):
        for r in [0.0, 0.3, 0.5, 0.7]:
            s = macro_f05(decode(P, oof, t, r), gt, len(s1))
            if s > best[0]:
                best = (s, t, r)
    score, T, R = best
    print(f"OOF macro F0.5 = {score:.4f} at threshold {T:.3f}, rel-to-max {R}")

    # ---------- final model on all training pairs
    model = lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=int(np.median(iters) * 1.1))
    imp = pd.Series(model.feature_importance("gain"), X.columns).sort_values(ascending=False)
    print("top features:\n", imp.head(12).round(0).to_string())

    # ---------- test
    s1t, tgt = load_split(a.data, "test")
    Vt = vectorise(s1t, tgt)
    Pt = block(s1t, tgt, Vt)
    Xt = features(s1t, tgt, Vt, Pt)
    pt = model.predict(Xt)
    matches = decode(Pt, pt, T, R)
    cands = Pt.groupby("i")["j"].apply(set).to_dict()

    os.makedirs(a.out, exist_ok=True)
    write_tsv(os.path.join(a.out, "candidate_pairs.tsv"), s1t, tgt, cands, "candidate_entity_ids")
    write_tsv(os.path.join(a.out, "matching_results.tsv"), s1t, tgt, matches, "matched_entity_ids")
    by_cty = s1t.assign(n=[len(matches.get(i, ())) for i in range(len(s1t))]).groupby("cty")["n"]
    print("test: predicted singleton rate by country:\n", (by_cty.apply(lambda s: (s == 0).mean())).round(3).to_string())
    print(f"wrote {a.out}/matching_results.tsv and candidate_pairs.tsv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="output")
    main(ap.parse_args())
