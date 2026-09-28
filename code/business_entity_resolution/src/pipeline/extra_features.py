"""Extra label-free pair features added to the classifier's feature table.

Subcommands (python pipeline/extra_features.py <subcommand> --help):

  rarity  (was rarity_cols.py)
    Name-rarity features (laptop's idea 1b, A/B +0.0038) for a split's final pairs -> <work>/rarity_{split}.parquet,
    used by run.py --extra. Label-free counts over ALL records of that split (so test/France counts come from test):
      name_df_s1          #S1 with the same (ctry, core)                      (on i)
      name_df_cand        #candidates (S2+S3, any address) with the same (ctry, core)  (on j)
      cand_addr_empty     candidate address empty after strip                 (on j)
      name_df_s1_x_empty  name_df_s1 * cand_addr_empty
      name_df_s1_eq1, name_df_cand_le1: scale-free flags (raw counts shift with split size)
    Empty core -> counts null.

    python pipeline/extra_features.py rarity --work work9 --tag k30_p1500_ap_ak_use20_rs_knn5_re

  word-tfidf  (was tfw_cols.py)
    Word-level TF-IDF cosines (tfw_core, tfw_addr) for v8's train and test pairs -> work/tfw_{train,test}.parquet.
    Vectorisers are fitted on train + test text together (label-free), so French words (test only) get real IDF weights;
    a train-only fit would treat every French word as unknown. Same settings as experiments/tfidf_ab.py (A/B: +0.0012).

    python pipeline/extra_features.py word-tfidf --work work --tag k30_p1500_ap_ak_use20
"""

import argparse
import sys
from pathlib import Path

import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def rarity(work: Path, split: str, pairs: Path) -> pl.DataFrame:
    s1 = pl.scan_parquet(str(work / f"{split}_s1" / "*.parquet")).select("idx", "ctry", "core")
    cand = pl.scan_parquet(str(work / f"{split}_cand" / "*.parquet")).select("idx", "ctry", "core", "business_address")
    df = lambda lf: pl.when(pl.col("core") != "").then(pl.len().over("ctry", "core")).cast(pl.Float32)
    a = s1.select(i="idx", name_df_s1=df(s1))
    b = cand.select(j="idx", name_df_cand=df(cand),
                    cand_addr_empty=(pl.col("business_address").fill_null("").str.strip_chars() == "").cast(pl.Int8))
    return (pl.scan_parquet(pairs).select("i", "j").join(a, on="i", how="left").join(b, on="j", how="left")
            .with_columns(name_df_s1_x_empty=pl.col("name_df_s1") * pl.col("cand_addr_empty"),
                          name_df_s1_eq1=(pl.col("name_df_s1") == 1).cast(pl.Int8),
                          name_df_cand_le1=(pl.col("name_df_cand") <= 1).cast(pl.Int8))
            .collect())


# ── rarity (was rarity_cols.py) ─────────────────────────────────────────────────
def main_rarity():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work9"))
    ap.add_argument("--tag", default="k30_p1500_ap_ak_use20_rs_knn5_re")
    a = ap.parse_args()
    for split in ("train", "test"):
        r = rarity(a.work, split, a.work / f"{split}_pairs_{a.tag}.parquet")
        r.write_parquet(a.work / f"rarity_{split}.parquet")
        print(f"{split}: {r.height} pairs, unique-S1-name share {r['name_df_s1_eq1'].mean():.3f}, "
              f"empty-address candidates {r['cand_addr_empty'].mean():.3f}", flush=True)


WORD = dict(analyzer="word", token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, min_df=2, dtype=np.float32)


def records(work, split, which):
    return (pl.scan_parquet(str(work / f"{split}_{which}" / "*.parquet")).select("idx", "core", "addr")
            .collect().sort("idx").with_columns(pl.col("core", "addr").fill_null("")))


# ── word-tfidf (was tfw_cols.py) ────────────────────────────────────────────────
def main_word_tfidf():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--tag", default="k30_p1500_ap_ak_use20", help="pairs file: {split}_pairs_<tag>.parquet")
    a = ap.parse_args()
    recs = {(s, w): records(a.work, s, w) for s in ("train", "test") for w in ("s1", "cand")}
    pairs = {s: pl.read_parquet(a.work / f"{s}_pairs_{a.tag}.parquet", columns=["i", "j"]) for s in ("train", "test")}
    out = {s: {"i": pairs[s]["i"], "j": pairs[s]["j"]} for s in pairs}
    for col in ("core", "addr"):
        vec = TfidfVectorizer(**WORD).fit(pl.concat([r[col] for r in recs.values()]).to_list())
        print(f"tfw_{col}: vocab {len(vec.vocabulary_)}", flush=True)
        for s, p in pairs.items():
            A, B = recs[(s, "s1")], recs[(s, "cand")]
            XA, XB = vec.transform(A[col].to_list()).tocsr(), vec.transform(B[col].to_list()).tocsr()
            ri = np.searchsorted(A["idx"].to_numpy(), p["i"].to_numpy())
            rj = np.searchsorted(B["idx"].to_numpy(), p["j"].to_numpy())
            cos = np.empty(p.height, np.float32)
            for o in range(0, p.height, 1_000_000):
                cos[o:o + 1_000_000] = np.asarray(XA[ri[o:o + 1_000_000]].multiply(XB[rj[o:o + 1_000_000]]).sum(1)).ravel()
            cos[(A[col].str.strip_chars() == "").to_numpy()[ri] | (B[col].str.strip_chars() == "").to_numpy()[rj]] = np.nan
            out[s][f"tfw_{col}"] = cos
            print(f"  {s}: {p.height} pairs, mean cos {np.nanmean(cos):.3f}", flush=True)
    for s in out:
        pl.DataFrame(out[s]).write_parquet(a.work / f"tfw_{s}.parquet")
    print("wrote work/tfw_train.parquet, work/tfw_test.parquet", flush=True)


SUBCOMMANDS = {"rarity": main_rarity, "word-tfidf": main_word_tfidf}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        sys.exit("usage: python extra_features.py {" + " | ".join(SUBCOMMANDS) + "} [options]")
    SUBCOMMANDS[sys.argv.pop(1)]()
