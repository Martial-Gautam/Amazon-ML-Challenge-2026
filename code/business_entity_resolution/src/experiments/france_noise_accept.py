"""Final France step (27 Sep, LB 0.986371 -> 0.986525): re-accept French generator noise copies that rule P1 removed.
P1 (france_rule.py) rejects same-address single common-word swaps whose added word looks symmetric (log add/remove
ratio <= 2 in the French test records). "services" is one of the generator's noise words (US/India: center / services /
service / partners, 97% true), but French names also use it as a generic word, so its French ratio (1.41) fell under
P1's cut and P1 removed ~3,000 true copies the models scored >= 0.91. Also "associes" (the French "partners", ratio 5.3).
Accepts (record not owned by any S1, one owner per record), on the same house number:
  associes (any score), services (prob >= 0.5), and optional --words W1,W2 (prob >= 0.91 and in the P1 list).

python experiments/france_noise_accept.py --base output_probe_hyb2_p5b_frt0.19 --out output_final
"""
import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process

W = Path("work9")


def swaps(prob):
    """French same-address single-word swaps (one word removed, one added, >= 1 shared, not typo-like) with house keys."""
    s1 = pl.scan_parquet(str(W / "test_s1" / "*.parquet")).filter(pl.col("country") == "France") \
           .select(i="idx", n1="core", a1="addr", h1="hn", source1_entity_id="entity_id").collect()
    cd = pl.scan_parquet(str(W / "test_cand" / "*.parquet")).filter(pl.col("country") == "France") \
           .select(j="idx", n2="core", a2="addr", h2="hn", e2="entity_id").collect()
    d = pl.read_parquet(prob, columns=["i", "j", "prob"]).join(s1, on="i").join(cd, on="j")
    d = d.with_columns(t1=pl.col("n1").str.split(" ").list.unique(), t2=pl.col("n2").str.split(" ").list.unique())
    d = d.with_columns(ra=pl.col("t1").list.set_difference("t2"), rb=pl.col("t2").list.set_difference("t1"),
                       k=pl.col("t1").list.set_intersection("t2").list.len())
    d = d.filter((pl.col("ra").list.len() == 1) & (pl.col("rb").list.len() == 1) & (pl.col("k") >= 1)) \
         .with_columns(rem=pl.col("ra").list.first(), add=pl.col("rb").list.first())
    d = d.with_columns(tsim=pl.Series(process.cpdist(d["rem"].to_list(), d["add"].to_list(), scorer=fuzz.ratio, workers=-1))) \
         .filter(pl.col("tsim") < 70)
    d = d.filter((pl.col("h1") != "") & (pl.col("h1") == pl.col("h2")))  # same house number
    nrec = cd.filter(pl.col("h2") != "").group_by("n2", "h2").len("nrec")
    ns1 = s1.filter(pl.col("h1") != "").group_by(n2="n1", h2="h1").len("ns1")
    d = d.join(nrec, on=["n2", "h2"], how="left").join(ns1, on=["n2", "h2"], how="left").fill_null(0)
    d = d.filter((pl.col("nrec") <= 1) & (pl.col("ns1") == 0))  # single record, no S1 with the record's name there
    d = d.with_columns(asim=pl.Series(process.cpdist(d["a1"].to_list(), d["a2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)))
    return d.filter(pl.col("asim") >= 95).select("source1_entity_id", "e2", "add", "prob")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--prob", type=Path, default=W / "test_prob_v9_p5b.parquet", help="French P5+P5b stage-2 probabilities")
    ap.add_argument("--p1", type=Path, default=W / "g_test_common.parquet", help="the P1 list (source1_entity_id, e2)")
    ap.add_argument("--words", default="", help="extra P1-removed words to re-accept at prob >= 0.91 (comma list)")
    ap.add_argument("--reverse-rules", action="store_true",
                    help="second pass (final, LB 0.986691): re-accept every P1/P2'-removed pair that is its record's one-owner "
                         "winner with prob >= 0.91 on a record no S1 owns (both France cross-encoders were confident)")
    ap.add_argument("--p2", type=Path, default=W / "g2_test_diffaddr_sym.parquet", help="the P2' list")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = swaps(a.prob)
    p1 = pl.read_parquet(a.p1).select("source1_entity_id", "e2").unique()
    sel = d.filter((pl.col("add") == "associes") | ((pl.col("add") == "services") & (pl.col("prob") >= 0.5)))
    extra = [w for w in a.words.split(",") if w]
    if extra:
        sel = pl.concat([sel, d.filter(pl.col("add").is_in(extra) & (pl.col("prob") >= 0.91)).join(p1, on=["source1_entity_id", "e2"], how="semi")])
    g = sel.sort("prob", descending=True).unique("e2", keep="first").select("source1_entity_id", "e2")
    gp = Path(f"{a.out}_accept.parquet")
    g.write_parquet(gp)
    print(f"noise-copy candidates: {g.height} ({sel.group_by('add').len().sort('len', descending=True).rows()})", flush=True)
    first = Path(f"{a.out}_noise") if a.reverse_rules else a.out
    subprocess.run([sys.executable, "experiments/france_probe.py", "--base", str(a.base), "--groups", str(gp), "--mode", "accept",
                    "--out", str(first)], check=True)
    if a.reverse_rules:
        s1 = pl.scan_parquet(str(W / "test_s1" / "*.parquet")).select(i="idx", source1_entity_id="entity_id").collect()
        cd = pl.scan_parquet(str(W / "test_cand" / "*.parquet")).select(j="idx", e2="entity_id").collect()
        p = pl.read_parquet(a.prob, columns=["i", "j", "prob"])
        win = p.filter(pl.col("prob") == pl.col("prob").max().over("j")).select("i", "j")
        rules = pl.concat([pl.read_parquet(f).select("source1_entity_id", "e2") for f in (a.p1, a.p2)]).unique() \
                  .join(s1, on="source1_entity_id").join(cd, on="e2").join(p, on=["i", "j"])
        rev = rules.filter(pl.col("prob") >= 0.91).join(win, on=["i", "j"], how="semi") \
                   .sort("prob", descending=True).unique("e2", keep="first").select("source1_entity_id", "e2")
        rp = Path(f"{a.out}_reverse.parquet")
        rev.write_parquet(rp)
        print(f"rule reversal candidates (winners, prob >= 0.91): {rev.height}", flush=True)
        subprocess.run([sys.executable, "experiments/france_probe.py", "--base", str(first), "--groups", str(rp), "--mode", "accept",
                        "--out", str(a.out)], check=True)
    subprocess.run([sys.executable, "utils/validate_submission.py", "-m", str(a.out / "matching_results.tsv"),
                    "-c", str(a.out / "candidate_pairs.tsv"), "-t", "dataset/test", "--check-ids"], check=True)


if __name__ == "__main__":
    main()
