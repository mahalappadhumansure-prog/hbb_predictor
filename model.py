"""HBB variant pathogenicity: feature extraction, model training, SHAP explanations.

IMPORTANT: the training labels here are SYNTHETIC (rule-based + noise) so the
prototype runs end to end. For real results, replace `build_training_data()`
with HBB variants from ClinVar (see README).
"""
import base64
import io
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio.SeqUtils import seq3
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- reference
# HBB coding sequence (NM_000518). Verify against NCBI if you change it.
REF_CDS = (
    "ATGGTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGT"
    "GAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCT"
    "GATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCAC"
    "CTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTC"
    "AGGCTCCTGGGCAACGTGCTGGTCTGCGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCC"
    "TATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCACTAA"
)
REF_PROTEIN = (
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAH"
    "LDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH*"
)
# Safety check: catches typos in the reference sequence at startup.
assert str(Seq(REF_CDS).translate()) == REF_PROTEIN, "REF_CDS does not encode HBB protein"

BLOSUM = substitution_matrices.load("BLOSUM62")
FEATURES = ["Amino acid change", "A-T content", "Conservation score", "k-mer (CTG)", "Nucleotide position"]
WINDOW = 30  # nt either side of the variant for A-T content and k-mer counts

# Placeholder conservation scores (0-1). Position numbering = mature HBB (Val1 ... His146).
# Replace with real values from a multi-species alignment / phyloP / ConSurf.
KNOWN_CONSERVED = {6: 0.98, 58: 0.95, 63: 1.0, 92: 1.0, 102: 0.9, 106: 0.9}


def conservation(aa_pos: int) -> float:
    if aa_pos in KNOWN_CONSERVED:
        return KNOWN_CONSERVED[aa_pos]
    return float(np.random.default_rng(aa_pos).uniform(0.3, 0.9))


# ---------------------------------------------------------------- features
def featurize(seq: str, nt_idx: int, ref_aa: str, alt_aa: str) -> list:
    aa_pos = nt_idx // 3  # codon index == mature numbering (Met is 0)
    if ref_aa == alt_aa:
        severity = 0.0
    elif alt_aa == "*" or aa_pos == 0:
        severity = 1.0
    else:
        severity = float(np.clip((4 - BLOSUM[ref_aa][alt_aa]) / 8, 0, 1))
    lo, hi = max(0, nt_idx - WINDOW), min(len(seq), nt_idx + WINDOW + 1)
    win = seq[lo:hi]
    at = (win.count("A") + win.count("T")) / len(win)
    return [severity, at, conservation(aa_pos), win.count("CTG"), nt_idx / len(REF_CDS)]


def codon_change(seq: str, nt_idx: int):
    c = nt_idx // 3 * 3
    ref_codon, alt_codon = REF_CDS[c:c + 3], seq[c:c + 3]
    return ref_codon, alt_codon, str(Seq(ref_codon).translate()), str(Seq(alt_codon).translate())


# ---------------------------------------------------------------- training
def build_training_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        idx = int(rng.integers(0, len(REF_CDS) - 3))
        alt_base = rng.choice([b for b in "ACGT" if b != REF_CDS[idx]])
        mut = REF_CDS[:idx] + alt_base + REF_CDS[idx + 1:]
        _, _, ref_aa, alt_aa = codon_change(mut, idx)
        f = featurize(mut, idx, ref_aa, alt_aa)
        # synthetic "ground truth": severity + conservation drive pathogenicity
        logit = 6 * (0.6 * f[0] + 0.4 * f[2] - 0.5) + 0.8 * (f[1] - 0.5)
        y.append(int(rng.random() < 1 / (1 + np.exp(-logit))))
        X.append(f)
    return np.array(X), np.array(y)


X, y = build_training_data()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
MODEL = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42).fit(X_tr, y_tr)
METRICS = {
    "accuracy": round(accuracy_score(y_te, MODEL.predict(X_te)), 3),
    "roc_auc": round(roc_auc_score(y_te, MODEL.predict_proba(X_te)[:, 1]), 3),
    "n_train": len(X_tr),
    "n_test": len(X_te),
    "note": "Trained on synthetic labels. Replace with ClinVar data for real performance.",
}
EXPLAINER = shap.TreeExplainer(MODEL)
_bg = X_te[:300]
_bg_sv = EXPLAINER.shap_values(_bg)
GLOBAL_IMPORTANCE = dict(zip(FEATURES, np.abs(_bg_sv).mean(axis=0).round(3).tolist()))


def _summary_png() -> bytes:
    plt.figure(figsize=(6, 3.6))
    shap.summary_plot(_bg_sv, _bg, feature_names=FEATURES, show=False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, transparent=True)
    plt.close()
    return buf.getvalue()


SUMMARY_PNG = _summary_png()


# ---------------------------------------------------------------- analysis
def mutate(idx: int, base: str) -> str:
    return REF_CDS[:idx] + base + REF_CDS[idx + 1:]


EXAMPLES = {
    "HbS (E6V) - Pathogenic": mutate(19, "T"),
    "HbC (E6K) - Pathogenic": mutate(18, "A"),
    "Synonymous (E6E) - Likely benign": mutate(20, "A"),
    "Reference HBB (no variant)": REF_CDS,
}


def parse_fasta(text: str) -> str:
    body = "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))
    return re.sub(r"[^ACGT]", "", body.upper().replace("U", "T"))


def analyze(text: str) -> dict:
    seq = parse_fasta(text)
    if len(seq) < 30:
        raise ValueError("Sequence too short. Paste at least 30 nt of the HBB coding sequence starting at ATG.")
    n = min(len(seq), len(REF_CDS))
    diffs = [i for i in range(n) if seq[i] != REF_CDS[i] and (i // 3 * 3 + 3) <= len(seq)]
    base = {"gene": "HBB (Hemoglobin subunit beta)", "length": len(seq), "sequence": seq}
    if not diffs:
        return {**base, "variant": None, "label": "Benign", "confidence": None,
                "message": "Sequence matches the HBB reference. No variant detected."}

    idx = diffs[0]
    ref_codon, alt_codon, ref_aa, alt_aa = codon_change(seq, idx)
    feats = featurize(seq, idx, ref_aa, alt_aa)
    arr = np.array([feats])
    p = float(MODEL.predict_proba(arr)[0, 1])
    sv = EXPLAINER.shap_values(arr)[0]
    pathogenic = p >= 0.5
    aa_pos = idx // 3
    return {
        **base,
        "label": "Pathogenic" if pathogenic else "Benign",
        "confidence": round(p if pathogenic else 1 - p, 3),
        "variant": {
            "position": aa_pos,
            "nt_position": idx + 1,
            "codon_start": idx // 3 * 3,
            "ref_codon": f"{ref_codon} ({seq3(ref_aa)})",
            "alt_codon": f"{alt_codon} ({seq3(alt_aa)})",
            "aa_change": f"{ref_aa}{aa_pos}{alt_aa}",
            "conservation": feats[2],
            "extra_variants": len(diffs) - 1,
        },
        "shap": sorted(
            [{"feature": f, "value": round(float(v), 3)} for f, v in zip(FEATURES, sv)],
            key=lambda d: -abs(d["value"]),
        ),
    }


def summary_png_b64() -> str:
    return base64.b64encode(SUMMARY_PNG).decode()
