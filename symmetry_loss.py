from __future__ import annotations
 
import argparse
import math
import warnings
from dataclasses import dataclass, field
 
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
 
 
# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
 
@dataclass
class SymmetryScoreResult:
    composition: str
    claimed_spacegroup: int
    detected_spacegroup: int
    n_ops_used: int
    score: float
    warnings: list[str] = field(default_factory=list)
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _site_symbol(site) -> str:
    """
    Extract a canonical species symbol from a pymatgen PeriodicSite.
 
    Handles:
      - Ordered sites with Element or Species (oxidation-state decorated).
      - Disordered sites (mixed occupancy) — returns sorted comma-joined symbols.
    """
    try:
        sp = site.specie  # ordered site
        return sp.element.symbol if hasattr(sp, "element") else sp.symbol
    except AttributeError:
        # Disordered site: .specie raises AttributeError, use .species_and_occu
        return ",".join(
            sorted(
                sp.element.symbol if hasattr(sp, "element") else sp.symbol
                for sp in site.species_and_occu
            )
        )
 
 
def _is_identity_op(symm_op, atol: float = 1e-8) -> bool:
    return np.allclose(symm_op.rotation_matrix, np.eye(3), atol=atol) and np.allclose(
        symm_op.translation_vector, np.zeros(3), atol=atol
    )
 
 
def _build_cartesian_kdtree(structure: Structure) -> cKDTree:
    """Build a KDTree over Cartesian site coordinates for O(log n) NN lookup."""
    cart_coords = np.array([site.coords for site in structure.sites])
    return cKDTree(cart_coords)
 
 
# ---------------------------------------------------------------------------
# Core: symmetry operation retrieval
# ---------------------------------------------------------------------------
 
def _get_scoring_ops(
    structure: Structure,
    claimed_spacegroup: int,
    symprec: float,
    angle_tolerance: float,
) -> tuple[list, int, list[str]]:
    """
    Return (ops, detected_sg_number, warning_messages).
 
    Strategy
    --------
    1. Run SpacegroupAnalyzer to detect the actual SG.
    2. If detected == claimed → use ops expressed in the *structure's own basis*
       (avoids setting-mismatch artefacts).
    3. If detected != claimed → fall back to standardised ops from the claimed SG
       and emit a warning so callers know the score may be penalised unfairly
       when the structure happens to be in a non-standard setting.
    """
    warn_msgs: list[str] = []
    detected_sg = -1
 
    try:
        analyzer = SpacegroupAnalyzer(
            structure, symprec=symprec, angle_tolerance=angle_tolerance
        )
        detected_sg = int(analyzer.get_space_group_number())
 
        if detected_sg == int(claimed_spacegroup):
            return list(analyzer.get_space_group_operations()), detected_sg, warn_msgs
 
        warn_msgs.append(
            f"Detected SG {detected_sg} ≠ claimed SG {claimed_spacegroup}. "
            "Falling back to standard-setting ops for claimed SG; score may "
            "partly reflect a setting mismatch rather than a true symmetry violation."
        )
    except Exception as exc:
        warn_msgs.append(
            f"SpacegroupAnalyzer failed ({exc}); using standard ops for claimed SG."
        )
 
    try:
        ops = list(SpaceGroup.from_int_number(int(claimed_spacegroup)).symmetry_ops)
    except Exception as exc2:
        warn_msgs.append(f"Could not load ops for SG {claimed_spacegroup}: {exc2}")
        ops = []
 
    return ops, detected_sg, warn_msgs
 
 
# ---------------------------------------------------------------------------
# Core: scoring
# ---------------------------------------------------------------------------
 
def score_structure_symmetry(
    structure: Structure,
    claimed_spacegroup: int,
    sigma: float = 0.25,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
) -> tuple[float, int, int, list[str]]:
    """
    Score how well a structure agrees with a claimed space group.
 
    Returns
    -------
    (score, detected_sg, n_ops_used, warnings)
 
    Score in roughly [-1, 1]:
      +1  → every symmetry-equivalent site has the correct species nearby.
       0  → neutral / random mismatch.
      -1  → systematic species mismatches at symmetry-equivalent positions.
 
    Changes vs original
    -------------------
    * Cartesian KDTree (O(log n)) replaces O(n) brute-force loop.
    * Safe species extraction handles disordered / oxidation-state-decorated sites.
    * Gaussian uses Cartesian distance (Å) instead of fractional — sigma is now
      physically meaningful and consistent across different unit-cell sizes.
    * Returns explicit warnings instead of silently discarding issues.
    """
    warn_msgs: list[str] = []
 
    try:
        ops, detected_sg, op_warns = _get_scoring_ops(
            structure=structure,
            claimed_spacegroup=claimed_spacegroup,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        warn_msgs.extend(op_warns)
    except Exception as exc:
        return float("nan"), -1, 0, [f"Op retrieval failed: {exc}"]
 
    non_identity_ops = [op for op in ops if not _is_identity_op(op)]
 
    if not non_identity_ops:
        # SG #1 (P1) — no non-identity constraints; trivially satisfied.
        return 1.0, detected_sg, 0, warn_msgs
 
    # Build KDTree once, reuse across all (op × site) pairs  ←  key speed-up
    kdtree = _build_cartesian_kdtree(structure)
    site_symbols = [_site_symbol(s) for s in structure.sites]
    sigma_sq = sigma ** 2
 
    per_site_scores: list[float] = []
 
    for op in non_identity_ops:
        for site in structure.sites:
            transformed_frac = np.mod(op.operate(site.frac_coords), 1.0)
            transformed_cart = structure.lattice.get_cartesian_coords(transformed_frac)
 
            # O(log n) nearest-neighbour query
            nn_dist, nn_idx = kdtree.query(transformed_cart, k=1)
 
            expected_symbol = _site_symbol(site)
            nearest_symbol = site_symbols[int(nn_idx)]
 
            dist_factor = math.exp(-(nn_dist ** 2) / sigma_sq)
            per_site_scores.append(dist_factor if nearest_symbol == expected_symbol else -dist_factor)
 
    if not per_site_scores:
        return float("nan"), detected_sg, len(non_identity_ops), warn_msgs
 
    return float(np.mean(per_site_scores)), detected_sg, len(non_identity_ops), warn_msgs
 
 
# ---------------------------------------------------------------------------
# Corruption helper (for negative testing)
# ---------------------------------------------------------------------------
 
def _corrupt_structure_for_negative_test(
    structure: Structure, mode: str = "aggressive"
) -> Structure:
    """
    Create a deterministic corruption that induces species-mismatch penalties.
 
    Modes
    -----
    aggressive
        Shuffle species assignments while keeping coordinates fixed.
        Guarantees species mismatches → clearly negative scores.
        (The original clustered atoms near one point, which drives scores toward
        0 rather than below 0 — making it a poor negative test.)
    mild
        Swap coordinates of two atoms with *different* species, plus a small
        positional nudge on the first atom.
    """
    corrupted = structure.copy()
    mode = mode.lower().strip()
 
    if len(corrupted) == 0:
        return corrupted
 
    if mode == "aggressive":
        rng = np.random.default_rng(seed=42)
        all_symbols = [_site_symbol(s) for s in corrupted.sites]
        shuffled = all_symbols.copy()
        for _ in range(200):  # guaranteed to terminate for multi-species structures
            rng.shuffle(shuffled)
            if shuffled != all_symbols:
                break
        for i, site in enumerate(corrupted.sites):
            try:
                new_sp = Element(shuffled[i].split(",")[0])
            except Exception:
                new_sp = site.specie
            corrupted.replace(i, new_sp, coords=site.frac_coords, coords_are_cartesian=False)
        return corrupted
 
    # ---- mild mode ----
    idx_a = idx_b = None
    for i in range(len(corrupted)):
        for j in range(i + 1, len(corrupted)):
            if _site_symbol(corrupted[i]) != _site_symbol(corrupted[j]):
                idx_a, idx_b = i, j
                break
        if idx_a is not None:
            break
 
    if idx_a is not None and idx_b is not None:
        a = np.array(corrupted[idx_a].frac_coords)
        b = np.array(corrupted[idx_b].frac_coords)
        corrupted.replace(idx_a, corrupted[idx_a].specie, coords=b, coords_are_cartesian=False)
        corrupted.replace(idx_b, corrupted[idx_b].specie, coords=a, coords_are_cartesian=False)
 
    if len(corrupted) > 0:
        p0 = np.mod(np.array(corrupted[0].frac_coords) + np.array([0.13, 0.07, 0.11]), 1.0)
        corrupted.replace(0, corrupted[0].specie, coords=p0, coords_are_cartesian=False)
 
    return corrupted
 
 
# ---------------------------------------------------------------------------
# CSV-level helpers
# ---------------------------------------------------------------------------
 
def score_csv(
    csv_path: str,
    limit: int = 10,
    sigma: float = 0.25,
    cif_column: str = "cif.conv",
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
    print_warnings: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.head(limit)
 
    rows: list[SymmetryScoreResult] = []
    for _, row in df.iterrows():
        cif_text = row.get(cif_column)
        claimed_sg = row.get("spacegroup.number")
        composition = row.get("pretty_formula", "")
 
        if pd.isna(cif_text) or pd.isna(claimed_sg):
            rows.append(
                SymmetryScoreResult(
                    composition=str(composition),
                    claimed_spacegroup=-1,
                    detected_spacegroup=-1,
                    n_ops_used=0,
                    score=float("nan"),
                    warnings=["Missing CIF text or spacegroup number."],
                )
            )
            continue
 
        try:
            structure = Structure.from_str(str(cif_text), fmt="cif")
            score, detected_sg, n_ops_used, warns = score_structure_symmetry(
                structure,
                int(claimed_sg),
                sigma=sigma,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
            )
        except Exception as exc:
            score, detected_sg, n_ops_used = float("nan"), -1, 0
            warns = [f"Structure parsing/scoring failed: {exc}"]
 
        if print_warnings:
            for w in warns:
                warnings.warn(f"[{composition}] {w}")
 
        rows.append(
            SymmetryScoreResult(
                composition=str(composition),
                claimed_spacegroup=int(claimed_sg),
                detected_spacegroup=int(detected_sg),
                n_ops_used=int(n_ops_used),
                score=score,
                warnings=warns,
            )
        )
 
    return pd.DataFrame(
        {
            "composition":          [r.composition        for r in rows],
            "claimed_space_group":  [r.claimed_spacegroup  for r in rows],
            "detected_space_group": [r.detected_spacegroup for r in rows],
            "ops_used":             [r.n_ops_used          for r in rows],
            "score":                [r.score               for r in rows],
            "warnings":             ["; ".join(r.warnings) for r in rows],
        }
    )
 
 
def make_corrupted_copy(
    csv_path: str,
    out_path: str,
    limit: int = 10,
    cif_column: str = "cif.conv",
    corruption_mode: str = "aggressive",
) -> str:
    df = pd.read_csv(csv_path)
    df_out = df.head(limit).copy()
 
    for idx, row in df_out.iterrows():
        cif_text = row.get(cif_column)
        if pd.isna(cif_text):
            continue
        try:
            structure = Structure.from_str(str(cif_text), fmt="cif")
            corrupted = _corrupt_structure_for_negative_test(structure, mode=corruption_mode)
            df_out.at[idx, cif_column] = corrupted.to(fmt="cif")
        except Exception:
            continue
 
    df_out.to_csv(out_path, index=False)
    return out_path
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score CIF symmetry compliance from a CSV file."
    )
    parser.add_argument("csv_path", nargs="?", default="val.csv",
                        help="Path to CSV containing CIF text.")
    parser.add_argument("--limit", type=int, default=10,
                        help="Maximum number of rows to score.")
    parser.add_argument("--cif-column", type=str, default="cif.conv",
                        help="CSV column containing CIF text (default: cif.conv).")
    parser.add_argument("--sigma", type=float, default=0.25,
                        help="Gaussian distance-decay in Å (physically meaningful now).")
    parser.add_argument("--symprec", type=float, default=0.1,
                        help="Symmetry tolerance for SG detection.")
    parser.add_argument("--angle-tolerance", type=float, default=5.0,
                        help="Angular tolerance (degrees) for SG detection.")
    parser.add_argument("--out", type=str, default="",
                        help="Optional output CSV path for results.")
    parser.add_argument("--print-warnings", action="store_true",
                        help="Print per-structure warnings to stderr.")
    parser.add_argument("--make-corrupted-copy", type=str, default="",
                        help="Create a corrupted copy at this output path.")
    parser.add_argument("--verify-corruption", action="store_true",
                        help="Score original and corrupted copies and print comparison.")
    parser.add_argument("--corruption-mode", type=str, default="aggressive",
                        choices=["aggressive", "mild"],
                        help="Corruption style for --make-corrupted-copy.")
    args = parser.parse_args()
 
    if args.make_corrupted_copy:
        out_path = make_corrupted_copy(
            csv_path=args.csv_path,
            out_path=args.make_corrupted_copy,
            limit=args.limit,
            cif_column=args.cif_column,
            corruption_mode=args.corruption_mode,
        )
        print(f"Created corrupted copy: {out_path}")
 
        if args.verify_corruption:
            kw = dict(
                limit=args.limit, sigma=args.sigma,
                cif_column=args.cif_column, symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
                print_warnings=args.print_warnings,
            )
            original_df  = score_csv(args.csv_path, **kw)
            corrupted_df = score_csv(out_path, **kw)
 
            comparison = pd.DataFrame({
                "composition":        original_df["composition"],
                "claimed_space_group": original_df["claimed_space_group"],
                "original_score":     original_df["score"],
                "corrupted_score":    corrupted_df["score"],
            })
            comparison["delta"] = comparison["corrupted_score"] - comparison["original_score"]
            print("\nCorruption verification:")
            print(comparison.to_string(index=False))
            neg_count = int((comparison["corrupted_score"] < 0).sum())
            print(f"\nNegative scores after corruption: {neg_count}/{len(comparison)}")
        return
 
    result_df = score_csv(
        args.csv_path,
        limit=args.limit,
        sigma=args.sigma,
        cif_column=args.cif_column,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
        print_warnings=args.print_warnings,
    )
 
    pd.set_option("display.max_colwidth", 40)
    cols = ["composition", "claimed_space_group", "detected_space_group", "score"]
    if args.print_warnings:
        cols.append("warnings")
    print(result_df[cols].to_string(index=False))
 
    if args.out:
        result_df.to_csv(args.out, index=False)
        print(f"\nSaved results to: {args.out}")
 
 
if __name__ == "__main__":
    main()