import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POST_DECODER_PATH = ROOT / "lib" / "models" / "sutrack" / "post_decoder_disambiguator.py"
SUTRACK_PATH = ROOT / "lib" / "models" / "sutrack" / "sutrack.py"
TRACKER_PATH = ROOT / "lib" / "test" / "tracker" / "sutrack.py"


def _assert_contains(text, snippet, label):
    if snippet not in text:
        raise AssertionError(f"Missing expected snippet for {label}: {snippet}")


def _assert_not_contains(text, snippet, label):
    if snippet in text:
        raise AssertionError(f"Found unexpected snippet for {label}: {snippet}")


def _load_sources():
    return (
        POST_DECODER_PATH.read_text(encoding="utf-8"),
        SUTRACK_PATH.read_text(encoding="utf-8"),
        TRACKER_PATH.read_text(encoding="utf-8"),
    )


def inspect_source_contracts():
    post_text, sutrack_text, tracker_text = _load_sources()

    _assert_contains(
        sutrack_text,
        "if self.encoder_postprocess is None:\n            return xz",
        "encoder direct output path",
    )
    _assert_contains(
        sutrack_text,
        "score_map_ctr, bbox, size_map, offset_map, f_map = self.decoder(feature, gt_score_map)",
        "center head output",
    )
    _assert_contains(
        sutrack_text,
        "refined_score_map, aux_info = self.post_disambiguator(",
        "post-decoder disambiguator call",
    )
    _assert_contains(
        sutrack_text,
        "if self.decoder_type == \"CENTER\":",
        "center decoder branch",
    )
    _assert_contains(
        sutrack_text,
        "return self.forward_decoder(feature), self.forward_task_decoder(feature)",
        "task decoder path preserved",
    )
    _assert_contains(
        post_text,
        "self.template_anchor_weight * sim_template_anchor +",
        "template-anchor score term",
    )
    _assert_contains(
        post_text,
        "self.first_frame_anchor_weight * sim_first_frame_anchor +",
        "first-frame score term",
    )
    _assert_contains(
        post_text,
        "self.mamba_history_weight * sim_mamba_history",
        "history score term",
    )
    _assert_contains(
        post_text,
        "if self.debug_history_only_rerank:",
        "history-only debug branch",
    )
    _assert_contains(
        post_text,
        "elif self.use_history_aware_rerank_score:",
        "formal history-aware branch",
    )
    _assert_contains(
        sutrack_text,
        "(id_margin > self.id_margin_thresh) &",
        "safe gate id margin",
    )
    _assert_contains(
        sutrack_text,
        "(switch_score_ratio > self.min_switch_score_ratio) &",
        "safe gate score ratio",
    )
    _assert_contains(
        sutrack_text,
        "(max_identity_sim > self.min_identity_sim) &",
        "safe gate identity sim",
    )
    _assert_contains(
        sutrack_text,
        "(motion_distance <= self.safe_gate_max_motion_dist)",
        "safe gate motion distance",
    )
    _assert_contains(
        tracker_text,
        "ambiguity_ok = ambiguity_ratio <= getattr(self.network, \"update_ratio_thresh\", 0.60)",
        "history update ambiguity guard",
    )
    _assert_contains(
        post_text,
        "peak_norm = F.normalize(peak_feats, p=2, dim=-1, eps=self.eps)",
        "peak normalization",
    )
    _assert_contains(
        post_text,
        "anchor_norm = F.normalize(anchor, p=2, dim=-1, eps=self.eps)",
        "anchor normalization",
    )
    _assert_contains(
        post_text,
        "token = target_feat.detach().unsqueeze(1)",
        "history token detach",
    )
    _assert_contains(
        post_text,
        "self.cached_tokens = seq.detach()",
        "cached token detach",
    )
    _assert_contains(
        post_text,
        "self.cached_history = history.detach()",
        "cached history detach",
    )
    _assert_not_contains(
        sutrack_text,
        "ambiguity_ratio_below_refine_ratio_thresh),",
        "legacy ambiguity gate branch in active code",
    )


def safe_gate_select(identity_scores, peak_scores, sim_template, sim_first, sim_history, peaks_xy, cfg):
    baseline_idx = 0
    best_id_idx = max(range(len(identity_scores)), key=lambda idx: identity_scores[idx])

    baseline_identity = identity_scores[baseline_idx]
    best_identity = identity_scores[best_id_idx]
    id_margin = best_identity - baseline_identity

    baseline_peak = peak_scores[baseline_idx]
    best_peak = peak_scores[best_id_idx]
    score_ratio = best_peak / (baseline_peak + 1e-6)

    max_identity_sim = max(
        sim_template[best_id_idx],
        sim_first[best_id_idx],
        sim_history[best_id_idx],
    )

    base_xy = peaks_xy[baseline_idx]
    best_xy = peaks_xy[best_id_idx]
    dist = math.sqrt((best_xy[0] - base_xy[0]) ** 2 + (best_xy[1] - base_xy[1]) ** 2)
    diag = math.sqrt(max(cfg["width"] - 1, 0) ** 2 + max(cfg["height"] - 1, 0) ** 2)
    motion_distance = 0.0 if diag <= 0 else dist / diag

    should_switch = (
        best_id_idx != baseline_idx
        and id_margin > cfg["id_margin_thresh"]
        and score_ratio > cfg["min_switch_score_ratio"]
        and max_identity_sim > cfg["min_identity_sim"]
        and motion_distance <= cfg["safe_gate_max_motion_dist"]
    )

    if should_switch:
        selected_idx = best_id_idx
        fallback_reason = "rerank_used"
    else:
        selected_idx = baseline_idx
        if best_id_idx == baseline_idx:
            fallback_reason = "identity_best_is_baseline"
        elif id_margin <= cfg["id_margin_thresh"]:
            fallback_reason = "id_margin_below_thresh"
        elif score_ratio <= cfg["min_switch_score_ratio"]:
            fallback_reason = "score_ratio_below_thresh"
        elif max_identity_sim <= cfg["min_identity_sim"]:
            fallback_reason = "identity_sim_below_thresh"
        elif motion_distance > cfg["safe_gate_max_motion_dist"]:
            fallback_reason = "motion_constraint_failed"
        else:
            fallback_reason = "safe_fallback"

    return {
        "selected_idx": selected_idx,
        "best_id_idx": best_id_idx,
        "id_margin": id_margin,
        "score_ratio": score_ratio,
        "max_identity_sim": max_identity_sim,
        "motion_distance": motion_distance,
        "fallback_reason": fallback_reason,
        "selected_center": peaks_xy[selected_idx],
        "baseline_center": peaks_xy[baseline_idx],
    }


def run_dry_run_cases():
    cfg = {
        "id_margin_thresh": 0.08,
        "min_switch_score_ratio": 0.60,
        "min_identity_sim": 0.65,
        "safe_gate_max_motion_dist": 0.35,
        "width": 20,
        "height": 20,
    }
    peaks_xy = [(10.0, 10.0), (12.0, 11.0)]

    case_switch = safe_gate_select(
        identity_scores=[0.50, 0.72],
        peak_scores=[0.95, 0.70],
        sim_template=[0.58, 0.81],
        sim_first=[0.61, 0.88],
        sim_history=[0.55, 0.79],
        peaks_xy=peaks_xy,
        cfg=cfg,
    )
    if case_switch["selected_idx"] != 1 or case_switch["fallback_reason"] != "rerank_used":
        raise AssertionError(f"Case switch failed: {case_switch}")

    case_margin_fail = safe_gate_select(
        identity_scores=[0.50, 0.56],
        peak_scores=[0.95, 0.70],
        sim_template=[0.58, 0.81],
        sim_first=[0.61, 0.88],
        sim_history=[0.55, 0.79],
        peaks_xy=peaks_xy,
        cfg=cfg,
    )
    if case_margin_fail["selected_idx"] != 0 or case_margin_fail["fallback_reason"] != "id_margin_below_thresh":
        raise AssertionError(f"Case id-margin fallback failed: {case_margin_fail}")

    case_baseline = safe_gate_select(
        identity_scores=[0.73, 0.72],
        peak_scores=[0.95, 0.70],
        sim_template=[0.82, 0.81],
        sim_first=[0.89, 0.88],
        sim_history=[0.80, 0.79],
        peaks_xy=peaks_xy,
        cfg=cfg,
    )
    if case_baseline["selected_idx"] != 0:
        raise AssertionError(f"Case baseline selected_idx failed: {case_baseline}")
    if case_baseline["selected_center"] != case_baseline["baseline_center"]:
        raise AssertionError(f"Case baseline center mismatch: {case_baseline}")
    if case_baseline["fallback_reason"] != "identity_best_is_baseline":
        raise AssertionError(f"Case baseline fallback failed: {case_baseline}")

    return [case_switch, case_margin_fail, case_baseline]


def main():
    inspect_source_contracts()
    cases = run_dry_run_cases()
    print("[DRY-RUN OK] source inspection passed")
    for idx, case in enumerate(cases, start=1):
        print(
            f"[DRY-RUN CASE {idx}] selected_idx={case['selected_idx']} "
            f"best_id_idx={case['best_id_idx']} fallback_reason={case['fallback_reason']} "
            f"id_margin={case['id_margin']:.4f} score_ratio={case['score_ratio']:.4f} "
            f"max_identity_sim={case['max_identity_sim']:.4f} motion_distance={case['motion_distance']:.4f}"
        )


if __name__ == "__main__":
    main()
