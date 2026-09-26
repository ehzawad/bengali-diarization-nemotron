"""Build the phase-3b final artefact descriptors under runs/final/ (PREREG_phase3b "Default system and
final artefact"). CPU only, read-only on every input.

Writes
* ``runs/final/final_config.json``  the default system + decoding config consumed by :mod:`bnd.diarize`,
  with full provenance (results.json / selection.json / frozen identity hashes, the five inner
  selections, the joint-mode computation, Q1 verdict, disclosures).
* ``runs/final/model_reference.json``  path + sha256 of the default model (ZS: the pinned HF base
  revision; FT: the all-10 model, which this module refuses to invent -- it must exist).
* ``runs/final/ft_refits_provenance.json``  the phase-2 fine-tuned refits (kept regardless of the
  default: the owner wanted a fine-tuned model), each with sha256 re-verified against its fit marker,
  training recordings, recipe, and the plain statement of whether FT beat ZS under equal decoding.

The rule is applied mechanically from runs/p3b/results.json: Default = FT iff Q1 'fine-tuning helps';
final configuration = joint mode of the default system's five inner selections (ties -> simpler).

Run:  CUDA_VISIBLE_DEVICES="" <venv python> -m bnd.final [--out runs/final]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

from bnd import embed, infer
from bnd import p3b

REPO = Path(__file__).resolve().parents[1]
P3B = REPO / "runs/p3b"
HF_MODEL_ID = "nvidia/Nemotron-3-Diarization"
HF_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"   # docs/SPEC.md "Environment (fixed)"
BASE_SHA256 = "867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d"  # runs/cv/zeroshot_provenance.json
ALL10_MODEL = REPO / "runs/final/final_all10.nemo"
FT_REFITS = {  # phase-2 registered systems (runs/cv/selection.json; PREREG_phase3b "Systems")
    "selected": {"F1": "lr5e-5_raw", "F2": "lr5e-5_shift", "F3": "lr5e-5_shift", "F4": "lr5e-5_shift",
                 "F5": "lr5e-5_shift"},
    "strict": {f"F{i}": "lr1e-5_raw" for i in range(1, 6)},
}


class FinalError(RuntimeError):
    pass


def sha_file(p: Path) -> str:
    return infer.sha256_file(p)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def check_results(results_p: Path, scored_p: Path) -> dict:
    """results.json must be the one outer scoring run recorded in scored.jsonl (exactly one line)."""
    lines = [json.loads(x) for x in scored_p.read_text().splitlines() if x.strip()]
    if len(lines) != 1:
        raise FinalError(f"{scored_p}: expected exactly one outer scoring, found {len(lines)}")
    got = sha_file(results_p)
    if got != lines[0]["results_sha256"]:
        raise FinalError(f"{results_p} sha {got} != scored.jsonl {lines[0]['results_sha256']}")
    return lines[0]


def decide(results: dict, selection: dict) -> dict:
    """Apply the registered default rule; cross-check against what the score stage recorded."""
    helps = bool(results["q1"]["finetuning_helps"])
    system = "FT" if helps else "ZS"
    key = system.lower()
    per_fold = {fi: selection["folds"][fi][key]["selected"] for fi in sorted(selection["folds"])}
    if len(per_fold) != 5:
        raise FinalError(f"expected 5 outer folds, got {sorted(per_fold)}")
    jm = p3b.joint_mode(list(per_fold.values()))
    rec = results["default_system"]
    if rec["system"] != system or rec["final_configuration"]["config"] != jm["config"]:
        raise FinalError(f"recomputed default {system}/{jm['config']} != results.json {rec['system']}/"
                         f"{rec['final_configuration']['config']}")
    return {"system": system, "finetuning_helps": helps, "per_fold_selection": per_fold, "joint_mode": jm}


def model_reference(system: str) -> dict:
    if system == "ZS":
        path = infer.DEFAULT_MODEL
        sha = sha_file(path)
        if sha != BASE_SHA256:
            raise FinalError(f"pinned base model sha {sha} != {BASE_SHA256}")
        return {"system": "ZS", "kind": "pinned base checkpoint (no training)", "path": str(path),
                "path_rel": str(path.relative_to(REPO)), "sha256": sha, "hf_model_id": HF_MODEL_ID,
                "hf_revision": HF_REVISION, "hf_file": "Nemotron-3-Diarization.nemo",
                "class": "nemo.collections.asr.models.SortformerEncLabelModel", "restore": "strict=True",
                "nemo_commit": infer.nemo_commit()}
    if not ALL10_MODEL.exists():
        raise FinalError(f"default is FT but {ALL10_MODEL} does not exist: train final_all10 first")
    return {"system": "FT", "kind": "final_all10 fine-tune", "path": str(ALL10_MODEL),
            "path_rel": str(ALL10_MODEL.relative_to(REPO)), "sha256": sha_file(ALL10_MODEL),
            "base_hf_model_id": HF_MODEL_ID, "base_hf_revision": HF_REVISION, "base_sha256": BASE_SHA256}


def ft_refit_provenance(q1: dict) -> dict:
    out = {}
    for group, recipes in FT_REFITS.items():
        for fi, recipe in recipes.items():
            tag = f"cv_refit_{fi}_{recipe}"
            model = REPO / "runs/models" / f"{tag}.nemo"
            marker = load_json(REPO / "runs/cv/done/fit" / f"{tag}.json")
            report = load_json(REPO / "runs/fits" / tag / "fit_report.json")
            sha = sha_file(model)
            if sha != marker["outputs"]["model_sha256"]:
                raise FinalError(f"{model}: sha {sha} != fit marker {marker['outputs']['model_sha256']}")
            inp = marker["inputs"]
            out[tag] = {
                "phase2_role": f"{group} system, outer fold {fi}",
                "path_rel": str(model.relative_to(REPO)), "sha256": sha,
                "base_model_sha256": inp["base_model_sha256"], "base_hf_revision": HF_REVISION,
                "train_recordings": inp["train_recs"], "held_out_outer_fold": fi,
                "lr": inp["lr"], "timing": inp["timing"], "delta_s": marker["outputs"].get("delta_s"),
                "manifest_sha256": inp["manifest_sha256"], "code_sha256": inp.get("code"),
                "optimizer_steps": report.get("optimizer_steps"), "fit_wall_s": report.get("fit_wall_s"),
                "device": report.get("device"), "fit_completed": marker.get("completed"),
            }
    return {
        "note": ("Phase-2 fine-tuned refits, each trained on 8 of the 10 labelled recordings (its outer fold "
                 "held out). None was trained on all 10. Kept and documented regardless of the default "
                 "system; not the default."),
        "did_ft_beat_zs_under_equal_decoding": False,
        "q1_statement": ("No. Under equal, inner-selected decoding (PREREG_phase3b Q1), fine-tuning shows no "
                         "demonstrated benefit over calibrated zero-shot decoding: corpus DER collar 0 "
                         f"ZS* {q1['zs']:.2f}% vs FT* {q1['ft']:.2f}% (delta {q1['delta']:+.2f} pp, 95% CI "
                         f"[{q1['lo']:+.2f}, {q1['hi']:+.2f}], FT better on {q1['wins']}/10). The CI spans 0, "
                         "so harm is not shown either."),
        "refits": out,
    }


def q1_numbers(results: dict) -> dict:
    q = results["q1"]
    conds = q["conditions"]
    b = q["by_collar"]["0.0"]
    return {"collar": 0.0, "zs": 100 * b["der_a"], "ft": 100 * b["der_b"], "delta": 100 * b["delta_b_minus_a"],
            "lo": 100 * b["ci_low"], "hi": 100 * b["ci_high"], "wins": b["b_wins"], "losses": b["b_losses"],
            "macro_pp": conds["macro_mean_delta_lt_0"]["value_pp"],
            "conditions_pass": {k: v["pass"] for k, v in conds.items()},
            "harm": q["harm"], "verdict": q["verdict"]}


def build(out: Path) -> dict:
    results_p, sel_p, scored_p, frozen_p = (P3B / "results.json", P3B / "selection.json", P3B / "scored.jsonl",
                                            P3B / "frozen.json")
    scored = check_results(results_p, scored_p)
    results, selection = load_json(results_p), load_json(sel_p)
    dec = decide(results, selection)
    cfg = p3b.parse_key(dec["joint_mode"]["config"])
    q1 = q1_numbers(results)
    mref = model_reference(dec["system"])
    now = _dt.datetime.now().isoformat(timespec="seconds")
    final = {
        "created": now,
        "prereg": "docs/PREREG_phase3b.md",
        "default_system": dec["system"],
        "config_key": cfg.key,
        "decoding": {
            "session_core_s": cfg.session, "context_s": infer.CONTEXT_S,
            "native_streaming": infer.OFFLINE, "precision": "bf16-autocast", "batch": 1,
            "theta_act": cfg.theta_act, "min_turn_s": infer.MIN_TURN_S, "weak_usable_s": infer.WEAK_S,
            "linker": cfg.linker, "theta_clu": cfg.theta_clu, "gap_fill_s": float(cfg.gap),
            "decoder": "phase-2 exclusive timeline (bnd.link.frame_labels) then same-speaker gap-fill",
            "sample_rate": infer.SR,
        },
        "model": {"sha256": mref["sha256"], "path_rel": mref["path_rel"], "reference": "model_reference.json"},
        "embedder": {"model": embed.MODEL_ID, "revision": embed.MODEL_REV, "crop_policy": embed.CROP_POLICY},
        "provenance": {
            "rule": results["default_system"]["rule"],
            "q1": q1,
            "finetuning_helps": dec["finetuning_helps"],
            "per_fold_inner_selection": dec["per_fold_selection"],
            "joint_mode": dec["joint_mode"],
            "results_json_sha256": scored["results_sha256"],
            "frozen_identity": scored["frozen_identity"],
            "scored": scored["scored"],
            "selection_json_sha256": sha_file(sel_p),
            "frozen_json_sha256": sha_file(frozen_p),
            "code_sha256": {m: sha_file(REPO / "bnd" / f"{m}.py") for m in ("infer", "embed", "link", "p3b", "diarize")
                            if (REPO / "bnd" / f"{m}.py").exists()},
            "nemo_commit": infer.nemo_commit(),
        },
        "disclosures": [
            "SECOND-LOOK: every phase-3b factor (session length, theta_act < 0.40, refine linker, gap-fill) was "
            "motivated by phase-2 OUTER-fold diagnostics; selection used inner folds only.",
            "Any DER of this configuration over all 10 labelled recordings is a DEVELOPMENT number, never held-out.",
            "One training seed; 10 recordings; coarse references (all onsets/durations whole seconds).",
            "Speaker over-counting: the selected ZS system predicted sum |pred-ref| = 192 speakers over the 10 "
            "outer recordings (e.g. 68 vs 17 on train_006).",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("final_config.json", final), ("model_reference.json", mref),
                      ("ft_refits_provenance.json", ft_refit_provenance(q1))):
        infer.write_json_atomic(out / name, obj)
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "runs/final")
    a = ap.parse_args(argv)
    f = build(a.out)
    print(json.dumps({"default_system": f["default_system"], "config": f["config_key"],
                      "model_sha256": f["model"]["sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
