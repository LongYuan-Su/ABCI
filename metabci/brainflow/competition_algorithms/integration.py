# -*- coding: utf-8 -*-
"""Glue code for the competition swallowing algorithms.

The original competition scripts are kept almost intact in this package:

* ``classification_train.py`` builds the part2 rest-vs-imagined-swallow model.
* ``classification_inference.py`` calls the trained model on part2 data.
* ``warm_prior_regression.py`` quantifies warm-water swallowing from part1 data.

This module provides small, GUI-friendly wrappers around those scripts.  The
wrappers avoid changing the model code and keep missing checkpoints as a
recoverable state, because the source folder provided with the scripts may not
include a trained ``final_model.pt``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


ALGO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ALGO_DIR.parents[2]
DEFAULT_MODEL_RELATIVE = Path(
    "output_009_tfr_raw_localglobal_first_ablation_final_model"
) / "final_model.pt"


@contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def _load_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    with _prepend_sys_path(script_path.parent):
        spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path and path.is_file():
            return path
    return None


def find_default_classifier_model(project_root: Path | str = PROJECT_ROOT) -> Optional[Path]:
    """Return the first available classifier checkpoint.

    Supported locations, in priority order:

    1. ``METABCI_SWALLOW_CLASSIFIER_MODEL`` environment variable.
    2. ``<project>/applications/swallow_bci/models/swallow_classifier/final_model.pt``.
    3. ``<project>/models/swallow_classifier/final_model.pt``.
    4. ``<project>/models/final_model.pt``.
    5. ``<project>/models/output_.../final_model.pt``.
    6. ``competition_algorithms/output_.../final_model.pt``.
    """
    root = Path(project_root)
    env_path = os.environ.get("METABCI_SWALLOW_CLASSIFIER_MODEL", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            root / "applications" / "swallow_bci" / "models" / "swallow_classifier" / "final_model.pt",
            root / "models" / "swallow_classifier" / "final_model.pt",
            root / "models" / "final_model.pt",
            root / "models" / DEFAULT_MODEL_RELATIVE,
            ALGO_DIR / DEFAULT_MODEL_RELATIVE,
        ]
    )
    return _existing(candidates)


def _result_base(
    status: str,
    patient_id: str,
    data_root: Path,
    output_dir: Path,
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": status,
        "patient_id": patient_id,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(extra)
    return payload


def run_part2_classification(
    recording_paths: Dict[str, str] | None,
    patient_id: str,
    project_root: Path | str = PROJECT_ROOT,
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Run the rest-vs-imagined-swallow classifier on a saved part2 session.

    Parameters
    ----------
    recording_paths:
        Recorder paths with ``npy``, ``labels`` and ``meta`` keys.
    patient_id:
        Current GUI patient id.
    project_root:
        Root of the integrated MetaBCI project.
    model_path:
        Optional trained ``final_model.pt``.  If omitted, common project
        locations and ``METABCI_SWALLOW_CLASSIFIER_MODEL`` are checked.
    output_dir:
        Optional output directory.  Defaults to
        ``<part2_data_dir>/classification_inference``.
    """
    if not recording_paths:
        raise ValueError("recording_paths is empty; no part2 EEG recording was saved.")

    npy_path = Path(recording_paths.get("npy", ""))
    labels_path = Path(recording_paths.get("labels", ""))
    meta_path = Path(recording_paths.get("meta", ""))
    if not npy_path.is_file():
        raise FileNotFoundError(f"NPY data not found: {npy_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels JSON not found: {labels_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Meta JSON not found: {meta_path}")

    data_root = npy_path.parent
    out_dir = Path(output_dir) if output_dir else data_root / "classification_inference"
    result_json = out_dir / "classification_integration_result.json"

    checkpoint = Path(model_path) if model_path else find_default_classifier_model(project_root)
    if not checkpoint or not checkpoint.is_file():
        candidates = [
            str(Path(project_root) / "applications" / "swallow_bci" / "models" / "swallow_classifier" / "final_model.pt"),
            str(Path(project_root) / "models" / "swallow_classifier" / "final_model.pt"),
            str(Path(project_root) / "models" / "final_model.pt"),
            str(Path(project_root) / "models" / DEFAULT_MODEL_RELATIVE),
            str(ALGO_DIR / DEFAULT_MODEL_RELATIVE),
        ]
        payload = _result_base(
            "skipped",
            patient_id,
            data_root,
            out_dir,
            reason="classifier_model_not_found",
            expected_model_candidates=candidates,
            hint=(
                "Train the model with classification_train.py, then put final_model.pt "
                "under applications/swallow_bci/models/swallow_classifier/ "
                "or set METABCI_SWALLOW_CLASSIFIER_MODEL."
            ),
        )
        _write_json(result_json, payload)
        return payload

    try:
        script_path = ALGO_DIR / "classification_inference.py"
        module = _load_module("metabci_competition_classification_inference", script_path)
        module.MODEL_PATH = str(checkpoint)
        module.PREDICT_DATA_ROOT = str(data_root)
        module.INFERENCE_OUTPUT_DIR = str(out_dir)
        module.main_inference()

        summary_path = out_dir / "prediction_summary.json"
        summary: Dict[str, Any] = {}
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        payload = _result_base(
            "success",
            patient_id,
            data_root,
            out_dir,
            model_path=str(checkpoint),
            prediction_summary=summary,
            predictions_csv=str(out_dir / "predictions.csv"),
        )
        _write_json(result_json, payload)
        return payload
    except Exception as exc:
        payload = _result_base(
            "failed",
            patient_id,
            data_root,
            out_dir,
            model_path=str(checkpoint),
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        _write_json(result_json, payload)
        return payload


def run_warm_prior_quantification(
    patient_id: str,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_score: float = 1.0,
    recording_paths: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Run the warm-prior quantification script on part1 warm-water data.

    This algorithm expects paired imagined-swallow and warm-water events, so it
    is intentionally not called automatically by the part2 workflow.
    """
    if recording_paths:
        npy_path = Path(recording_paths.get("npy", ""))
        labels_path = Path(recording_paths.get("labels", ""))
        meta_path = Path(recording_paths.get("meta", ""))
        if not npy_path.is_file():
            raise FileNotFoundError(f"NPY data not found: {npy_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels JSON not found: {labels_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Meta JSON not found: {meta_path}")
        data_path = npy_path.parent
    elif data_dir is not None:
        data_path = Path(data_dir)
        npy_path = labels_path = meta_path = None
    else:
        raise ValueError("Either data_dir or recording_paths must be provided.")

    out_dir = Path(output_dir) if output_dir else data_path / "warm_prior_quantification"
    target_csv = out_dir / f"{patient_id}_subject_score_target.csv"
    result_json = out_dir / "warm_prior_integration_result.json"

    try:
        script_path = ALGO_DIR / "warm_prior_regression.py"
        module = _load_module("metabci_competition_warm_prior_regression", script_path)
        if recording_paths:
            module.load_paths = lambda _cfg: (npy_path, labels_path, meta_path)
        cfg = module.ExperimentConfig(
            data_dir=str(data_path),
            target_csv=str(target_csv),
            output_dir=str(out_dir),
            subject_id=str(patient_id),
        )
        cfg.n_times = int(cfg.window_sec * cfg.sfreq)
        module.set_seed(cfg.random_state)
        module.safe_mkdir(cfg.output_dir)
        module.ensure_target_csv(cfg.target_csv, cfg.subject_id, score=target_score)

        target = module.read_target_score(cfg.target_csv, cfg.subject_id)
        warm_raw, prior_raw, rows = module.build_warm_prior_windows(cfg)
        standardizer = module.RawWindowStandardizer().fit(
            module.np.concatenate([warm_raw, prior_raw], axis=1)
        )
        warm_raw = standardizer.transform(warm_raw)
        prior_raw = standardizer.transform(prior_raw)
        warm_tf_np, prior_tf_np = module.load_or_compute_tfr_pair(
            warm_raw, prior_raw, cfg, rows
        )

        torch = module.torch
        nn = module.nn
        device = "cuda" if torch.cuda.is_available() else "cpu"
        warm_tf = torch.from_numpy(warm_tf_np).float().to(device)
        prior_tf = torch.from_numpy(prior_tf_np).float().to(device)
        target_tensor = torch.tensor([target], dtype=torch.float32, device=device)

        model = module.build_model(module.to_model_config(cfg)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        criterion = nn.MSELoss()

        best_loss = float("inf")
        best_score = 0.0
        best_weight = module.np.ones(cfg.n_tasks, dtype=module.np.float32) / float(cfg.n_tasks)
        for _ in range(cfg.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred, aux = model(warm_tf, prior_tf)
            loss = criterion(pred, target_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_score = float(pred.detach().cpu().item())
                best_weight = aux["task_weight"].detach().cpu().numpy()[0]

        result_path = out_dir / "fit_score.csv"
        module.write_result_csv(
            result_path, cfg.subject_id, target, best_score, best_loss, best_weight
        )
        payload = _result_base(
            "success",
            patient_id,
            data_path,
            out_dir,
            target_score_1point=float(target),
            fitted_score_1point=float(best_score),
            fit_loss=float(best_loss),
            fit_score_csv=str(result_path),
        )
        _write_json(result_json, payload)
        return payload
    except Exception as exc:
        payload = _result_base(
            "failed",
            patient_id,
            data_path,
            out_dir,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        _write_json(result_json, payload)
        return payload
