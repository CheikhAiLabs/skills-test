from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from churn_ml.artifacts import (
    load_bundle,
    save_bundle,
    write_json,
    write_markdown_report,
    write_sample_payload,
)
from churn_ml.config import PipelineConfig
from churn_ml.data import (
    coerce_schema,
    dataset_summary,
    load_raw_data,
    split_dataset,
    validate_dataframe,
)
from churn_ml.evaluation import (
    compute_binary_metrics,
    enforce_model_gates,
    evaluate_baseline,
    select_operating_threshold,
)
from churn_ml.mlflow_tracking import MLflowTracker
from churn_ml.modeling import build_model_pipeline, extract_feature_importances
from churn_ml.tracking import RunTracker

LOGGER = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    name: str
    estimator: str
    params: dict[str, Any]
    selection_score: float
    threshold: float
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    final_model: Any
    candidate_dir: Path
    feature_importance_path: Path | None
    predictions_path: Path


def _sanitize_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.tracker = RunTracker(config.data.tracking_path)
        self.mlflow = MLflowTracker(config)
        self.mlflow.enable_sklearn_autolog()

    def _create_run_dir(self, run_name_suffix: str | None = None) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"-{_sanitize_name(run_name_suffix)}" if run_name_suffix else ""
        run_dir = self.config.data.artifact_root / f"{self.config.run_name_prefix}-{timestamp}{suffix}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "data").mkdir(parents=True, exist_ok=True)
        (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
        return run_dir

    def _persist_splits(self, run_dir: Path, splits: dict[str, pd.DataFrame]) -> None:
        for split_name, frame in splits.items():
            frame.to_csv(run_dir / "data" / f"{split_name}.csv", index=False)

    def _candidate_params(self, candidate: dict[str, Any], *, effective_random_state: int) -> dict[str, Any]:
        params = dict(candidate.get("params", {}))
        if candidate["estimator"] in {"logistic_regression", "random_forest", "extra_trees"}:
            params.setdefault("random_state", effective_random_state)
        return params

    def _build_report(
        self,
        *,
        run_id: str,
        summary: dict[str, Any],
        best_candidate: CandidateResult,
        baseline_metrics: dict[str, Any],
        leaderboard: pd.DataFrame,
    ) -> list[str]:
        top_rows = leaderboard.head(5).to_dict(orient="records")
        return [
            f"# Churn Evaluation Report: {run_id}",
            "",
            "## Dataset Summary",
            f"- rows: {summary['rows']}",
            f"- columns: {summary['columns']}",
            f"- positive_rate: {summary['positive_rate']}",
            "",
            "## Selected Candidate",
            f"- candidate: {best_candidate.name}",
            f"- estimator: {best_candidate.estimator}",
            f"- threshold: {best_candidate.threshold}",
            *(f"- validation_{key}: {value}" for key, value in best_candidate.validation_metrics.items()),
            *(f"- test_{key}: {value}" for key, value in best_candidate.test_metrics.items()),
            "",
            "## Baseline Metrics",
            *(f"- {key}: {value}" for key, value in baseline_metrics.items()),
            "",
            "## Leaderboard",
            *(
                f"- {row['candidate_name']} | validation_{self.config.training.selection_metric}={row['validation_score']} | "
                f"test_roc_auc={row['test_roc_auc']} | test_f1={row['test_f1']}"
                for row in top_rows
            ),
        ]

    def _log_parent_run_context(
        self,
        *,
        run_dir: Path,
        run_id: str,
        effective_random_state: int,
        summary: dict[str, Any],
    ) -> None:
        self.mlflow.log_params(
            {
                "run_id": run_id,
                "effective_random_state": effective_random_state,
                "selection_metric": self.config.training.selection_metric,
                "candidate_count": len(self.config.training.candidate_models),
                "feature_count": len(self.config.model.feature_columns),
                "train_fraction": self.config.data.train_fraction,
                "validation_fraction": self.config.data.validation_fraction,
                "test_fraction": self.config.data.test_fraction,
            }
        )
        self.mlflow.log_params(
            {
                "rows": summary["rows"],
                "columns": summary["columns"],
                "positive_rate": summary["positive_rate"],
            },
            prefix="dataset_summary",
        )
        self.mlflow.log_json_text(summary, "dataset_summary.json")
        self.mlflow.log_artifact(run_dir / "dataset_summary.json")

    def _persist_candidate_diagnostics(
        self,
        *,
        candidate_dir: Path,
        y_true: Any,
        probabilities: Any,
        threshold: float,
    ) -> dict[str, Path]:
        roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true, probabilities)
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, probabilities)

        diagnostics = {
            "roc_curve": candidate_dir / "roc_curve.csv",
            "precision_recall_curve": candidate_dir / "precision_recall_curve.csv",
            "confusion_matrix": candidate_dir / "confusion_matrix.json",
        }
        pd.DataFrame(
            {
                "false_positive_rate": roc_fpr,
                "true_positive_rate": roc_tpr,
                "threshold": roc_thresholds,
            }
        ).to_csv(diagnostics["roc_curve"], index=False)
        pd.DataFrame(
            {
                "precision": pr_precision[:-1],
                "recall": pr_recall[:-1],
                "threshold": pr_thresholds,
            }
        ).to_csv(diagnostics["precision_recall_curve"], index=False)
        write_json(
            diagnostics["confusion_matrix"],
            {
                "threshold": threshold,
                "true_negatives": int(((probabilities < threshold) & (y_true == 0)).sum()),
                "false_positives": int(((probabilities >= threshold) & (y_true == 0)).sum()),
                "false_negatives": int(((probabilities < threshold) & (y_true == 1)).sum()),
                "true_positives": int(((probabilities >= threshold) & (y_true == 1)).sum()),
            },
        )
        return diagnostics

    def _train_candidate(
        self,
        *,
        candidate: dict[str, Any],
        candidate_index: int,
        run_dir: Path,
        X_train: pd.DataFrame,
        y_train: Any,
        X_validation: pd.DataFrame,
        y_validation: Any,
        X_train_val: pd.DataFrame,
        y_train_val: Any,
        X_test: pd.DataFrame,
        y_test: Any,
        baseline_metrics: dict[str, Any],
        effective_random_state: int,
    ) -> CandidateResult:
        candidate_name = candidate["name"]
        estimator_name = candidate["estimator"]
        params = self._candidate_params(candidate, effective_random_state=effective_random_state)
        candidate_slug = f"{candidate_index:02d}_{_sanitize_name(candidate_name)}"
        candidate_dir = run_dir / "candidates" / candidate_slug
        candidate_dir.mkdir(parents=True, exist_ok=True)
        feature_importance_path = None
        predictions_path = candidate_dir / "test_predictions.csv"

        with self.mlflow.start_run(
            run_name=f"candidate-{candidate_slug}",
            tags={
                "candidate_name": candidate_name,
                "estimator": estimator_name,
                "selection_metric": self.config.training.selection_metric,
            },
            nested=True,
        ):
            LOGGER.info(
                "Training candidate %s using %s with params=%s",
                candidate_name,
                estimator_name,
                params,
            )
            self.mlflow.log_params(
                {
                    "candidate_name": candidate_name,
                    "estimator": estimator_name,
                    **params,
                }
            )
            warmup_model = build_model_pipeline(
                self.config,
                estimator_name=estimator_name,
                estimator_params=params,
            )
            warmup_model.fit(X_train, y_train)
            validation_probabilities = warmup_model.predict_proba(X_validation)[:, 1]
            threshold, validation_metrics = select_operating_threshold(
                y_validation,
                validation_probabilities,
                min_precision=self.config.thresholding.min_precision,
                min_recall=self.config.thresholding.min_recall,
            )

            final_model = build_model_pipeline(
                self.config,
                estimator_name=estimator_name,
                estimator_params=params,
            )
            final_model.fit(X_train_val, y_train_val)
            test_probabilities = final_model.predict_proba(X_test)[:, 1]
            test_metrics = compute_binary_metrics(y_test, test_probabilities, threshold)

            feature_importances = extract_feature_importances(final_model)
            if not feature_importances.empty:
                feature_importance_path = candidate_dir / "feature_importance.csv"
                feature_importances.to_csv(feature_importance_path, index=False)

            prediction_frame = pd.DataFrame(
                {
                    "y_true": y_test,
                    "churn_probability": test_probabilities,
                    "predicted_churn": (test_probabilities >= threshold).astype(int),
                }
            )
            prediction_frame.to_csv(predictions_path, index=False)
            diagnostics = self._persist_candidate_diagnostics(
                candidate_dir=candidate_dir,
                y_true=y_test,
                probabilities=test_probabilities,
                threshold=threshold,
            )
            candidate_summary = {
                "candidate_name": candidate_name,
                "estimator": estimator_name,
                "params": params,
                "threshold": threshold,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "baseline_metrics": baseline_metrics,
            }
            write_json(candidate_dir / "metrics.json", candidate_summary)
            self.mlflow.log_metrics(validation_metrics, prefix="validation")
            self.mlflow.log_metrics(test_metrics, prefix="test")
            self.mlflow.log_metrics(baseline_metrics, prefix="baseline")
            self.mlflow.log_json_text(candidate_summary, "candidate_summary.json")
            self.mlflow.log_artifact(candidate_dir / "metrics.json", artifact_path="candidate")
            self.mlflow.log_artifact(predictions_path, artifact_path="candidate")
            for diagnostic_name, diagnostic_path in diagnostics.items():
                self.mlflow.log_artifact(diagnostic_path, artifact_path=f"candidate/{diagnostic_name}")
            if feature_importance_path is not None:
                self.mlflow.log_artifact(feature_importance_path, artifact_path="candidate")
            if self.config.training.log_model_candidates:
                self.mlflow.log_model(
                    model=final_model,
                    artifact_name=f"candidate-model-{candidate_slug}",
                    input_example=X_test.head(self.config.packaging.sample_payload_rows),
                    predictions=final_model.predict(
                        X_test.head(self.config.packaging.sample_payload_rows)
                    ),
                )

        return CandidateResult(
            name=candidate_name,
            estimator=estimator_name,
            params=params,
            selection_score=float(validation_metrics[self.config.training.selection_metric]),
            threshold=threshold,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            final_model=final_model,
            candidate_dir=candidate_dir,
            feature_importance_path=feature_importance_path,
            predictions_path=predictions_path,
        )

    def run(
        self,
        *,
        random_state: int | None = None,
        run_name_suffix: str | None = None,
        nested_mlflow_run: bool = False,
        run_tags: dict[str, str] | None = None,
    ) -> Path:
        effective_random_state = (
            self.config.data.random_state if random_state is None else random_state
        )
        raw_data = load_raw_data(self.config.data.raw_data_path)
        prepared = coerce_schema(raw_data, self.config)
        validate_dataframe(prepared, self.config)
        summary = dataset_summary(prepared, self.config)

        run_dir = self._create_run_dir(run_name_suffix=run_name_suffix)
        run_id = run_dir.name
        LOGGER.info("Starting pipeline run %s with random_state=%s", run_id, effective_random_state)
        write_json(run_dir / "dataset_summary.json", summary)

        with self.mlflow.start_run(
            run_name=run_id,
            tags={
                "run_id": run_id,
                "pipeline_mode": "training",
                "random_state": str(effective_random_state),
                **(run_tags or {}),
            },
            nested=nested_mlflow_run,
        ):
            self._log_parent_run_context(
                run_dir=run_dir,
                run_id=run_id,
                effective_random_state=effective_random_state,
                summary=summary,
            )

            splits = split_dataset(prepared, self.config, random_state=effective_random_state)
            split_mapping = {
                "train": splits.train,
                "validation": splits.validation,
                "test": splits.test,
            }
            self._persist_splits(run_dir, split_mapping)
            for split_name in split_mapping:
                self.mlflow.log_artifact(run_dir / "data" / f"{split_name}.csv", artifact_path="splits")

            target_column = self.config.validation.target_column
            feature_columns = self.config.model.feature_columns

            X_train = splits.train[feature_columns]
            y_train = splits.train[target_column].astype(int).to_numpy()
            X_validation = splits.validation[feature_columns]
            y_validation = splits.validation[target_column].astype(int).to_numpy()
            X_test = splits.test[feature_columns]
            y_test = splits.test[target_column].astype(int).to_numpy()

            X_train_val = pd.concat([X_train, X_validation], axis=0).reset_index(drop=True)
            y_train_val = (
                pd.concat(
                    [
                        splits.train[target_column].astype(int),
                        splits.validation[target_column].astype(int),
                    ],
                    axis=0,
                )
                .reset_index(drop=True)
                .to_numpy()
            )

            baseline_metrics = evaluate_baseline(X_train_val, y_train_val, X_test, y_test)
            self.mlflow.log_metrics(baseline_metrics, prefix="baseline")

            candidate_results = [
                self._train_candidate(
                    candidate=candidate,
                    candidate_index=index,
                    run_dir=run_dir,
                    X_train=X_train,
                    y_train=y_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    X_train_val=X_train_val,
                    y_train_val=y_train_val,
                    X_test=X_test,
                    y_test=y_test,
                    baseline_metrics=baseline_metrics,
                    effective_random_state=effective_random_state + index,
                )
                for index, candidate in enumerate(self.config.training.candidate_models, start=1)
            ]

            selection_metric = self.config.training.selection_metric
            best_candidate = max(
                candidate_results,
                key=lambda result: (
                    result.validation_metrics[selection_metric],
                    result.test_metrics[selection_metric],
                ),
            )
            LOGGER.info(
                "Selected candidate %s with validation_%s=%.4f and test_roc_auc=%.4f",
                best_candidate.name,
                selection_metric,
                best_candidate.validation_metrics[selection_metric],
                best_candidate.test_metrics["roc_auc"],
            )
            enforce_model_gates(best_candidate.test_metrics, baseline_metrics, self.config)

            leaderboard = pd.DataFrame(
                [
                    {
                        "candidate_name": candidate.name,
                        "estimator": candidate.estimator,
                        "validation_score": candidate.validation_metrics[selection_metric],
                        "test_roc_auc": candidate.test_metrics["roc_auc"],
                        "test_f1": candidate.test_metrics["f1"],
                        "threshold": candidate.threshold,
                    }
                    for candidate in candidate_results
                ]
            ).sort_values(["validation_score", "test_roc_auc"], ascending=False)
            leaderboard_path = run_dir / "leaderboard.csv"
            leaderboard.to_csv(leaderboard_path, index=False)
            self.mlflow.log_artifact(leaderboard_path, artifact_path="leaderboard")
            leaderboard_markdown = "\n".join(
                [
                    "# Candidate Leaderboard",
                    "",
                    *(
                        f"- {row['candidate_name']} | estimator={row['estimator']} | "
                        f"validation_{selection_metric}={row['validation_score']:.4f} | "
                        f"test_roc_auc={row['test_roc_auc']:.4f} | test_f1={row['test_f1']:.4f} | "
                        f"threshold={row['threshold']:.2f}"
                        for row in leaderboard.to_dict(orient="records")
                    ),
                ]
            )
            self.mlflow.log_text(leaderboard_markdown, "leaderboard/leaderboard.md")
            self.mlflow.set_tags(
                {
                    "selected_candidate": best_candidate.name,
                    "selected_estimator": best_candidate.estimator,
                    "selection_metric": selection_metric,
                }
            )

            bundle_payload = {
                "model": best_candidate.final_model,
                "threshold": best_candidate.threshold,
                "feature_columns": feature_columns,
                "numeric_features": self.config.model.numeric_features,
                "categorical_features": self.config.model.categorical_features,
                "boolean_features": self.config.model.boolean_features,
                "entity_id_column": self.config.validation.entity_id_column,
                "target_column": target_column,
                "selected_candidate_name": best_candidate.name,
                "estimator": best_candidate.estimator,
                "estimator_params": best_candidate.params,
                "metrics": {
                    "validation": best_candidate.validation_metrics,
                    "test": best_candidate.test_metrics,
                    "baseline": baseline_metrics,
                },
                "run_id": run_id,
                "mlflow_run_id": self.mlflow.active_run_id(),
                "created_at_utc": datetime.now(UTC).isoformat(),
                "config": json.loads(json.dumps(asdict(self.config), default=str)),
            }
            save_bundle(run_dir / "model_bundle.joblib", bundle_payload)
            write_json(run_dir / "metrics.json", bundle_payload["metrics"])
            write_sample_payload(
                run_dir / "sample_payload.json",
                X_test.head(self.config.packaging.sample_payload_rows),
            )

            report_lines = self._build_report(
                run_id=run_id,
                summary=summary,
                best_candidate=best_candidate,
                baseline_metrics=baseline_metrics,
                leaderboard=leaderboard,
            )
            write_markdown_report(run_dir / "model_card.md", report_lines)
            write_markdown_report(self.config.data.report_path, report_lines)
            run_summary = {
                "run_id": run_id,
                "selected_candidate": best_candidate.name,
                "selected_estimator": best_candidate.estimator,
                "threshold": best_candidate.threshold,
                "validation_metrics": best_candidate.validation_metrics,
                "test_metrics": best_candidate.test_metrics,
                "baseline_metrics": baseline_metrics,
                "leaderboard": leaderboard.to_dict(orient="records"),
                "artifact_dir": str(run_dir),
            }
            write_json(run_dir / "run_summary.json", run_summary)

            self.mlflow.log_metrics(best_candidate.validation_metrics, prefix="best_validation")
            self.mlflow.log_metrics(best_candidate.test_metrics, prefix="best_test")
            self.mlflow.log_artifact(run_dir / "metrics.json", artifact_path="bundle")
            self.mlflow.log_artifact(run_dir / "sample_payload.json", artifact_path="bundle")
            self.mlflow.log_artifact(run_dir / "model_card.md", artifact_path="bundle")
            self.mlflow.log_artifact(run_dir / "model_bundle.joblib", artifact_path="bundle")
            self.mlflow.log_artifact(run_dir / "run_summary.json", artifact_path="bundle")
            self.mlflow.log_artifact(best_candidate.predictions_path, artifact_path="bundle/selected_candidate")
            if best_candidate.feature_importance_path is not None:
                self.mlflow.log_artifact(
                    best_candidate.feature_importance_path,
                    artifact_path="bundle/selected_candidate",
                )
            self.mlflow.log_model(
                model=best_candidate.final_model,
                artifact_name="best-model",
                input_example=X_test.head(self.config.packaging.sample_payload_rows),
                predictions=best_candidate.final_model.predict(
                    X_test.head(self.config.packaging.sample_payload_rows)
                ),
                registered_model_name=self.config.training.registered_model_name,
            )

            self.tracker.append(
                {
                    "run_id": run_id,
                    "created_at_utc": bundle_payload["created_at_utc"],
                    "summary": summary,
                    "selected_candidate": best_candidate.name,
                    "selected_estimator": best_candidate.estimator,
                    "validation_metrics": best_candidate.validation_metrics,
                    "test_metrics": best_candidate.test_metrics,
                    "baseline_metrics": baseline_metrics,
                    "bundle_path": str((run_dir / "model_bundle.joblib").resolve()),
                    "mlflow_run_id": bundle_payload["mlflow_run_id"],
                }
            )

        LOGGER.info("Finished pipeline run %s", run_id)
        return run_dir


def retrain(
    config: PipelineConfig,
    *,
    rounds: int,
    promote_best: bool = True,
) -> dict[str, Any]:
    runner = PipelineRunner(config)
    results: list[dict[str, Any]] = []
    campaign_name = f"retraining-campaign-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    with runner.mlflow.start_run(
        run_name=campaign_name,
        tags={
            "pipeline_mode": "retraining_campaign",
            "round_count": str(rounds),
            "promote_best": str(promote_best),
        },
    ):
        runner.mlflow.log_params(
            {
                "rounds": rounds,
                "promote_best": promote_best,
                "selection_metric": config.training.selection_metric,
            }
        )
        for round_index in range(1, rounds + 1):
            random_state = config.data.random_state + round_index - 1
            LOGGER.info(
                "Starting retraining round %s/%s with random_state=%s",
                round_index,
                rounds,
                random_state,
            )
            run_dir = runner.run(
                random_state=random_state,
                run_name_suffix=f"round-{round_index}",
                nested_mlflow_run=runner.mlflow.enabled,
                run_tags={
                    "retraining_round": str(round_index),
                    "retraining_campaign": campaign_name,
                },
            )
            bundle = load_bundle(run_dir / "model_bundle.joblib")
            results.append(
                {
                    "round_index": round_index,
                    "random_state": random_state,
                    "run_dir": str(run_dir),
                    "run_id": bundle["run_id"],
                    "selected_candidate": bundle["selected_candidate_name"],
                    "selected_estimator": bundle["estimator"],
                    "test_metrics": bundle["metrics"]["test"],
                    "validation_metrics": bundle["metrics"]["validation"],
                    "mlflow_run_id": bundle.get("mlflow_run_id"),
                }
            )

        selection_metric = config.training.selection_metric
        best_result = max(
            results,
            key=lambda result: result["test_metrics"][selection_metric],
        )
        promotion_run_dir = None
        if promote_best and rounds > 1:
            LOGGER.info(
                "Promotion run for best round=%s candidate=%s random_state=%s",
                best_result["round_index"],
                best_result["selected_candidate"],
                best_result["random_state"],
            )
            promotion_run_dir = runner.run(
                random_state=best_result["random_state"],
                run_name_suffix="promotion",
                nested_mlflow_run=runner.mlflow.enabled,
                run_tags={
                    "retraining_role": "promotion",
                    "source_round": str(best_result["round_index"]),
                    "retraining_campaign": campaign_name,
                },
            )

        summary = {
            "campaign_name": campaign_name,
            "rounds": results,
            "best_round": best_result,
            "promotion_run_dir": str(promotion_run_dir) if promotion_run_dir else None,
        }
        report_root = config.data.report_path.parent
        write_json(report_root / "retraining_summary.json", summary)
        write_markdown_report(
            report_root / "retraining_summary.md",
            [
                "# Retraining Summary",
                "",
                *(
                    f"- round_{result['round_index']}: run={result['run_id']} | "
                    f"candidate={result['selected_candidate']} | estimator={result['selected_estimator']} | "
                    f"test_{selection_metric}={result['test_metrics'][selection_metric]}"
                    for result in results
                ),
                "",
                f"- best_round: {best_result['round_index']}",
                f"- best_run_id: {best_result['run_id']}",
                f"- best_candidate: {best_result['selected_candidate']}",
                f"- best_estimator: {best_result['selected_estimator']}",
                f"- promotion_run_dir: {summary['promotion_run_dir']}",
            ],
        )
        runner.mlflow.set_tags(
            {
                "best_round": best_result["round_index"],
                "best_candidate": best_result["selected_candidate"],
                "best_estimator": best_result["selected_estimator"],
            }
        )
        runner.mlflow.log_metrics(best_result["test_metrics"], prefix="best_round_test")
        runner.mlflow.log_metrics(best_result["validation_metrics"], prefix="best_round_validation")
        runner.mlflow.log_json_text(summary, "retraining_summary.json")
        runner.mlflow.log_artifact(report_root / "retraining_summary.json", artifact_path="campaign")
        runner.mlflow.log_artifact(report_root / "retraining_summary.md", artifact_path="campaign")
        return summary
