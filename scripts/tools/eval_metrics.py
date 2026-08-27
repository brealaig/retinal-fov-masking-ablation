import argparse, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    observed_matrix = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(np.float64)
    sample_count = observed_matrix.sum()
    if sample_count == 0: return 0.0
    weight_matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
    for true_class in range(n_classes):
        for predicted_class in range(n_classes):
            weight_matrix[true_class, predicted_class] = ((true_class - predicted_class) ** 2) / ((n_classes - 1) ** 2)
    actual_histogram = observed_matrix.sum(axis=1)
    predicted_histogram = observed_matrix.sum(axis=0)
    expected_matrix = np.outer(actual_histogram, predicted_histogram) / sample_count
    weighted_observed = (weight_matrix * observed_matrix).sum()
    weighted_expected = (weight_matrix * expected_matrix).sum()
    return 1.0 - weighted_observed / weighted_expected if weighted_expected > 0 else 0.0


def expected_calibration_error(probs, y_true, n_bins=15):
    confidence = probs.max(axis=1)
    predicted_labels = probs.argmax(axis=1)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration_error = 0.0
    sample_count = len(y_true)
    for bin_index in range(n_bins):
        lower_bound, upper_bound = bin_edges[bin_index], bin_edges[bin_index + 1]
        bin_mask = (confidence > lower_bound) & (confidence <= upper_bound)
        if not np.any(bin_mask):
            continue
        bin_accuracy = (predicted_labels[bin_mask] == y_true[bin_mask]).mean()
        bin_confidence = confidence[bin_mask].mean()
        calibration_error += (bin_mask.sum() / sample_count) * abs(bin_accuracy - bin_confidence)
    return float(calibration_error)


def brier_score_multiclass(probs, y_true, n_classes=5):
    one_hot_labels = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - one_hot_labels) ** 2, axis=1)))


def per_class_metrics(y_true, y_pred, n_classes=5):
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    sensitivity_by_class, ppv_by_class = {}, {}
    for class_index in range(n_classes):
        true_positives = confusion[class_index, class_index]
        false_negatives = confusion[class_index, :].sum() - true_positives
        false_positives = confusion[:, class_index].sum() - true_positives
        sensitivity_by_class[class_index] = float(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else float('nan')
        ppv_by_class[class_index] = float(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else float('nan')
    return confusion, sensitivity_by_class, ppv_by_class


def referable_scores(probs, y_true, thr_class=2):
    binary_labels = (y_true >= thr_class).astype(int)
    referable_probability = probs[:, thr_class:].sum(axis=1)
    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(binary_labels, referable_probability)
    precision, recall, pr_thresholds = precision_recall_curve(binary_labels, referable_probability)
    return dict(
        roc_auc=float(auc(false_positive_rate, true_positive_rate)),
        pr_auc=float(auc(recall, precision)),
        fpr=false_positive_rate, tpr=true_positive_rate, prec=precision, rec=recall, thr_pr=pr_thresholds, thr_roc=roc_thresholds,
        y_bin=binary_labels, score=referable_probability
    )


def ppv_at_recall(prec, rec, thr, recall_target=0.90):
    valid_indices = np.where(rec >= recall_target)[0]
    if len(valid_indices) == 0:
        return 0.0, 1.0

    best_index = valid_indices[np.argmax(prec[valid_indices])]

    threshold_index = max(best_index - 1, 0)
    threshold_index = min(threshold_index, len(thr) - 1) if len(thr) > 0 else 0

    selected_threshold = float(thr[threshold_index]) if len(thr) else 0.0
    return float(prec[best_index]), selected_threshold


def safe_parse_probs(s):
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=np.float64)
    text = str(s).strip()
    text = re.sub(r"'", '"', text)
    try:
        probabilities = np.array(json.loads(text), dtype=np.float64)
    except Exception:
        text = text.replace(" ", "")
        probabilities = np.array(json.loads(text), dtype=np.float64)
    return probabilities


def load_df(csv_path):
    dataframe = pd.read_csv(csv_path)
    if "true_label" not in dataframe.columns and "y_true" in dataframe.columns:
        dataframe["true_label"] = dataframe["y_true"]
    if "pred_label" not in dataframe.columns and "y_pred" in dataframe.columns:
        dataframe["pred_label"] = dataframe["y_pred"]
    if "y_true" in dataframe.columns:
        dataframe["true_label"] = dataframe["true_label"].fillna(dataframe["y_true"])
    if "y_pred" in dataframe.columns:
        dataframe["pred_label"] = dataframe["pred_label"].fillna(dataframe["y_pred"])
    required_columns = {"true_label", "pred_label", "probs_json"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns} (minimum: true_label/pred_label/probs_json).")
    return dataframe


def stack_probs(col, n_classes=5):
    parsed_probabilities = [safe_parse_probs(value) for value in col]
    probability_matrix = np.vstack(parsed_probabilities)
    if probability_matrix.ndim != 2 or probability_matrix.shape[1] != n_classes:
        raise ValueError(f"probs_json is not (N,{n_classes}). Shape={probability_matrix.shape}")
    row_sums = probability_matrix.sum(axis=1, keepdims=True)
    invalid_sum_mask = ~np.isclose(row_sums, 1.0, atol=1e-3)
    if invalid_sum_mask.any():
        probability_matrix = probability_matrix / np.clip(row_sums, 1e-9, None)
    return probability_matrix


def reliability_points(probs, y_true, n_bins=15):
    confidence = probs.max(axis=1)
    predicted_labels = probs.argmax(axis=1)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mean_confidences, mean_accuracies = [], []
    for bin_index in range(n_bins):
        lower_bound, upper_bound = bin_edges[bin_index], bin_edges[bin_index + 1]
        bin_mask = (confidence > lower_bound) & (confidence <= upper_bound)
        if not np.any(bin_mask): continue
        mean_confidences.append(confidence[bin_mask].mean())
        mean_accuracies.append((predicted_labels[bin_mask] == y_true[bin_mask]).mean())
    return np.array(mean_confidences), np.array(mean_accuracies)


def pretty_cm_csv(cm, out_csv):
    pd.DataFrame(cm, index=[f"T{class_index}" for class_index in range(cm.shape[0])],
                    columns=[f"P{class_index}" for class_index in range(cm.shape[1])]).to_csv(out_csv, index=True)


def plot_all(save_dir, cm, curves, title_tag, probs=None, y_true=None, n_bins=15):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        output_directory = Path(save_dir); output_directory.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Confusion Matrix — {title_tag}")
        plt.xlabel("Pred"); plt.ylabel("True"); plt.colorbar()
        plt.xticks(range(5), range(5)); plt.yticks(range(5), range(5))
        plt.tight_layout()
        plt.savefig(output_directory / f"cm_{title_tag}.png", dpi=160)
        plt.close()

        plt.figure()
        plt.plot(curves["fpr"], curves["tpr"])
        plt.plot([0,1],[0,1],'--')
        plt.title(f"ROC ≥ referable — AUC={curves['roc_auc']:.3f} — {title_tag}")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.tight_layout()
        plt.savefig(output_directory / f"roc_{title_tag}.png", dpi=160)
        plt.close()

        plt.figure()
        plt.plot(curves["rec"], curves["prec"])
        plt.title(f"PR ≥ referable — AUC={curves['pr_auc']:.3f} — {title_tag}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.tight_layout()
        plt.savefig(output_directory / f"pr_{title_tag}.png", dpi=160)
        plt.close()

        if probs is not None and y_true is not None:
            mean_confidences, mean_accuracies = reliability_points(probs, y_true, n_bins=n_bins)
            plt.figure()
            plt.plot([0,1],[0,1],'--')
            plt.plot(mean_confidences, mean_accuracies, marker='o')
            plt.title(f"Reliability (ECE bins={n_bins}) — {title_tag}")
            plt.xlabel("Confidence"); plt.ylabel("Accuracy")
            plt.tight_layout()
            plt.savefig(output_directory / f"reliability_{title_tag}.png", dpi=160)
            plt.close()
    except Exception as error:
        print(f"[WARN] Could not generate plots: {error}")


def choose_cases(df, probs, y_true, y_pred, k=12):
    selected_cases = {}
    working_dataframe = df
    if "max_prob" not in working_dataframe.columns:
        working_dataframe = working_dataframe.copy()
        working_dataframe["max_prob"] = probs.max(axis=1)

    false_negative_mask = (y_true >= 3) & (y_pred < y_true)
    false_negative_dataframe = working_dataframe[false_negative_mask].copy()
    false_negative_dataframe["gap"] = (y_true[false_negative_mask] - y_pred[false_negative_mask])
    selected_cases["FN_3_4_top"] = false_negative_dataframe.sort_values(["gap", "max_prob"], ascending=[False, True]).head(k).to_dict(orient="records")

    false_positive_mask = (y_true == 2) & (y_pred >= 3)
    false_positive_dataframe = working_dataframe[false_positive_mask].copy()
    false_positive_dataframe["pred"] = y_pred[false_positive_mask]
    selected_cases["FP_2_to_34_top"] = false_positive_dataframe.sort_values(["pred", "max_prob"], ascending=[False, False]).head(k).to_dict(orient="records")

    ambiguous_mask = working_dataframe["max_prob"].between(0.35, 0.55)
    selected_cases["ambiguous_low_conf"] = working_dataframe[ambiguous_mask].sample(min(k, ambiguous_mask.sum()), random_state=42).to_dict(orient="records")
    return selected_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--referable_def", type=int, default=2, choices=[2,3])
    parser.add_argument("--ppv_target", type=float, default=0.90)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--summary_csv_append", default=None)
    parser.add_argument("--export_cm_csv", default=None)
    parser.add_argument("--threshold_sweep_csv", default=None)
    parser.add_argument("--save_plots", default=None)
    parser.add_argument("--export_cases_json", default=None)
    parser.add_argument("--cases_k", type=int, default=12)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    dataframe = load_df(csv_path)
    y_true = dataframe['true_label'].astype(int).to_numpy()
    y_pred = dataframe['pred_label'].astype(int).to_numpy()
    probabilities = stack_probs(dataframe['probs_json'])

    cm, sensitivity, ppv = per_class_metrics(y_true, y_pred, n_classes=5)
    qwk = quadratic_weighted_kappa(y_true, y_pred, n_classes=5)
    ece = expected_calibration_error(probabilities, y_true, n_bins=args.bins)
    brier = brier_score_multiclass(probabilities, y_true, n_classes=5)

    curves = referable_scores(probabilities, y_true, thr_class=args.referable_def)
    ppv_at_target, threshold_at_target = ppv_at_recall(curves["prec"], curves["rec"], curves["thr_pr"], recall_target=args.ppv_target)

    if args.export_cm_csv:
        Path(args.export_cm_csv).parent.mkdir(parents=True, exist_ok=True)
        pretty_cm_csv(cm, args.export_cm_csv)

    if args.threshold_sweep_csv:
        precision, recall, thresholds = curves["prec"], curves["rec"], curves["thr_pr"]
        sweep_rows = []
        for index in range(len(thresholds)):
            precision_value, recall_value, threshold_value = float(precision[index]), float(recall[index]), float(thresholds[index])
            f1_score = (2 * precision_value * recall_value / (precision_value + recall_value)) if (precision_value + recall_value) > 0 else 0.0
            sweep_rows.append(dict(threshold=threshold_value, precision=precision_value, recall=recall_value, f1=f1_score))
        pd.DataFrame(sweep_rows).to_csv(args.threshold_sweep_csv, index=False)

    if args.save_plots:
        title_tag = csv_path.stem
        plot_all(args.save_plots, cm, curves, title_tag, probs=probabilities, y_true=y_true, n_bins=args.bins)

    cases = None
    if args.export_cases_json:
        candidate_columns = ["filepath", "true_label", "pred_label", "max_prob"]
        available_columns = [column for column in candidate_columns if column in dataframe.columns]
        cases_dataframe = dataframe[available_columns].copy()
        if "max_prob" not in cases_dataframe.columns:
            cases_dataframe["max_prob"] = probabilities.max(axis=1)
        cases = choose_cases(cases_dataframe, probabilities, y_true, y_pred, k=args.cases_k)
        cases_output_path = Path(args.export_cases_json)
        cases_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cases_output_path, "w", encoding="utf-8") as output_file:
            json.dump(cases, output_file, ensure_ascii=False, indent=2)

    output = {
        "path": str(csv_path),
        "referable_def": args.referable_def,
        "ppv_target": args.ppv_target,
        "n_samples": int(len(y_true)),
        "qwk": qwk,
        "ece": ece,
        "brier": brier,
        "confusion_matrix": cm.tolist(),
        "sensitivity_by_class": {str(class_index): value for class_index, value in sensitivity.items()},
        "ppv_by_class": {str(class_index): value for class_index, value in ppv.items()},
        "roc_auc": curves["roc_auc"],
        "pr_auc": curves["pr_auc"],
        "ppv_at_recall_target": ppv_at_target,
        "threshold_at_recall_target": threshold_at_target
    }
    if args.output_json:
        output_path = Path(args.output_json); output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(output, output_file, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.summary_csv_append:
        summary_row = dict(
            csv=str(csv_path),
            n=output["n_samples"],
            referable_def=args.referable_def,
            qwk=output["qwk"],
            ece=output["ece"],
            brier=output["brier"],
            roc_auc=output["roc_auc"],
            pr_auc=output["pr_auc"],
            ppv_at_recall=output["ppv_at_recall_target"],
            thr_at_recall=output["threshold_at_recall_target"],
            sens1=output["sensitivity_by_class"].get("1", float("nan")),
            sens3=output["sensitivity_by_class"].get("3", float("nan")),
            sens4=output["sensitivity_by_class"].get("4", float("nan")),
            ppv1=output["ppv_by_class"].get("1", float("nan")),
            ppv3=output["ppv_by_class"].get("3", float("nan")),
            ppv4=output["ppv_by_class"].get("4", float("nan"))
        )
        summary_path = Path(args.summary_csv_append)
        if not summary_path.exists():
            pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
        else:
            pd.DataFrame([summary_row]).to_csv(summary_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
