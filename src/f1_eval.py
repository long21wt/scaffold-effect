import json
import os
from tqdm import tqdm


def process_file(filepath, true_label, force_extract=False):
    """
    Process a JSONL file and extract predictions.

    Args:
        filepath: Path to the JSONL file
        true_label: The true label for all samples in this file ("Control" or "MDD")
        force_extract: If True, re-extract labels even if they already exist

    Returns:
        List of tuples (true_label, predicted_label)
    """
    with open(filepath, "r") as f:
        file = f.readlines()
    data = [json.loads(line) for line in file]

    results = []
    modified = False

    for entry in tqdm(data, desc=f"Processing {os.path.basename(filepath)}"):
        # Check if extracted label already exists
        if not force_extract and "extracted_label" in entry:
            predicted_label = entry["extracted_label"]
            print(
                f"Using existing label for {entry.get('filename', 'Unknown')}: {predicted_label}"
            )
        else:
            output = entry["output"]
            # Determine predicted label
            if (
                '"category": "Major Depressive Disorder' in output
                or '"Major Depressive Disorder' in output
                or "{'category': 'Major Depressive Disorder" in output
            ):
                predicted_label = "MDD"
            elif (
                '"category": "Control' in output
                or '"Control (no disorder detected)' in output
                or "{'category': 'Control" in output
            ):
                predicted_label = "Control"
            else:
                # Manual classification needed
                print(f"\n{'=' * 50}")
                print(f"TRUE LABEL: {true_label}")
                print(f"Filename: {entry.get('filename', 'Unknown')}")
                print(f"Output: {output}")
                extracted_response = input("Enter 1 if correct, 0 if incorrect: ")
                if extracted_response == "1":
                    predicted_label = true_label
                else:
                    predicted_label = "MDD" if true_label == "Control" else "Control"
                modified = True

            # Add extracted label to entry
            entry["extracted_label"] = predicted_label
            entry["true_label"] = true_label
            modified = True

        results.append((true_label, predicted_label))
        if predicted_label == true_label:
            print(f"✓ {entry.get('filename', 'Unknown')}")

    # Write back to file with extracted labels
    if modified:
        with open(filepath, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"\n Updated {filepath} with extracted labels")

    return results


def calculate_metrics(all_results):
    """
    Calculate classification metrics from prediction results.

    Args:
        all_results: List of tuples (true_label, predicted_label)

    Returns:
        Dictionary with metrics
    """
    # Initialize confusion matrix components
    tp = 0  # True Positives (MDD predicted as MDD)
    tn = 0  # True Negatives (Control predicted as Control)
    fp = 0  # False Positives (Control predicted as MDD)
    fn = 0  # False Negatives (MDD predicted as Control)

    for true_label, predicted_label in all_results:
        if true_label == "MDD" and predicted_label == "MDD":
            tp += 1
        elif true_label == "Control" and predicted_label == "Control":
            tn += 1
        elif true_label == "Control" and predicted_label == "MDD":
            fp += 1
        elif true_label == "MDD" and predicted_label == "Control":
            fn += 1

    # Calculate metrics
    total = len(all_results)
    accuracy = (tp + tn) / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Evaluate MDD classification results from JSONL output files."
    )
    p.add_argument(
        "--control_file",
        required=True,
        help="Path to JSONL file with Control (healthy) predictions",
    )
    p.add_argument(
        "--mdd_file", required=True, help="Path to JSONL file with MDD predictions"
    )
    p.add_argument(
        "--force_extract",
        action="store_true",
        help="Re-extract labels even if already present in the file",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Processing Control samples...")
    control_results = process_file(
        args.control_file, "Control", force_extract=args.force_extract
    )

    print("\n" + "=" * 50)
    print("Processing MDD samples...")
    mdd_results = process_file(args.mdd_file, "MDD", force_extract=args.force_extract)

    all_results = control_results + mdd_results
    metrics = calculate_metrics(all_results)

    print("\n" + "=" * 50)
    print("CLASSIFICATION METRICS")
    print("=" * 50)
    print(
        f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['tp'] + metrics['tn']}/{metrics['total']})"
    )
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Positives (MDD):     {metrics['tp']}")
    print(f"  True Negatives (Control): {metrics['tn']}")
    print(f"  False Positives:          {metrics['fp']}")
    print(f"  False Negatives:          {metrics['fn']}")
