import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def process_file(
    filepath: str | Path, true_label: str, force_extract: bool = False
) -> list[tuple[str, str]]:
    """
    Process a JSONL file and extract predictions.

    Args:
        filepath: Path to the JSONL file
        true_label: The true label for all samples in this file
            ("Cognitive Normal" or "Cognitive Decline")
        force_extract: If True, re-extract labels even if they already exist

    Returns:
        List of tuples (true_label, predicted_label)
    """
    filepath = Path(filepath)
    data = [
        json.loads(line) for line in filepath.read_text().splitlines() if line.strip()
    ]

    results = []
    modified = False

    for entry in tqdm(data, desc=f"Processing {filepath.name}"):
        # Check if extracted label already exists
        if not force_extract and "extracted_label" in entry:
            predicted_label = entry["extracted_label"]
            logger.info(
                "Using existing label for %s: %s",
                entry.get("filename", "Unknown"),
                predicted_label,
            )
        else:
            output = entry["output"]
            # Determine predicted label
            if (
                '"category": "Cognitive Decline' in output
                or '"Cognitive Decline' in output
                or "{'category': 'Cognitive Decline" in output
            ):
                predicted_label = "Cognitive Decline"
            elif (
                '"category": "Cognitive Normal' in output
                or '"Cognitive Normal' in output
                or "{'category': 'Cognitive Normal" in output
            ):
                predicted_label = "Cognitive Normal"
            else:
                # Manual classification needed
                logger.info(f"\n{'=' * 50}")
                logger.info(f"TRUE LABEL: {true_label}")
                logger.info(f"Filename: {entry.get('filename', 'Unknown')}")
                logger.info(f"Output: {output}")
                extracted_response = input("Enter 1 if correct, 0 if incorrect: ")
                if extracted_response == "1":
                    predicted_label = true_label
                else:
                    predicted_label = (
                        "Cognitive Decline"
                        if true_label == "Cognitive Normal"
                        else "Cognitive Normal"
                    )
                modified = True

            # Add extracted label to entry
            entry["extracted_label"] = predicted_label
            entry["true_label"] = true_label
            modified = True

        results.append((true_label, predicted_label))
        if predicted_label == true_label:
            logger.info(f"{entry.get('filename', 'Unknown')}")

    # Write back to file with extracted labels
    if modified:
        filepath.write_text("\n".join(json.dumps(e) for e in data) + "\n")
        logger.info(f"\nUpdated {filepath} with extracted labels")

    return results


def calculate_metrics(all_results: list[tuple[str, str]]) -> dict[str, float | int]:
    """
    Calculate classification metrics from prediction results.

    Args:
        all_results: List of tuples (true_label, predicted_label)

    Returns:
        Dictionary with metrics
    """
    # Initialize confusion matrix components
    tp = 0  # True Positives (Cognitive Decline predicted as Cognitive Decline)
    tn = 0  # True Negatives (Cognitive Normal predicted as Cognitive Normal)
    fp = 0  # False Positives (Cognitive Normal predicted as Cognitive Decline)
    fn = 0  # False Negatives (Cognitive Decline predicted as Cognitive Normal)

    for true_label, predicted_label in all_results:
        if true_label == "Cognitive Decline" and predicted_label == "Cognitive Decline":
            tp += 1
        elif true_label == "Cognitive Normal" and predicted_label == "Cognitive Normal":
            tn += 1
        elif (
            true_label == "Cognitive Normal" and predicted_label == "Cognitive Decline"
        ):
            fp += 1
        elif (
            true_label == "Cognitive Decline" and predicted_label == "Cognitive Normal"
        ):
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate cognitive decline classification results from JSONL output files.",  # noqa: E501
    )
    p.add_argument(
        "--cn_file",
        required=True,
        help="Path to JSONL file with Cognitive Normal predictions",
    )
    p.add_argument(
        "--cd_file",
        required=True,
        help="Path to JSONL file with Cognitive Decline predictions",
    )
    p.add_argument(
        "--force_extract",
        action="store_true",
        help="Re-extract labels even if already present in the file",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    logger.info("Processing Cognitive Normal samples...")
    control_results = process_file(
        args.cn_file, "Cognitive Normal", force_extract=args.force_extract
    )

    logger.info("\n" + "=" * 50)
    logger.info("Processing Cognitive Decline samples...")
    mdd_results = process_file(
        args.cd_file, "Cognitive Decline", force_extract=args.force_extract
    )

    all_results = control_results + mdd_results
    metrics = calculate_metrics(all_results)

    logger.info("\n" + "=" * 50)
    logger.info("CLASSIFICATION METRICS")
    logger.info("=" * 50)
    correct = metrics["tp"] + metrics["tn"]
    logger.info(
        "Accuracy:    %.4f (%d/%d)", metrics["accuracy"], correct, metrics["total"]
    )
    logger.info(f"Precision:   {metrics['precision']:.4f}")
    logger.info(f"Recall:      {metrics['recall']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")
    logger.info(f"F1 Score:    {metrics['f1_score']:.4f}")
    logger.info("\nConfusion Matrix:")
    logger.info(f"  True Positives (Cognitive Decline): {metrics['tp']}")
    logger.info(f"  True Negatives (Cognitive Normal):  {metrics['tn']}")
    logger.info(f"  False Positives:                    {metrics['fp']}")
    logger.info(f"  False Negatives:                    {metrics['fn']}")
