from __future__ import annotations

import torch

from benchmarks.typo import PairSample, evaluate_logits


def _sample(subset: str) -> PairSample:
    return PairSample(
        image=None,
        correct_label="banana",
        distractor_label="apple",
        sample_id="sample",
        subset=subset,
    )


def test_read_task_targets_attack_text_on_attacked_images() -> None:
    summary, records = evaluate_logits(
        [_sample("SCAM")],
        ["banana", "apple"],
        torch.tensor([[0.1, 0.9, 0.2]]),
        mode="read",
        mode_label="<text>+<null>",
    )
    assert summary["binary_accuracy"] == 1.0
    assert records[0]["goal"] == "apple"


def test_read_task_targets_null_on_unattacked_images() -> None:
    summary, records = evaluate_logits(
        [_sample("NoSCAM")],
        ["banana", "apple"],
        torch.tensor([[0.2, 0.1, 0.9]]),
        mode="read",
        mode_label="<text>+<null>",
    )
    assert summary["binary_accuracy"] == 1.0
    assert summary["no_text_detected_count"] == 1
    assert records[0]["goal"] == "NO_TEXT_DETECTED"


def test_any_task_remains_object_recognition() -> None:
    summary, records = evaluate_logits(
        [_sample("SCAM")],
        ["banana", "apple"],
        torch.tensor([[0.9, 0.1]]),
        mode="any",
        mode_label="<any>",
    )
    assert summary["binary_accuracy"] == 1.0
    assert records[0]["goal"] == "banana"
