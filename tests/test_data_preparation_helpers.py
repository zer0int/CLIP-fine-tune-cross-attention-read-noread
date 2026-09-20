from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import build_imagenet_derivatives as imagenet_builder
import prepare_objectnet_mvt as objectnet_mvt


def test_imagenet_digital_materialization_skips_historical_controls(tmp_path: Path) -> None:
    imagenet_root = tmp_path / "imagenet"
    source_dir = imagenet_root / "train" / "n00000001"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (320, 240), (100, 150, 200)).save(source_dir / "sample.JPEG", "JPEG")

    derivative_root = tmp_path / "derivative"
    clean = {
        "uid": "g__clean__s0",
        "group_id": "g",
        "source_image_path": "train/n00000001/sample.JPEG",
        "image_relpath": "images/train/g__clean__s0.jpg",
        "mask_relpath": None,
        "variant": "clean",
        "relation": "none",
    }

    def text_row(uid: str, variant: str, relation: str, text: str, bbox: list[int]) -> dict:
        return {
            "uid": uid,
            "group_id": "g",
            "source_image_path": "train/n00000001/sample.JPEG",
            "image_relpath": f"images/train/{uid}.jpg",
            "mask_relpath": f"masks/train/{uid}.png",
            "variant": variant,
            "relation": relation,
            "overlay_text": text,
            "metadata": {
                "text": {
                    "bbox_xyxy": bbox,
                    "angle_deg": 1.5,
                    "font_size": 24,
                    "mirrored": False,
                    "style_box": False,
                }
            },
        }

    rows = [
        clean,
        text_row("g__digital_support__s0", "digital_support", "supportive", "thing", [20, 20, 100, 55]),
        text_row("g__digital_adversarial__s0", "digital_adversarial", "adversarial", "wrong", [80, 130, 160, 165]),
        text_row("g__pseudo_or_mirror__s0", "pseudo_or_mirror", "control", "x", [10, 100, 50, 130]),
    ]

    counts = imagenet_builder.build_digital_split(
        rows=rows,
        split="train",
        root=derivative_root,
        imagenet_root=imagenet_root,
        font_path=None,
        overwrite=False,
        comparator=imagenet_builder.ReferenceComparator(None, None, 0),
    )

    assert (derivative_root / "images/train/g__clean__s0.jpg").is_file()
    assert (derivative_root / "images/train/g__digital_support__s0.jpg").is_file()
    assert (derivative_root / "masks/train/g__digital_support__s0.png").is_file()
    assert not (derivative_root / "images/train/g__pseudo_or_mirror__s0.jpg").exists()
    assert counts["skipped_pseudo_or_mirror"] == 1
    imagenet_builder.verify_outputs(derivative_root, {"train": rows}, handwriting=False)


def test_objectnet_archive_mapping_and_config_patch_are_evaluation_only(tmp_path: Path) -> None:
    archive = tmp_path / "mvt.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("nested/a.png", b"a")
        zf.writestr("somewhere/else/b.jpg", b"b")
        zf.writestr("irrelevant.txt", b"ignore")

    with zipfile.ZipFile(archive, "r") as zf:
        mapping = objectnet_mvt.archive_member_map(zf, {"a.png", "b.jpg"})
    assert set(mapping) == {"a.png", "b.jpg"}

    config_path = tmp_path / "training_config.local.json"
    config = {
        "paths": {"mvt_csv": None, "mvt_image_root": None},
        "shared_args": {
            "final_anytext": {
                "benchmark_include_mvt": False,
                "select_by_benchmark_score": True,
            }
        },
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    csv_path = tmp_path / "human_responses_dedup.csv"
    csv_path.write_text("image,label\na.png,a\n", encoding="utf-8")
    image_root = tmp_path / "images"
    image_root.mkdir()

    objectnet_mvt.patch_config(config_path, csv_path, image_root)
    patched = json.loads(config_path.read_text(encoding="utf-8"))
    assert patched["shared_args"]["final_anytext"]["benchmark_include_mvt"] is True
    assert patched["shared_args"]["final_anytext"]["select_by_benchmark_score"] is False
    assert patched["paths"]["mvt_csv"] == str(csv_path.resolve())
    assert patched["paths"]["mvt_image_root"] == str(image_root.resolve())


def test_objectnet_nested_cropped_images_zip_is_resolved(tmp_path: Path) -> None:
    from io import BytesIO

    def image_bytes(fmt: str, color: tuple[int, int, int]) -> bytes:
        buf = BytesIO()
        Image.new("RGB", (8, 8), color).save(buf, format=fmt)
        return buf.getvalue()

    inner_path = tmp_path / "cropped_images.zip"
    with zipfile.ZipFile(inner_path, "w", compression=zipfile.ZIP_STORED) as inner:
        inner.writestr("cropped_images/class_a/a.png", image_bytes("PNG", (255, 0, 0)))
        inner.writestr("cropped_images/class_b/b.jpg", image_bytes("JPEG", (0, 255, 0)))

    outer_path = tmp_path / "flash_data_release_2023.zip"
    with zipfile.ZipFile(outer_path, "w", compression=zipfile.ZIP_STORED) as outer:
        outer.writestr("data_release_2023/README.txt", b"release")
        outer.write(inner_path, "data_release_2023/cropped_images.zip")

    image_root = tmp_path / "installed" / "images"
    result = objectnet_mvt.extract_expected(
        outer_path,
        {"a.png": "a", "b.jpg": "b"},
        image_root,
        reset=False,
    )

    assert (image_root / "a.png").is_file()
    assert (image_root / "b.jpg").is_file()
    assert result["payload_member"] == "data_release_2023/cropped_images.zip"
    assert result["payload_zip_sha256"] != ""
