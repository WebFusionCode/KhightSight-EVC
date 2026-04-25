from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageEnhance, ImageFilter

YOLO_CONFIG_DIR = (Path.cwd() / ".ultralytics").resolve()
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
MPL_CONFIG_DIR = (Path.cwd() / ".matplotlib").resolve()
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

try:
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
except ModuleNotFoundError as error:
    if error.name == "ultralytics":
        print(
            "Missing dependency: ultralytics\n"
            "Install it with:\n"
            "  python3 -m pip install ultralytics\n"
            "If you use conda/miniforge, make sure the same python3 interpreter runs this script.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    raise


PHOTO_AUG_SUFFIX = "__photoaug"
DEFAULT_PROJECT = Path("runs/license_plate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8n license plate detector.")
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--model", type=str, default="runs/license_plate/quick_check_plus5_tight/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=str, default="auto")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default="yolov8n_plate_finetuned")
    parser.add_argument("--freeze", type=int, default=0, help="Freeze the first N model layers for faster fine-tuning.")
    parser.add_argument(
        "--cache",
        type=str,
        default="disk",
        choices=("false", "ram", "disk"),
        help="Dataset caching mode for Ultralytics training.",
    )
    parser.add_argument(
        "--photo-augment-ratio",
        type=float,
        default=0.40,
        help="Fraction of training images to duplicate with reproducible brightness/contrast/blur augmentation when albumentations is unavailable.",
    )
    parser.add_argument("--pred-conf", type=float, default=0.25)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate. Only used when explicitly set.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction. Only used when explicitly set.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay. Only used when explicitly set.")
    parser.add_argument("--box", type=float, default=7.5, help="Box loss gain.")
    parser.add_argument("--dfl", type=float, default=1.5, help="DFL loss gain.")
    parser.add_argument("--degrees", type=float, default=10.0, help="Rotation augmentation in degrees.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability.")
    parser.add_argument("--hsv-s", type=float, default=0.15, help="HSV saturation augmentation strength.")
    parser.add_argument("--hsv-v", type=float, default=0.25, help="HSV value augmentation strength.")
    parser.add_argument(
        "--skip-val-predictions",
        action="store_true",
        help="Skip saving annotated predictions on the validation set after training to reduce turnaround time.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_data_yaml(path: Path) -> dict:
    with path.open("r") as file:
        return yaml.safe_load(file)


def resolve_dataset_root(data_yaml: Path, data_config: dict) -> Path:
    path_value = Path(data_config["path"])
    if not path_value.is_absolute():
        path_value = (data_yaml.parent / path_value).resolve()
    return path_value


def accelerator_available() -> bool:
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    return torch.cuda.is_available() or mps_available


def as_ultralytics_batch(batch: str) -> int | str:
    if batch.lower() == "auto":
        mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if torch.cuda.is_available():
            return -1
        if mps_available:
            print("MPS detected, falling back from batch=auto to batch=8 for stable Apple Silicon training.")
            return 8
        print("No CUDA or MPS device detected, falling back from batch=auto to batch=8 for CPU training.")
        return 8
        return -1
    return int(batch)


def as_ultralytics_device(device: str) -> str | None:
    if device.lower() == "auto":
        mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if mps_available:
            print("Using Apple Silicon acceleration with device=mps.")
            return "mps"
        if torch.cuda.is_available():
            print("Using CUDA acceleration with device=0.")
            return "0"
        print("No hardware accelerator detected, using device=cpu.")
        return "cpu"
    return device


def as_ultralytics_cache(cache: str) -> bool | str:
    if cache == "false":
        return False
    return cache


def cleanup_previous_photo_augments(images_dir: Path, labels_dir: Path) -> None:
    for image_path in images_dir.glob(f"*{PHOTO_AUG_SUFFIX}.*"):
        image_path.unlink()
    for label_path in labels_dir.glob(f"*{PHOTO_AUG_SUFFIX}.txt"):
        label_path.unlink()


def collect_train_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and not path.stem.endswith(PHOTO_AUG_SUFFIX)
    )


def apply_photo_augment(image: Image.Image, rng: random.Random) -> Image.Image:
    brightness_factor = rng.uniform(0.70, 1.30)
    contrast_factor = rng.uniform(0.75, 1.30)
    blur_radius = rng.uniform(0.5, 2.5)

    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return image


def create_photo_augmented_copies(dataset_root: Path, ratio: float, seed: int) -> int:
    if ratio <= 0:
        return 0

    train_images_dir = dataset_root / "images" / "train"
    train_labels_dir = dataset_root / "labels" / "train"
    cleanup_previous_photo_augments(train_images_dir, train_labels_dir)

    base_images = collect_train_images(train_images_dir)
    if not base_images:
        raise FileNotFoundError(f"No training images found in {train_images_dir}")

    rng = random.Random(seed)
    sample_count = max(1, int(len(base_images) * ratio))
    selected_images = rng.sample(base_images, sample_count)

    for image_path in selected_images:
        label_path = train_labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing YOLO label for {image_path.name}")

        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
            augmented = apply_photo_augment(image, rng)
            augmented_path = train_images_dir / f"{image_path.stem}{PHOTO_AUG_SUFFIX}{image_path.suffix.lower()}"
            augmented.save(augmented_path, quality=95)

        copied_label_path = train_labels_dir / f"{image_path.stem}{PHOTO_AUG_SUFFIX}.txt"
        shutil.copy2(label_path, copied_label_path)

    return sample_count


def try_build_albumentations():
    try:
        import albumentations as A
    except ImportError:
        return None

    return [
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.5, 2.5), p=0.35),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.40),
    ]


def collect_val_images(dataset_root: Path) -> list[Path]:
    image_dir = dataset_root / "images" / "val"
    return sorted(path for path in image_dir.iterdir() if path.is_file())


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    SETTINGS.update({"wandb": False, "mlflow": False, "tensorboard": False, "clearml": False, "comet": False})

    data_config = read_data_yaml(args.data)
    dataset_root = resolve_dataset_root(args.data, data_config)
    project_dir = args.project.resolve()
    batch = as_ultralytics_batch(args.batch)
    device = as_ultralytics_device(args.device)
    cache_mode = as_ultralytics_cache(args.cache)
    val_images = collect_val_images(dataset_root)
    train_images_dir = dataset_root / "images" / "train"
    train_labels_dir = dataset_root / "labels" / "train"

    if not val_images:
        raise FileNotFoundError(
            f"No validation images found under {dataset_root / 'images' / 'val'}. Run convert.py first."
        )

    print("Training configuration:")
    config_to_print = {
        "model": args.model,
        "data": str(args.data),
        "dataset_root": str(dataset_root),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "patience": args.patience,
        "device": args.device,
        "seed": args.seed,
        "freeze": args.freeze,
        "cache": args.cache,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "box": args.box,
        "dfl": args.dfl,
        "degrees": args.degrees,
        "fliplr": args.fliplr,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "train_images": len(list((dataset_root / "images" / "train").iterdir())),
        "val_images": len(val_images),
        "project": str(project_dir),
    }
    print(json.dumps(config_to_print, indent=2))

    train_kwargs = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": batch,
        "patience": args.patience,
        "device": device,
        "freeze": args.freeze,
        "cache": cache_mode,
        "seed": args.seed,
        "deterministic": True,
        "project": str(project_dir),
        "name": args.name,
        "save": True,
        "plots": True,
        "workers": min(8, os.cpu_count() or 1),
        "fliplr": args.fliplr,
        "flipud": 0.0,
        "degrees": args.degrees,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "optimizer": args.optimizer,
        "box": args.box,
        "dfl": args.dfl,
        "verbose": True,
    }

    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = args.weight_decay

    augmentations = try_build_albumentations()
    extra_photo_augments = 0
    used_offline_photo_augments = False
    if augmentations is not None:
        train_kwargs["augmentations"] = augmentations
        print("Using Albumentations for brightness/contrast and gaussian blur.")
    else:
        used_offline_photo_augments = True
        extra_photo_augments = create_photo_augmented_copies(dataset_root, args.photo_augment_ratio, args.seed)
        print(
            "Albumentations is not installed, so brightness/contrast and gaussian blur were applied "
            f"offline to {extra_photo_augments} training images."
        )
    try:
        model = YOLO(args.model)
        model.train(**train_kwargs)

        train_run_dir = Path(model.trainer.save_dir)
        best_weights_path = Path(model.trainer.best)
        print(f"Best weights saved to {best_weights_path}")

        best_model = YOLO(str(best_weights_path))
        metrics = best_model.val(
            data=str(args.data),
            imgsz=args.imgsz,
            batch=batch,
            device=device,
            split="val",
            plots=True,
            project=str(project_dir),
            name=f"{args.name}_val",
        )

        val_prediction_dir_name = f"{args.name}_val_predictions"
        if not args.skip_val_predictions:
            best_model.predict(
                source=str(dataset_root / "images" / "val"),
                imgsz=args.imgsz,
                conf=args.pred_conf,
                device=device,
                save=True,
                project=str(project_dir),
                name=val_prediction_dir_name,
                verbose=False,
            )

        metrics_payload = {
            "best_weights": str(best_weights_path),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "train_run_dir": str(train_run_dir),
            "val_run_dir": str(project_dir / f"{args.name}_val"),
            "val_predictions_dir": str(project_dir / val_prediction_dir_name) if not args.skip_val_predictions else None,
            "offline_photo_augmented_images": extra_photo_augments,
        }
        metrics_json_path = train_run_dir / "metrics_summary.json"
        save_json(metrics_json_path, metrics_payload)

        print(f"mAP@0.5: {metrics_payload['map50']:.4f}")
        print(f"mAP@0.5:0.95: {metrics_payload['map50_95']:.4f}")
        print(f"Metrics summary saved to {metrics_json_path}")
    finally:
        if used_offline_photo_augments:
            cleanup_previous_photo_augments(train_images_dir, train_labels_dir)


if __name__ == "__main__":
    main()
