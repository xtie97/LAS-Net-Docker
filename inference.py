import os
import argparse
from pathlib import Path
from typing import List
from utils import process_pt_ct, generate_RTstruct
from segmenter import inference

DEFAULT_DICOM_INPUT_PATH = Path("/input_dicom")
DEAULT_INPUT_PATH = Path("/input")
DEFAULT_OUTPUT_PATH = Path("/output")
DEFAULT_MODEL_ROOT = Path("/opt/app/resources")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LASNet lymphoma PET/CT segmentation Docker entry point."
    )
    parser.add_argument("--input-PET1", type=Path, default=DEFAULT_DICOM_INPUT_PATH)
    parser.add_argument("--input-CT1", type=Path, default=DEFAULT_DICOM_INPUT_PATH)
    parser.add_argument("--input-PET2", type=Path, default=DEFAULT_DICOM_INPUT_PATH)
    parser.add_argument("--input-CT2", type=Path, default=DEFAULT_DICOM_INPUT_PATH)
    parser.add_argument("--input-dir", type=Path, default=DEAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--model-type",
        choices=("rigid", "deform"),
        default="rigid",
        help="Checkpoint family to use: rigid or deformable.",
    )
    parser.add_argument(
        "--use-dicom-input",
        action="store_true",
        help="Whether to use DICOM input. If set, --input-PET1, --input-CT1, --input-PET2, and --input-CT2 will be needed",
    )
    parser.add_argument(
        "--generate-interim-lesion-mask",
        action="store_true",
        help="Generate the interim/residual lesion mask. If omitted, generate the baseline lymphoma mask.",
    )
    return parser.parse_args()


def checkpoint_paths(
    model_root: Path, model_type: str, num_folds: int = 5
) -> List[Path]:
    return [
        model_root / f"lasnet_{model_type}_f{i}.pt" for i in range(1, num_folds + 1)
    ]


def validate_inputs(
    input_dir: Path,
    generate_interim: bool,
    use_dicom: bool,
    pt1_path: Path = None,
    ct1_path: Path = None,
    pt2_path: Path = None,
    ct2_path: Path = None,
) -> tuple[Path, Path, Path, Path]:

    pet1 = input_dir / "PET1.nii.gz"
    ct1 = input_dir / "CT1.nii.gz"
    pet2 = input_dir / "PET2.nii.gz"
    ct2 = input_dir / "CT2.nii.gz"

    if use_dicom:
        if not all([pt1_path, ct1_path]):
            raise ValueError("--use-dicom-input requires --input-PET1 and --input-CT1.")
        if generate_interim and not all([pt2_path, ct2_path]):
            raise ValueError(
                "--use-dicom-input with --generate-interim-lesion-mask requires "
                "--input-PET2 and --input-CT2."
            )
        if generate_interim:
            process_pt_ct(pt1_path, ct1_path, input_dir, pt2_path, ct2_path)
        else:
            process_pt_ct(pt1_path, ct1_path, input_dir)

    if not pet1.exists() or not ct1.exists():
        raise FileNotFoundError(
            f"Baseline PET/CT not found in {input_dir}. Expected /input/PET1 and /input/CT1."
        )

    if generate_interim:
        if not pet2.exists() or not ct2.exists():
            raise FileNotFoundError(
                "--generate-interim-lesion-mask requires interim PET/CT. "
                f"Expected {pet2} and {ct2}."
            )
    else:
        # Baseline-only inference still uses the longitudinal model interface.
        # We duplicate baseline PET/CT into the interim branch and keep the baseline output head.
        pet2 = pet1
        ct2 = ct1

    return pet1, ct1, pet2, ct2


def run() -> int:
    args = parse_args()
    _show_torch_cuda_info()

    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    pet1, ct1, pet2, ct2 = validate_inputs(
        input_dir=args.input_dir,
        generate_interim=args.generate_interim_lesion_mask,
        use_dicom=args.use_dicom_input,
        pt1_path=args.input_PET1,
        ct1_path=args.input_CT1,
        pt2_path=args.input_PET2,
        ct2_path=args.input_CT2,
    )
    if args.generate_interim_lesion_mask:
        ckpt_path_list = checkpoint_paths(args.model_root, args.model_type)
    else:
        ckpt_path_list = checkpoint_paths(args.model_root, "rigid")

    missing = [str(p) for p in ckpt_path_list if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(missing))
    print(f"Using {args.model_type} checkpoints from {args.model_root}")

    inf = inference(
        roi_size=[112, 112, 112],
        resample_resolution=[3.0, 3.0, 3.0],
        modality="PET",
        extra_modalities={"image2": "CT", "image3": "PET", "image4": "CT"},
    )

    inf.infer_image(
        ckpt_path_list=ckpt_path_list,
        PET2_location=pet2,
        CT2_location=ct2,
        PET1_location=pet1,
        CT1_location=ct1,
        mask_location=args.output_dir,
        is_interim=args.generate_interim_lesion_mask,
    )
    if args.use_dicom_input:
        if args.generate_interim_lesion_mask:
            generate_RTstruct(
                args.input_PET2,
                str(args.output_dir) + "/output_interim.nii.gz",
                str(args.output_dir) + "/output_interim.dcm",
            )
        else:
            generate_RTstruct(
                args.input_PET1,
                str(args.output_dir) + "/output_baseline.nii.gz",
                str(args.output_dir) + "/output_baseline.dcm",
            )
    else:
        print(
            "DICOM input not used, skipping RTstruct generation. To enable, provide DICOM inputs and set --use-dicom-input."
        )
    return 0


def _show_torch_cuda_info() -> None:
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    available = torch.cuda.is_available()
    print(f"Torch CUDA is available: {available}")
    if available:
        current_device = torch.cuda.current_device()
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {current_device}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())

"""
docker run --rm --gpus all \
  -v /mnt/DGXUserData/sxc106/Xin/Monai_Auto3dSeg/COG_lymph_seg/Docker/test:/input \
  -v /mnt/DGXUserData/sxc106/Xin/Monai_Auto3dSeg/COG_lymph_seg/Docker/test:/output \
  lasnet-lymphoma \
  --input-dir /input/input \
  --output-dir /output/output \
  --generate-interim-lesion-mask \
  --model-type rigid

docker run --rm --gpus all \
  -v /mnt/DGXUserData/sxc106/Xin/Monai_Auto3dSeg/COG_lymph_seg/Docker/test/862538:/input \
  -v /mnt/DGXUserData/sxc106/Xin/Monai_Auto3dSeg/COG_lymph_seg/Docker/test/862538:/output \
  lasnet-lymphoma \
  --use-dicom-input \
  --input-dir /input/nifti \
  --output-dir /output/output \
  --input-PET1 /input/2016_06_PT_rigid \
  --input-CT1 /input/2016_06_CT_rigid \
  --input-PET2 /input/2016_08_PT \
  --input-CT2 /input/2016_08_CT \
  --generate-interim-lesion-mask \
  --model-type rigid

docker run --rm --gpus all \
  -v /mnt/DGXUserData/sxc106/Xin/Monai_Auto3dSeg/COG_lymph_seg/Docker/test/862538:/input \
  -v /mnt/DGXUserData/sxc106/Xin/Monai_Auto3dSeg/COG_lymph_seg/Docker/test/862538:/output \
  lasnet-lymphoma \
  --use-dicom-input \
  --input-dir /input/nifti \
  --output-dir /output/output \
  --input-PET1 /input/2016_06_PT \
  --input-CT1 /input/2016_06_CT \
  --model-type rigid

docker run --rm --gpus all \
  -v /mnt/DGXUserData/sxc106/Xin/Monai_Auto3dSeg/COG_lymph_seg/Docker/test/862538:/input \
  -v /mnt/DGXUserData/sxc106/Xin/Monai_Auto3dSeg/COG_lymph_seg/Docker/test/862538:/output \
  lasnet-lymphoma \
  --input-dir /input/nifti \
  --output-dir /output/output \
  --model-type rigid
"""
