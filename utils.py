from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import nibabel as nib
import dicom2nifti
import dicom2nifti.settings as settings
from pydicom import dcmread
from pydicom.dataset import Dataset
from rt_utils import RTStructBuilder
import cc3d

settings.disable_validate_slice_increment()


def process_dicom(dicom_series_dir: str | Path, nifti_file: str | Path) -> None:
    """
    Convert one DICOM series folder to a NIfTI file.
    """
    dicom_series_dir = Path(dicom_series_dir)
    nifti_file = Path(nifti_file)
    nifti_file.parent.mkdir(parents=True, exist_ok=True)

    dicom2nifti.dicom_series_to_nifti(
        str(dicom_series_dir),
        str(nifti_file),
        reorient_nifti=False,
    )


def read_first_dicom(dicom_series_dir: str | Path) -> Dataset:
    """
    Read the first DICOM file from a DICOM series directory.
    """
    dicom_series_dir = Path(dicom_series_dir)

    dicom_files = sorted(dicom_series_dir.glob("*.dcm"))
    if not dicom_files:
        dicom_files = sorted([p for p in dicom_series_dir.iterdir() if p.is_file()])

    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_series_dir}")

    return dcmread(str(dicom_files[0]), stop_before_pixels=True, force=True)


def _get_dicom_value(ds: Dataset, tag: tuple[int, int], default=None):
    return ds[tag].value if tag in ds else default


def _digits_only(value) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def _combine_dicom_date_time(date_value, time_value) -> str:
    """
    Convert DICOM date and time values into YYYYMMDDHHMMSS.
    Handles fractional seconds by keeping only digits.
    """
    date_str = _digits_only(date_value)[:8]
    time_str = _digits_only(time_value)

    if len(date_str) != 8:
        raise ValueError(f"Invalid DICOM date: {date_value}")

    if len(time_str) == 0:
        raise ValueError("Missing DICOM time.")

    time_str = (time_str + "000000")[:6]
    return date_str + time_str


def _get_scan_datetime(ds: Dataset) -> datetime:
    """
    Extract scan datetime from PET DICOM metadata.
    """
    manufacturer = str(_get_dicom_value(ds, (0x0008, 0x0070), "")).lower()

    # GE private scan datetime tag used in some PET DICOMs.
    if manufacturer.startswith("ge") and (0x0009, 0x100D) in ds:
        scan_datetime_str = _digits_only(ds[(0x0009, 0x100D)].value)[:14]
        return datetime.strptime(scan_datetime_str, "%Y%m%d%H%M%S")

    series_date = _get_dicom_value(ds, (0x0008, 0x0021), "")
    acquisition_date = _get_dicom_value(ds, (0x0008, 0x0022), "")
    study_date = _get_dicom_value(ds, (0x0008, 0x0020), "")

    scan_date = series_date or acquisition_date or study_date
    if not scan_date:
        raise ValueError("No valid PET scan date found in DICOM metadata.")

    series_time = _get_dicom_value(ds, (0x0008, 0x0031), "")
    acquisition_time = _get_dicom_value(ds, (0x0008, 0x0032), "")
    scan_time = series_time or acquisition_time

    scan_datetime_str = _combine_dicom_date_time(scan_date, scan_time)
    return datetime.strptime(scan_datetime_str, "%Y%m%d%H%M%S")


def _get_injection_datetime(
    ds: Dataset, radiopharm_ds: Dataset, scan_datetime: datetime
) -> datetime:
    """
    Extract injection datetime from the Radiopharmaceutical Information Sequence.
    """
    injection_datetime = _get_dicom_value(radiopharm_ds, (0x0018, 0x1078), None)

    if injection_datetime:
        injection_datetime_str = _digits_only(injection_datetime)[:14]
    else:
        injection_time = _get_dicom_value(radiopharm_ds, (0x0018, 0x1072), "")
        injection_datetime_str = _combine_dicom_date_time(
            scan_datetime.strftime("%Y%m%d"),
            injection_time,
        )

    return datetime.strptime(injection_datetime_str, "%Y%m%d%H%M%S")


def get_suv_conversion_factor(ds: Dataset) -> Tuple[bool, str, float]:
    """
    Calculate SUVbw conversion factor for PET data stored in BQML.

    Returns:
        attenuation_corrected: whether attenuation correction appears to be applied
        unit: DICOM unit string
        suv_factor: multiplicative factor to convert BQML activity concentration to SUVbw
    """
    corrections = str(_get_dicom_value(ds, (0x0028, 0x0051), "")).upper()
    attenuation_corrected = "ATTN" in corrections

    unit = str(getattr(ds, "Units", "None")).upper()

    patient_weight_kg = float(_get_dicom_value(ds, (0x0010, 0x1030), 0) or 0)
    if patient_weight_kg <= 0:
        raise ValueError("Missing or invalid patient weight in PET DICOM metadata.")

    radiopharm_sequence = _get_dicom_value(ds, (0x0054, 0x0016), None)
    if not radiopharm_sequence:
        raise ValueError("Missing Radiopharmaceutical Information Sequence.")

    radiopharm_ds = radiopharm_sequence[0]

    injected_dose_bq = _get_dicom_value(radiopharm_ds, (0x0018, 0x1074), None)
    half_life_seconds = _get_dicom_value(radiopharm_ds, (0x0018, 0x1075), None)

    if injected_dose_bq is None or half_life_seconds is None:
        raise ValueError(
            "Missing injected dose or radionuclide half-life in PET DICOM metadata."
        )

    injected_dose_bq = float(injected_dose_bq)
    half_life_seconds = float(half_life_seconds)

    scan_datetime = _get_scan_datetime(ds)
    injection_datetime = _get_injection_datetime(ds, radiopharm_ds, scan_datetime)

    elapsed_seconds = (scan_datetime - injection_datetime).total_seconds()
    while elapsed_seconds < 0:
        elapsed_seconds += 24 * 3600

    decay_corrected_dose_bq = injected_dose_bq * 2 ** (
        -elapsed_seconds / half_life_seconds
    )

    if decay_corrected_dose_bq <= 0:
        raise ValueError("Invalid decay-corrected injected dose.")

    suv_factor = patient_weight_kg * 1000.0 / decay_corrected_dose_bq

    return attenuation_corrected, unit, suv_factor


def convert_nifti_to_suv(pet_nifti_file: str | Path, suv_factor: float) -> None:
    """
    Convert a PET NIfTI image from BQML to SUVbw in place.
    """
    pet_nifti_file = Path(pet_nifti_file)

    pet_img = nib.load(str(pet_nifti_file))
    pet_data = pet_img.get_fdata(dtype=np.float32)

    suv_data = pet_data * np.float32(suv_factor)
    suv_img = nib.Nifti1Image(suv_data, pet_img.affine, pet_img.header)

    nib.save(suv_img, str(pet_nifti_file))


def convert_pet_series_to_suv_nifti(
    pet_dicom_dir: str | Path,
    output_nifti_file: str | Path,
    scan_label: str,
) -> None:
    """
    Convert one PET DICOM series to NIfTI and convert the image intensity to SUVbw.
    """
    pet_dicom_dir = Path(pet_dicom_dir)
    output_nifti_file = Path(output_nifti_file)

    ds = read_first_dicom(pet_dicom_dir)
    attenuation_corrected, unit, suv_factor = get_suv_conversion_factor(ds)

    if not attenuation_corrected:
        raise ValueError(
            f"Attenuation correction is not applied to the {scan_label} PET scan."
        )

    if unit.upper() != "BQML":
        raise ValueError(
            f"Unexpected unit '{unit}' in {scan_label} PET DICOM metadata. "
            "Expected BQML."
        )

    process_dicom(pet_dicom_dir, output_nifti_file)
    convert_nifti_to_suv(output_nifti_file, suv_factor)


def process_pt_ct(
    pt1_path: str | Path,
    ct1_path: str | Path,
    save_dir: str | Path,
    pt2_path: Optional[str | Path] = None,
    ct2_path: Optional[str | Path] = None,
) -> None:
    """
    Convert baseline and optional interim PET/CT DICOM series to NIfTI.

    Output folder structure is compatible with the current Docker inference code:

        save_dir/
            input/PET1.nii.gz
            input/CT1.nii.gz
            input/PET2.nii.gz   optional
            input/CT2.nii.gz    optional
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if (pt2_path is None) != (ct2_path is None):
        raise ValueError(
            "Interim PET and interim CT must either both be provided or both be absent."
        )

    pet1_nifti = save_dir / "PET1.nii.gz"
    ct1_nifti = save_dir / "CT1.nii.gz"

    convert_pet_series_to_suv_nifti(
        pet_dicom_dir=pt1_path,
        output_nifti_file=pet1_nifti,
        scan_label="baseline",
    )
    process_dicom(ct1_path, ct1_nifti)

    if pt2_path is not None and ct2_path is not None:
        pet2_nifti = save_dir / "PET2.nii.gz"
        ct2_nifti = save_dir / "CT2.nii.gz"

        convert_pet_series_to_suv_nifti(
            pet_dicom_dir=pt2_path,
            output_nifti_file=pet2_nifti,
            scan_label="interim",
        )
        process_dicom(ct2_path, ct2_nifti)


# sample different color codes
def sample_color_codes():
    color_codes = []
    for i in range(0, 256, 10):
        for j in range(0, 256, 10):
            for k in range(0, 256, 10):
                color_codes.append([i, j, k])
    return color_codes


# apply connected component analysis to the segmentation array
def con_comp(seg_array):
    # input: a binary segmentation array output: an array with seperated (indexed) connected components of the segmentation array
    connectivity = 18  # 18 or 26
    conn_comp = cc3d.connected_components(seg_array, connectivity=connectivity)
    return conn_comp


def generate_RTstruct(dicom_series_path, nifti_label_path, save_rtstruct_path):

    color_codes = sample_color_codes()
    # set random generator seed
    np.random.seed(1)
    # shuffle color codes
    np.random.shuffle(color_codes)

    # load nifti label
    nifti_label = nib.load(nifti_label_path).get_fdata()
    # permute the axes to match the dicom series
    nifti_label = np.transpose(nifti_label, (1, 0, 2))
    # get connected components
    nifti_label_comp = con_comp(nifti_label)

    # reassgin the study instance UID
    rtstruct_new = RTStructBuilder.create_new(dicom_series_path=dicom_series_path)
    count = 0
    for i in range(1, nifti_label_comp.max() + 1):
        mask_3d = nifti_label_comp == i
        if np.sum(mask_3d) < 1:
            continue
        roi_new_name = f"ROI-{count+1}"
        rtstruct_new.add_roi(
            mask=mask_3d,
            color=color_codes[count],
            name=roi_new_name,
            use_pin_hole=True,
            approximate_contours=False,
        )
        count += 1

    rtstruct_new.save(save_rtstruct_path)
    ds = dcmread(save_rtstruct_path)
    ds.SeriesDescription = "AI_lymphoma_seg"

    with open(save_rtstruct_path, "wb") as outfile:
        ds.save_as(outfile)
