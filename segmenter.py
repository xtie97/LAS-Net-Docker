import os
from typing import Optional
import numpy as np
import torch
from monai.data import decollate_batch, list_data_collate
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    ConcatItemsd,
    Lambdad,
    LabelToMaskd,
    CropForegroundd,
    DeleteItemsd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    NormalizeIntensityd,
    ResampleToMatchd,
    Spacingd,
    ScaleIntensityRanged,
)
from monai.utils import convert_to_dst_type
from glob import glob
import SimpleITK as sitk
from model import get_network
from skimage import morphology, measure
import scipy.ndimage as ndimage
from utils import con_comp

suv_threshold = 0.2


# convert the logits to prediction (apply softmax)
def logits2pred(logits, dim=1):
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return torch.softmax(logits, dim=dim)


class DataTransformBuilder:
    def __init__(
        self,
        roi_size: list,
        image_key: str = "image",
        resample: bool = True,
        resample_resolution: Optional[list] = [3.0, 3.0, 3.0],
        normalize_mode: str = "PET",
        normalize_params: Optional[dict] = None,
        extra_modalities: Optional[dict] = None,
    ) -> None:

        self.roi_size, self.image_key = roi_size, image_key

        self.resample, self.resample_resolution = resample, resample_resolution
        self.normalize_mode = normalize_mode
        self.normalize_params = normalize_params if normalize_params is not None else {}
        self.extra_modalities = extra_modalities if extra_modalities is not None else {}

    def get_load_transforms(self):
        ts = []
        keys = [self.image_key] + list(self.extra_modalities)
        ts.append(
            LoadImaged(
                keys=keys,
                ensure_channel_first=True,
                dtype=None,
                allow_missing_keys=True,
                image_only=True,
            )
        )
        ts.append(
            EnsureTyped(
                keys=keys,
                data_type="tensor",
                dtype=torch.float,
                allow_missing_keys=True,
            )
        )
        return ts

    def threshold_for_pet(self, x):
        # threshold at 0.2
        return x > suv_threshold  # SUV uptake = 1.0 +- 10% is a normal SUV uptake

    def get_resample_transforms(self, crop_foreground=True):
        ts = []
        keys = [self.image_key]
        extra_keys = self.extra_modalities  # dict
        # self.image_key is PET, self.extra_modalities is CT

        if crop_foreground:
            ts.append(
                CropForegroundd(
                    keys=keys,
                    source_key=self.image_key,
                    select_fn=self.threshold_for_pet,
                    margin=0,
                    allow_missing_keys=True,
                    allow_smaller=True,
                )
            )  # it can be accomplished in a pre-processing step

        if self.resample:
            if self.resample_resolution is None:
                raise ValueError("resample_resolution is not provided")
            pixdim = self.resample_resolution
            ts.append(
                Spacingd(
                    keys=keys,
                    pixdim=pixdim,
                    mode="bilinear",
                    dtype=torch.float,
                    allow_missing_keys=True,
                )
            )

        # match extra modalities to the key image.
        for extra_key in extra_keys:
            ts.append(
                ResampleToMatchd(
                    keys=extra_key,
                    key_dst=self.image_key,
                    dtype=np.float32,
                    mode="bilinear",
                )
            )

        return ts

    def get_normalize_transforms(self):
        ts = []
        modalities = {
            self.image_key: self.normalize_mode
        }  # default input is PET_interim
        modalities.update(self.extra_modalities)

        for key, normalize_mode in modalities.items():
            normalize_mode = normalize_mode.lower()
            if "pet" in normalize_mode:  # SUV input
                intensity_bounds = [0, 30]  # 0-30, 0-10
                ts.append(
                    ScaleIntensityRanged(
                        keys=key,
                        a_min=intensity_bounds[0],
                        a_max=intensity_bounds[1],
                        b_min=0,
                        b_max=1,
                        clip=True,
                    )
                )
            elif "ct" in normalize_mode:
                intensity_bounds = [-150, 250]
                ts.append(
                    ScaleIntensityRanged(
                        keys=key,
                        a_min=intensity_bounds[0],
                        a_max=intensity_bounds[1],
                        b_min=-1,
                        b_max=1,
                        clip=False,
                    )
                )
                ts.append(
                    Lambdad(keys=key, func=lambda x: torch.sigmoid(x))
                )  # scale to 0-1
            else:
                raise ValueError(
                    "Unsupported normalize_mode" + str(self.normalize_mode)
                )

        if len(self.extra_modalities) > 0:
            ts.append(
                ConcatItemsd(keys=list(modalities), name=self.image_key)
            )  # concatenate all modalities at the channels
            ts.append(DeleteItemsd(keys=list(self.extra_modalities)))  # release memory
        return ts

    @classmethod
    def get_postprocess_transform(cls, invert=False, transform=None) -> Compose:
        ts = []
        if invert and transform is not None:
            ts.append(
                Invertd(
                    keys="pred",
                    orig_keys="image",
                    transform=transform,
                    nearest_interp=False,
                )
            )

        return Compose(ts)

    def __call__(self) -> Compose:

        ts = []
        ts.extend(self.get_load_transforms())
        ts.extend(self.get_resample_transforms())
        ts.extend(self.get_normalize_transforms())

        return Compose(ts)


class inference:
    def __init__(
        self,
        roi_size: list,
        resample_resolution: Optional[list] = [3.0, 3.0, 3.0],
        modality: str = None,
        extra_modalities: Optional[dict] = None,
    ) -> None:
        # specify the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the model
        self.n_class = 2  # # number of classes is always 2 (background and lymphoma)
        model = get_network(img_size=roi_size, in_channels=2, n_class=self.n_class)
        model = model.to(self.device)
        self.model = model

        # Sliding window inference
        self.sliding_inferrer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=2,
            overlap=0.625,
            mode="gaussian",
            cache_roi_weight_map=True,
            progress=True,
            cpu_thresh=512**3 // self.n_class,
        )

        self.data_tranform_builder = DataTransformBuilder(
            roi_size=roi_size,
            resample_resolution=resample_resolution,
            normalize_mode=modality,
            extra_modalities=extra_modalities,
        )  # "PET", {"image2": "CT", "image3": "PET", "image4": "CT"}

        self.inf_transform = self.data_tranform_builder()
        self.post_transforms = DataTransformBuilder.get_postprocess_transform(
            invert=True, transform=self.inf_transform
        )

    def checkpoint_load(self, ckpt: str, model: torch.nn.Module):
        if not os.path.isfile(ckpt):
            raise ValueError("Invalid checkpoint file" + str(ckpt))
        else:
            checkpoint = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=True)

    @torch.no_grad()
    def infer_image(
        self,
        ckpt_path_list: list,
        PET2_location: str,
        CT2_location: str,
        PET1_location: str,
        CT1_location: str,
        mask_location: str,
        is_interim: bool = False,
    ) -> None:
        self.model.eval()

        PET2_file = PET2_location
        CT2_file = CT2_location
        PET1_file = PET1_location
        CT1_file = CT1_location

        batch_data = self.inf_transform(
            {
                "image": PET2_file,
                "image2": CT2_file,
                "image3": PET1_file,
                "image4": CT1_file,
            }
        )
        batch_data = list_data_collate([batch_data])

        data = batch_data["image"].as_subclass(torch.Tensor).to(self.device)

        pred1 = torch.zeros(
            data.shape[0], self.n_class, *data.shape[2:], device=self.device
        )
        pred2 = torch.zeros_like(pred1)
        for ckpt_path in ckpt_path_list:
            self.checkpoint_load(ckpt=ckpt_path, model=self.model)
            logits1, logits2 = self.sliding_inferrer(inputs=data, network=self.model)
            pred1 += logits2pred(logits=logits1.float())  # interim prediction
            pred2 += logits2pred(logits=logits2.float())  # baseline prediction

        if is_interim:
            # interim PET masks
            pred1 /= len(ckpt_path_list)
            batch_data["pred"] = convert_to_dst_type(
                pred1, batch_data["image"], dtype=pred1.dtype, device=pred1.device
            )[0]
            pred1 = torch.stack(
                [self.post_transforms(x)["pred"] for x in decollate_batch(batch_data)]
            )
            pred1 = pred1.argmax(dim=1, keepdim=True).squeeze()
            pred1 = pred1.permute(2, 1, 0)
            pred1 = pred1.detach().cpu().numpy()
            write_mask_file(mask_location, pred1, PET2_file, True)
        else:
            # baseline PET masks
            pred2 /= len(ckpt_path_list)
            batch_data["pred"] = convert_to_dst_type(
                pred2, batch_data["image"], dtype=pred2.dtype, device=pred2.device
            )[0]
            pred2 = torch.stack(
                [self.post_transforms(x)["pred"] for x in decollate_batch(batch_data)]
            )
            pred2 = pred2.argmax(dim=1, keepdim=True).squeeze()
            pred2 = pred2.permute(2, 1, 0)
            pred2 = pred2.detach().cpu().numpy()
            write_mask_file(mask_location, pred2, PET1_file, False)


# write the mask file
def write_mask_file(
    location: str, segmentation: np.ndarray, input_file: str, is_interim: bool = False
):
    location.mkdir(parents=True, exist_ok=True)
    pred_seg = np.where(segmentation == 1, 1, 0)
    pred_final = np.zeros_like(pred_seg)
    if np.sum(pred_seg) > 0:
        image = sitk.ReadImage(input_file)
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        assert (
            pred_seg.shape == image_array.shape
        ), "The shape of the predicted segmentation does not match the shape of the input image."
        # get the pixel dimension of segmentation image
        voxel_size = image.GetSpacing()
        voxel_volume = voxel_size[0] * voxel_size[1] * voxel_size[2]  # 3*3*3 mm3
        # use connected components
        pred_seg = con_comp(pred_seg)
        for c in range(1, np.max(pred_seg) + 1):
            maps = np.zeros_like(pred_seg)
            maps[pred_seg == c] = 1
            if np.sum(maps) * voxel_volume < 200:  # 200 mm3
                continue  # remove small connected components
            # dilate the mask by 1 voxel and reapply the thresholding
            maps = morphology.binary_dilation(maps, morphology.ball(radius=1))
            if not is_interim:  # baseline prediction
                max_SUV_in_map = np.max(maps * image_array)
                maps_1 = np.logical_and(maps, maps * image_array > 2.5)
                maps_2 = np.logical_and(maps, maps * image_array > 0.4 * max_SUV_in_map)
                maps = np.logical_or(maps_1, maps_2)
            maps = ndimage.binary_closing(maps, iterations=9)  # 3D closing
            pred_final = np.logical_or(pred_final, maps)
        segmentation = pred_final.astype(np.uint8)

    # convert the numpy array back to a SimpleITK image
    segmentation_image = sitk.GetImageFromArray(segmentation)

    segmentation_image.CopyInformation(image)
    # Cast the segmentation image to 8-bit unsigned int
    segmentation_image = sitk.Cast(segmentation_image, sitk.sitkUInt8)
    # Write to a MHA file
    suffix = "_interim.nii.gz" if is_interim else "_baseline.nii.gz"

    sitk.WriteImage(
        segmentation_image, location / f"output{suffix}", useCompression=True
    )
