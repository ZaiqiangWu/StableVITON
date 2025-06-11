import os
from os.path import join as opj

import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.DensePose.densepose_extractor import DensePoseExtractor

from viton_utils import get_mask_location
from PIL import Image

def imread(
        p, h, w,
        is_mask=False,
        in_inverse_mask=False,
        img=None
):
    if img is None:
        img = cv2.imread(p)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w, h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:, :, None]
        if in_inverse_mask:
            img = 1 - img
    return img


def imread_for_albu(
        p,
        is_mask=False,
        in_inverse_mask=False,
        cloth_mask_check=False,
        use_resize=False,
        height=512,
        width=384,
):
    img = cv2.imread(p)
    if use_resize:
        img = cv2.resize(img, (width, height))
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img >= 128).astype(np.float32)
        if cloth_mask_check:
            if img.sum() < 30720 * 4:
                img = np.ones_like(img).astype(np.float32)
        if in_inverse_mask:
            img = 1 - img
        img = np.uint8(img * 255.0)
    return img


def norm_for_albu(img, is_mask=False):
    if not is_mask:
        img = (img.astype(np.float32) / 127.5) - 1.0
    else:
        img = img.astype(np.float32) / 255.0
        img = img[:, :, None]
    return img


class VITONHDDataset(Dataset):
    def __init__(
            self,
            data_root_dir,
            img_H,
            img_W,
            is_paired=True,
            is_test=False,
            is_sorted=False,
            transform_size=None,
            transform_color=None,
            **kwargs
    ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "test"
        self.is_test = True
        self.resize_ratio_H = 1.0
        self.resize_ratio_W = 1.0

        self.resize_transform = A.Resize(img_H, img_W)
        self.transform_size = None
        self.transform_crop_person = None
        self.transform_crop_cloth = None
        self.transform_color = None

        #### spatial aug >>>>
        transform_crop_person_lst = []
        transform_crop_cloth_lst = []
        transform_size_lst = [A.Resize(int(img_H * self.resize_ratio_H), int(img_W * self.resize_ratio_W))]



        self.transform_crop_person = A.Compose(
            transform_crop_person_lst,
            additional_targets={"agn": "image",
                                "agn_mask": "image",
                                "cloth_mask_warped": "image",
                                "cloth_warped": "image",
                                "image_densepose": "image",
                                "image_parse": "image",
                                "gt_cloth_warped_mask": "image",
                                }
        )
        self.transform_crop_cloth = A.Compose(
            transform_crop_cloth_lst,
            additional_targets={"cloth_mask": "image"}
        )

        self.transform_size = A.Compose(
            transform_size_lst,
            additional_targets={"agn": "image",
                                "agn_mask": "image",
                                "cloth": "image",
                                "cloth_mask": "image",
                                "cloth_mask_warped": "image",
                                "cloth_warped": "image",
                                "image_densepose": "image",
                                "image_parse": "image",
                                "gt_cloth_warped_mask": "image",
                                }
        )






        self.cloth_name=None
        self.person_image=None
        self.openpose_model = OpenPose(0)
        self.parsing_model = Parsing(0)
        self.densepose_extractor = DensePoseExtractor()

    def __len__(self):
        return 1

    def set_cloth_name(self, cloth_name):
        self.cloth_name = cloth_name
    def set_person_image(self,img,isRGB=False):
        assert img.shape[2]==3
        self.person_image = cv2.resize(img,(768,1024))
        densepose = self.densepose_extractor.get_dp_map(self.person_image, isRGB=False)
        if not isRGB:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(img)
        img=img.resize((768,1024),Image.Resampling.BILINEAR)
        keypoints = self.openpose_model(img.resize((384, 512),Image.Resampling.BILINEAR))
        model_parse, _ = self.parsing_model(img.resize((384, 512),Image.Resampling.BILINEAR))
        category_dict_utils = ['upper_body', 'lower_body', 'dresses']
        model_type = 'dc'#hd

        mask, mask_gray = get_mask_location(model_type, category_dict_utils[0], model_parse, keypoints)
        mask = mask.resize((768, 1024), Image.NEAREST)
        mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

        masked_vton_img = Image.composite(mask_gray, img, mask)
        masked_vton_img = np.array(masked_vton_img)
        masked_vton_img=cv2.cvtColor(masked_vton_img, cv2.COLOR_BGR2RGB)
        self.agn=masked_vton_img
        self.agn_mask=np.array(mask)
        self.image=self.person_image
        self.image_densepose=densepose
        print(self.agn.shape)
        print(self.agn_mask.shape)
        print(self.image_densepose.shape)
        print(self.image.shape)

    def __getitem__(self, idx):


        agn = self.agn
        agn_mask = self.agn_mask
        cloth = imread(
                opj('./data', "cloth", self.cloth_name+'.jpg'),
                self.img_H,
                self.img_W
            )
        cloth_mask = imread(
                opj('./data', "cloth_mask", self.cloth_name+'.jpg'),
                self.img_H,
                self.img_W,
                is_mask=True,
                #cloth_mask_check=True
            )

        gt_cloth_warped_mask = np.zeros_like(agn_mask)

        image = self.image
        image_densepose = self.image_densepose

        return dict(
            agn=agn,#maksed image
            agn_mask=agn_mask,
            cloth=cloth,
            cloth_mask=cloth_mask,
            image=image,
            image_densepose=image_densepose,
            gt_cloth_warped_mask=gt_cloth_warped_mask,
            txt="",
            img_fn=1,
            cloth_fn=1,
        )