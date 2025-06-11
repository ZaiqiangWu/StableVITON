import argparse
import glob
import logging
import os
import sys
from typing import Any, ClassVar, Dict, List
import torch
sys.path.append("./preprocess/DensePose")

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer as Visualizer,
)
from .apply_net import create_argument_parser, DumpAction

import torch
import numpy as np
import cv2

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

class DensePoseExtractor(DumpAction):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.dp_model = DumpAction()
        #self.dp_model.add_arguments(self.parser)
        self.args = self.parser.parse_args([])
        opts = []
        self.cfg = './preprocess/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml'
        self.model = 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl'
        cfg = self.dp_model.setup_config(self.cfg, self.model, self.args, opts)
        self.predictor = DefaultPredictor(cfg)
        self.palette = np.array(get_palette(25), np.uint8).reshape(-1,3)

    def forward(self,img):
        img = img[:,:,[2,1,0]] # convert to BGR
        outputs = self.predictor(img)["instances"]#BGR fromat
        image_fpath = "file_name"
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]

        box_list = (result['pred_boxes_XYXY'][0]).tolist()
        box_list = [int(round(coord)) for coord in box_list]
        x_min, y_min, x_max, y_max = box_list
        labels = result['pred_densepose'][0].labels
        uv = result['pred_densepose'][0].uv
        mask_h, mask_w = labels.shape
        raw_h, raw_w, _ = img.shape

        #convert label to float
        labels = labels.float()/24.0
        output_tensor = torch.zeros(3, raw_h, raw_w).cuda()
        output_tensor[0,y_min:y_min+mask_h,x_min:x_min+mask_w]=labels
        output_tensor[1:, y_min:y_min + mask_h, x_min:x_min + mask_w] = uv

        return output_tensor

    def get_IUV(self,img, isRGB=False):
        if isRGB:
            img = img[:,:,[2,1,0]] # convert to BGR
        outputs = self.predictor(img)["instances"]
        image_fpath = "file_name"
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]

        if len(result['pred_boxes_XYXY'])==0:
            return None
        box_list = (result['pred_boxes_XYXY'][0]).tolist()
        box_list = [int(round(coord)) for coord in box_list]
        x_min, y_min, x_max, y_max = box_list
        labels = result['pred_densepose'][0].labels
        uv = result['pred_densepose'][0].uv
        mask_h, mask_w = labels.shape
        raw_h, raw_w, _ = img.shape

        #convert label to float
        labels = labels.float()#/24.0
        output_tensor = torch.zeros(3, raw_h, raw_w).cuda()
        output_tensor[0,y_min:y_min+mask_h,x_min:x_min+mask_w]=labels
        output_tensor[1:, y_min:y_min + mask_h, x_min:x_min + mask_w] = uv
        output_tensor[1:,:]*=255.0
        IUV = output_tensor.permute(1,2,0).cpu().numpy().astype(np.uint8)
        #print(labels.max())

        return IUV

    def get_dp_map(self,img, isRGB=False):
        if isRGB:
            img = img[:,:,[2,1,0]] # convert to BGR
        height,width=img.shape[:2]
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
        results = DensePoseResultExtractor()(outputs)

        # MagicAnimate uses the Viridis colormap for their training data
        cmap = cv2.COLORMAP_VIRIDIS
        # Visualizer outputs black for background, but we want the 0 value of
        # the colormap, so we initialize the array with that value
        arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
        out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)


        return out_frame

    def IUV2img(self,IUV:np.ndarray):
        IUV=IUV.astype(np.float32)
        IUV[:,:,0]/=24.0
        IUV[:,:,0]*=255
        IUV=IUV.astype(np.uint8)
        return IUV

    def get_dp_img(self,img,isRGB=False):
        return self.IUV2img(self.get_IUV(img,isRGB))

    def get_hand_mask(self,img):
        # input must be BGR
        outputs = self.predictor(img)["instances"]#BGR fromat
        image_fpath = "file_name"
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]

        raw_h, raw_w, _ = img.shape
        if len(result['pred_boxes_XYXY']) == 0:
            return np.zeros([raw_h, raw_w]).astype(bool)

        box_list = (result['pred_boxes_XYXY'][0]).tolist()
        box_list = [int(round(coord)) for coord in box_list]
        x_min, y_min, x_max, y_max = box_list
        labels = result['pred_densepose'][0].labels
        uv = result['pred_densepose'][0].uv
        mask_h, mask_w = labels.shape


        #convert label to float
        labels = labels.cpu().numpy().astype(np.uint8)
        raw_labels = np.zeros([raw_h, raw_w])
        raw_labels[y_min:y_min + mask_h, x_min:x_min + mask_w] = labels

        hand_mask = (raw_labels==3)|(raw_labels==4)

        return hand_mask

    def get_vis_img(self,img_path):
        output_tensor = self.forward(img_path).cpu()
        output_tensor = output_tensor.permute(1,2,0)*255
        output_img = output_tensor.numpy().astype(np.uint8)


        cv2.imwrite('seg.jpg',output_img)

# 1, 2 = Torso, 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot, 7, 9 = Upper Leg Right, 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left, 15, 17 = Upper Arm Left, 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right, 23, 24 = Head;