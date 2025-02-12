# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import numpy as np

# ROS
import rospy
from mmdet.apis import inference_detector, init_detector
from mmdet.evaluation import INSTANCE_OFFSET

# viplanner-ros
from viplanner.config.coco_sem_meta import get_class_for_id_mmdet
from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler


class Mask2FormerInference:
    """Run Inference on Mask2Former model to estimate semantic segmentation"""

    debug: bool = False

    def __init__(
        self,
        config_file="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        checkpoint_file="model_final.pth",
    ) -> None:
        # Build the model from a config file and a checkpoint file
        self.model = init_detector(config_file, checkpoint_file, device="cuda:0")

        # mapping from coco class id to viplanner class id and color
        viplanner_meta = VIPlannerSemMetaHandler()
        coco_viplanner_cls_mapping = get_class_for_id_mmdet(self.model.dataset_meta["classes"])
        self.viplanner_sem_class_color_map = viplanner_meta.class_color
        self.coco_viplanner_color_mapping = {}
        for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items():
            self.coco_viplanner_color_mapping[coco_id] = viplanner_meta.class_color[viplanner_cls_name]

        return

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict semantic segmentation from image

        Args:
            image (np.ndarray): image to be processed in BGR format
        """

        result = inference_detector(self.model, image)
        result = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()[0]
        # create output
        panoptic_mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for curr_sem_class in np.unique(result):
            curr_label = curr_sem_class % INSTANCE_OFFSET
            try:
                panoptic_mask[result == curr_sem_class] = self.coco_viplanner_color_mapping[curr_label]
            except KeyError:
                if curr_sem_class != len(self.model.dataset_meta["classes"]):
                    rospy.logwarn(f"Category {curr_label} not found in" " coco_viplanner_cls_mapping.")
                panoptic_mask[result == curr_sem_class] = self.viplanner_sem_class_color_map["static"]

        if self.debug:
            import matplotlib.pyplot as plt

            plt.imshow(panoptic_mask)
            plt.show()

        return panoptic_mask


# EoF
