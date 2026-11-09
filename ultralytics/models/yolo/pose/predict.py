# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class PosePredictor(DetectionPredictor):
    """A class extending the DetectionPredictor class for prediction based on a pose model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize PosePredictor for pose estimation tasks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct the result object from the prediction, including keypoints."""
        result = super().construct_result(pred, img, orig_img, img_path)

        # 🔧 安全获取 kpt_shape (注意：以下4行必须与上一行 result=... 保持相同缩进)
        kpt_shape = getattr(self.model, "kpt_shape", None) or (6, 3)
        if len(pred) == 0:
            pred_kpts = torch.zeros(0, *kpt_shape, device=pred.device)
        else:
            nk = kpt_shape[0] * kpt_shape[1]
            pred_kpts = pred[:, -nk:].view(len(pred), *kpt_shape)
        pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)

        # 🔧 更新结果 (与上面代码块同级缩进)
        result.update(keypoints=pred_kpts)
        return result
