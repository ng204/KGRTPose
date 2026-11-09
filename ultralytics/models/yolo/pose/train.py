# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_results


class PoseTrainer(yolo.detect.DetectionTrainer):
    """A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (tuple): Names of the loss components used in training.

    Methods:
        get_model: Retrieve a pose estimation model with specified configuration.
        set_model_attributes: Set keypoints shape attribute on the model.
        get_validator: Create a validator instance for model evaluation.
        plot_training_samples: Visualize training samples with keypoints.
        plot_metrics: Generate and save training/validation metric plots.
        get_dataset: Retrieve the dataset and ensure it contains required kpt_shape key.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a PoseTrainer object for training YOLO pose estimation models.

        This initializes a trainer specialized for pose estimation tasks, setting the task to 'pose' and
        handling specific configurations needed for keypoint detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Examples:
            >>> from ultralytics.models.yolo.pose import PoseTrainer
            >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
            >>> trainer = PoseTrainer(overrides=args)
            >>> trainer.train()

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        # 注册骨骼对称约束回调，在训练开始时注入 sym_pairs
        self.add_callback("on_train_start", lambda trainer: trainer._inject_symmetry_pairs())

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> PoseModel:
        """Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (PoseModel): Initialized pose estimation model.
        """
        model = PoseModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Set keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Return an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "bone_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _inject_symmetry_pairs(self):
        """从 data.yaml 的 flip_idx 推导对称骨骼对，注入到 criterion 中。.

        在第一个 epoch 开始之前调用（通过 on_train_start 回调触发）。
        不需要遍历训练数据，只需读取 flip_idx 和 skeleton。
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(model, "criterion") or model.criterion is None:
            model.criterion = model.init_criterion()

        criterion = model.criterion
        if not hasattr(criterion, "set_sym_pairs"):
            return  # 当前 loss 不支持对称约束，静默跳过

        flip_idx = self.data.get("flip_idx", [])
        skeleton = getattr(criterion, "skeleton", None)
        if not flip_idx or not skeleton:
            LOGGER.warning("PoseTrainer: flip_idx 或 skeleton 缺失，跳过骨骼对称约束。")
            return

        # 从 flip_idx 推导对称骨骼对
        # 对于骨骼 [i,j]，其对称骨骼为 [flip_idx[i], flip_idx[j]]
        skel_set = {(min(a, b), max(a, b)): (a, b) for a, b in skeleton}
        sym_pairs = []
        seen = set()
        for i, j in skeleton:
            fi, fj = flip_idx[i], flip_idx[j]
            key_orig = (min(i, j), max(i, j))
            key_sym = (min(fi, fj), max(fi, fj))
            if key_sym in skel_set and key_sym != key_orig:
                pair_key = (min(key_orig, key_sym), max(key_orig, key_sym))
                if pair_key not in seen:
                    seen.add(pair_key)
                    sym_pairs.append(((i, j), skel_set[key_sym]))

        if not sym_pairs:
            LOGGER.warning("PoseTrainer: 未找到对称骨骼对（flip_idx 中无对称关键点），跳过骨骼对称约束。")
            return

        criterion.set_sym_pairs(sym_pairs)
        LOGGER.info(f"PoseTrainer: 注入 {len(sym_pairs)} 对对称骨骼约束 ✓  {sym_pairs}")

    def on_train_start(self):
        """训练开始时注入骨骼对称约束对。."""
        super_fn = getattr(super(), "on_train_start", None)
        if super_fn is not None:
            super_fn()
        self._inject_symmetry_pairs()

    def plot_metrics(self):
        """Plot training/validation metrics."""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png

    def get_dataset(self) -> dict[str, Any]:
        """Retrieve the dataset and ensure it contains the required `kpt_shape` key.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.

        Raises:
            KeyError: If the `kpt_shape` key is not present in the dataset.
        """
        data = super().get_dataset()
        if "kpt_shape" not in data:
            raise KeyError(f"No `kpt_shape` in the {self.args.data}. See https://docs.ultralytics.com/datasets/pose/")
        return data
