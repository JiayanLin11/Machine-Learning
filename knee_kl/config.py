
from dataclasses import dataclass


@dataclass
class Config:
    """集中管理实验配置、超参数与消融开关。"""

    # 数据相关
    data_root: str = "~/datasets/knee-kl/KneeXrayData/ClsKLData/kneeKL224"
    img_size: int = 224
    center_crop_ratio: float = 0.8
    num_classes: int = 5
    n_folds: int = 5
    patient_aware_split: bool = True

    # 消融开关
    backbone: str = "convnext_tiny"
    attention: str = "triplet"
    loss_type: str = "cdw_ce"
    use_imbalance: bool = True
    use_oversampling: bool = True
    use_class_weight: bool = True
    use_contrastive: bool = True

    # 超参
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    cdw_power: float = 3.0
    contrastive_lambda: float = 0.5
    contrastive_temp: float = 0.1
    contrastive_beta: float = 1.0
    proj_dim: int = 128

    # 训练
    batch_size: int = 128
    epochs: int = 100
    early_stop_patience: int = 15
    seed: int = 42
    monitor_metric: str = "qwk"
    num_workers: int = 8
    output_dir: str = "outputs"
    save_checkpoints: bool = False

