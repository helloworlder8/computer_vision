# Ultralytics YOLO 🚀, AGPL-3.0 license

import itertools

from ultralytics.data import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import DEFAULT_CFG, RANK, checks
from ultralytics.utils.torch_utils import de_parallel

#专属yoloworld的回调函数
def on_pretrain_routine_end(trainer):
    """Callback."""
    if RANK in {-1, 0}:
        # NOTE: for evaluation
        names = [name.split("/")[0] for name in list(trainer.test_loader.dataset.data_dict["names"].values())]
        de_parallel(trainer.ema.ema).generate_name_feats(names, cache_clip_model=False) #clip模型发力了
    device = next(trainer.model.parameters()).device
    trainer.text_model, _ = trainer.clip.load("ViT-B/32", device=device)
    for p in trainer.text_model.parameters():
        p.requires_grad_(False) #clip模型的参数不更新


class WorldTrainer(yolo.detect.DetectionTrainer):
    """
    A class to fine-tune a world model on a close-set dataset.

    Example:
        ```python
        from ultralytics.models.yolo.world import WorldTrainer

        args = dict(model='yolov8s-world.pt', data='coco8.yaml', epochs=3)
        trainer = WorldTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

        # Import and assign clip
        try:
            import clip
        except ImportError:
            checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        self.clip = clip

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return WorldModel initialized with specified config and weights."""
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.

        ch = self.data_dict["ch"] if "ch" in self.data_dict else 3
        model = WorldModel(cfg,ch=ch,nc=min(self.data_dict["nc"], 80),verbose=verbose and RANK == -1,)
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    # 重载 主要是多模态训练
    def _build_dataset(self, img_path, mode="train", batch=None):
        
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, self.data_dict, batch, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )

    def _train_data_preprocess(self, batch):
        """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""
        batch = super()._train_data_preprocess(batch)

        # NOTE: add text features
        texts = list(itertools.chain(*batch["texts"]))
        text_token = self.clip.tokenize(texts).to(batch["img"].device)
        name_feats = self.text_model.encode_text(text_token).to(dtype=batch["img"].dtype)  # torch.float32
        name_feats = name_feats / name_feats.norm(p=2, dim=-1, keepdim=True)
        batch["name_feats"] = name_feats.reshape(len(batch["texts"]), -1, name_feats.shape[-1]) #批 token 维度
        return batch
