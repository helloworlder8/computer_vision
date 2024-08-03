---
comments: true
description: Discover how to customize and extend base Ultralytics YOLO Trainer engines. Support your custom model and dataloader by overriding built-in functions.
keywords: Ultralytics, YOLO, trainer engines, Engine_Trainer, Detection_Trainer, customizing trainers, extending trainers, custom model, custom dataloader
---

Both the Ultralytics YOLO command-line and Python interfaces are simply a high-level abstraction on the base engine executors. Let's take a look at the Trainer engine.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=104"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLOv8: Advanced Customization
</p>

## Engine_Trainer

Engine_Trainer contains the generic boilerplate training routine. It can be customized for any task_name based over overriding the required functions or operations as long the as correct formats are followed. For example, you can support your own custom model and dataloader by just overriding these functions:

- `build_model(cfg, weights)` - The function that builds the model to be trained
- `get_dataloader()` - The function that builds the dataloader More details and source code can be found in [`Engine_Trainer` Reference](../reference/engine/trainer.md)

## Detection_Trainer

Here's how you can use the YOLOv8 `Detection_Trainer` and customize it.

```python
from ultralytics.models.yolo.detect import Detection_Trainer

trainer = Detection_Trainer(overrides={...})
trainer.DDP_or_normally_train()
trained_model = trainer.best  # get best model
```

### Customizing the Detection_Trainer

Let's customize the trainer **to train a custom detection model** that is not supported directly. You can do this by simply overloading the existing the `build_model` functionality:

```python
from ultralytics.models.yolo.detect import Detection_Trainer


class CustomTrainer(Detection_Trainer):
    def build_model(self, cfg, weights):
        ...


trainer = CustomTrainer(overrides={...})
trainer.DDP_or_normally_train()
```

You now realize that you need to customize the trainer further to:

- Customize the `loss function`.
- Add `callback` that uploads model to your Google Drive after every 10 `epochs` Here's how you can do it:

```python
from ultralytics.models.yolo.detect import Detection_Trainer
from ultralytics.nn.tasks import Detection_Model


class MyCustomModel(Detection_Model):
    def build_loss_class(self):
        ...


class CustomTrainer(Detection_Trainer):
    def build_model(self, cfg, weights):
        return MyCustomModel(...)


# callback to upload model weights
def log_model(trainer):
    last_weight_path = trainer.last
    print(last_weight_path)


trainer = CustomTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
trainer.DDP_or_normally_train()
```

To know more about Callback triggering events and entry point, checkout our [Callbacks Guide](callbacks.md)

## Other engine components

There are other components that can be customized similarly like `Validators` and `Predictors`. See Reference section for more information on these.
