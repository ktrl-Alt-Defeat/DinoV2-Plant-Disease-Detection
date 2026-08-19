# Model

## Architecture

| Component | Value | Source |
| --- | --- | --- |
| Backbone | `dinov2_vitb14` | [configs/config.yaml:32](../configs/config.yaml) |
| Source repository | `facebookresearch/dinov2` via `torch.hub` | [src/model.py:25](../src/model.py) |
| Weights URL template | `https://dl.fbaipublicfiles.com/dinov2/{name}/{name}_pretrain.pth` | [src/model.py:28](../src/model.py) |
| Feature dimension | 768 | [configs/config.yaml:38](../configs/config.yaml) |
| Input resolution | 224 | [configs/config.yaml:35](../configs/config.yaml) |
| Head | `nn.Linear(768, num_classes)` | [src/model.py:374](../src/model.py) |
| Dropout | 0.0 → bare `Linear`; `>0` → `Sequential(Dropout, Linear)` | [src/model.py:375](../src/model.py) |

### Accepted backbones

`KNOWN_BACKBONES` ([src/model.py:35](../src/model.py)): `dinov2_vits14`,
`dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`. The tuple is documentation and
error-message material — `load_backbone` passes any configured name to
`torch.hub.load` ([src/model.py:334](../src/model.py)).

`SUPPORTED_CLASSIFIER_TYPES` = `{"linear"}` ([src/model.py:43](../src/model.py)).

## Build-time validation

`DinoV2Classifier.__init__` ([src/model.py:173](../src/model.py)) enforces:

| Check | Failure | Source |
| --- | --- | --- |
| Backbone advertises `embed_dim` or `num_features` | `ModelBuildError` | [:380](../src/model.py) |
| `model.feature_dim` equals the advertised width | `ModelBuildError` | [:398](../src/model.py) |
| `model.image_size % patch_size == 0` | `ModelBuildError` | [:418](../src/model.py) |

The backbone is the source of truth for `feature_dim`; the configured value is
checked against it, never trusted.

## `build_model` pipeline

[src/model.py:284](../src/model.py):

1. `ModelSpecification.from_config` + `ClassifierSpecification.from_config`
2. `load_backbone(name, pretrained=...)` via `torch.hub.load(..., trust_repo=True)`
3. `DinoV2Classifier(backbone, spec, classifier_spec)`
4. Optional `freeze_backbone()` when `model.freeze_backbone`
5. `model.to(device)` using `device.preferred`
6. `model.eval()`

`build_model` never creates an optimizer, scheduler or training state.

## Preprocessing

Both pipelines are built from `TransformSpecification`
([src/datasets/transforms.py:64](../src/datasets/transforms.py)).

### Training — `build_train_transform` ([:157](../src/datasets/transforms.py))

| Order | Transform | Configured by |
| --- | --- | --- |
| 1 | `Resize(256)` | `dataset.resize_size` |
| 2 | `RandomResizedCrop(224, scale=[0.7,1.0], ratio=[0.75,1.3333])` | `dataset.augmentation.random_resized_crop_*` |
| 3 | `RandomHorizontalFlip(p=0.5)` | `horizontal_flip_probability` |
| 4 | `RandomRotation(15.0)` | `rotation_degrees` |
| 5 | `ColorJitter(0.2, 0.2, 0.2, 0.05)` | `color_jitter_*` |
| 6 | `ToTensor()` | — |
| 7 | `Normalize(mean, std)` | `dataset.normalization.*` |
| 8 | `RandomErasing(p=0.25)` | `random_erasing_probability` |

`RandomErasing` operates on tensors, hence its position after `Normalize`
([src/datasets/transforms.py:161](../src/datasets/transforms.py)).

### Evaluation / inference — `build_eval_transform` ([:190](../src/datasets/transforms.py))

| Order | Transform |
| --- | --- |
| 1 | `Resize(256)` |
| 2 | `CenterCrop(224)` |
| 3 | `ToTensor()` |
| 4 | `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` |

Deterministic by construction — separate function, no flag toggling. The API
imports this same function ([src/api/inference.py:153](../src/api/inference.py)),
so served preprocessing is identical to evaluation preprocessing.

### Validation of transform settings

`resize_size >= image_size` ([:86](../src/datasets/transforms.py)); mean/std must
have 3 entries with positive std ([:106](../src/datasets/transforms.py));
probabilities within `[0,1]`; `color_jitter_hue <= 0.5`
([:148](../src/datasets/transforms.py)).

## Inference

| Path | Mode | Grad context | Precision | Source |
| --- | --- | --- | --- | --- |
| Training validation | `model.eval()` | `torch.no_grad()` | autocast per `training.amp` | [src/training/engine.py:120](../src/training/engine.py) |
| Evaluation | `model.eval()` | `torch.no_grad()` | **fp32, no autocast** | [src/evaluation/inference.py:139](../src/evaluation/inference.py) |
| API | `model.eval()` | `torch.inference_mode()` | **fp32, no autocast** | [src/api/inference.py:191](../src/api/inference.py) |
| Structural verification | `model.eval()` | `torch.inference_mode()` | fp32 | [src/verification.py:114](../src/verification.py) |

`EVALUATION_PRECISION = "fp32"` ([src/evaluation/inference.py:32](../src/evaluation/inference.py))
and `INFERENCE_PRECISION = "fp32"` ([src/api/inference.py:39](../src/api/inference.py)).
The reason recorded in both docstrings: calibration and curve metrics read
probability values directly.

**Test-time augmentation: Not Found.** No TTA code exists.

## Postprocessing

| Step | Implementation | Source |
| --- | --- | --- |
| Evaluation probabilities | `torch.softmax(logits.to(float64), dim=1)` → NumPy | [src/evaluation/inference.py:288](../src/evaluation/inference.py) |
| API probabilities | `torch.softmax(logits.float(), dim=1)` | [src/api/inference.py:193](../src/api/inference.py) |
| API ranking | `scores.topk(self._top_k, dim=1)` | [src/api/inference.py:201](../src/api/inference.py) |
| API `top_k` clamp | `min(top_k, len(class_names))` | [src/api/inference.py:121](../src/api/inference.py) |

Evaluation uses float64 softmax so the probability-normalisation check is not
limited by float32 rounding.

## Inputs and outputs

| Interface | Input | Output |
| --- | --- | --- |
| `DinoV2Classifier.forward` | `[B,3,224,224]` float tensor | `[B,num_classes]` logits |
| `DinoV2Classifier.forward_features` | same | `[B,768]` embeddings |
| `InferenceEngine.predict` | `Sequence[(filename, PIL.Image)]` | `BatchPrediction` |
| `POST /predict` | multipart image | `PredictionResponse` |

`decode_image` ([src/api/inference.py:285](../src/api/inference.py)) converts any
decodable upload to `RGB` (`IMAGE_MODE`), so RGBA and greyscale inputs are
accepted.

## Class vocabulary

| Context | Source of classes | Reference |
| --- | --- | --- |
| Training | `ImageFolder` directory names, cross-checked across splits | [src/datasets/loaders.py:239](../src/datasets/loaders.py) |
| Evaluation | Dataset, then reconciled against the checkpoint | [src/training/checkpoints.py:187](../src/training/checkpoints.py) |
| API | **Checkpoint only** — no dataset required | [src/api/inference.py:145](../src/api/inference.py) |

`model.num_classes` in config is a fallback used only by standalone
`python -m src.model`; training, evaluation and the API all override it
([configs/config.yaml:40](../configs/config.yaml),
[src/train.py](../src/train.py), [src/evaluate.py:438](../src/evaluate.py)).

`_shared_class_to_idx` raises `DatasetValidationError` if any split disagrees on
the mapping ([src/datasets/loaders.py:239](../src/datasets/loaders.py)).

## Checkpoint format

Written by `save_checkpoint` ([src/training/checkpoints.py:115](../src/training/checkpoints.py)).
All values are tensors or primitives, so checkpoints load under
`weights_only=True` ([:176](../src/training/checkpoints.py)).

| Key | Content |
| --- | --- |
| `model` | `model.state_dict()` |
| `optimizer` | `optimizer.state_dict()` |
| `scheduler` | `scheduler.state_dict()` |
| `scaler` | `scaler.state_dict()` |
| `epoch` | int |
| `metrics` | list of history rows |
| `config` | `config.as_dict()` |
| `class_to_idx` | dict[str,int] |
| `best_value` | float |
| `epochs_without_improvement` | int |

Files: `checkpoints/best_model.pt`, `checkpoints/last_model.pt`
([configs/config.yaml](../configs/config.yaml)).

### Readers

| Function | Restores | Vocabulary check | Source |
| --- | --- | --- | --- |
| `load_checkpoint` | model + optimizer + scheduler + scaler + `ResumeState` | required | [:238](../src/training/checkpoints.py) |
| `load_model_checkpoint` | model only → `CheckpointMetadata` | required | [:198](../src/training/checkpoints.py) |
| `read_checkpoint` | raw `CheckpointContents` (config, classes, state) | **skipped** (`None`) | [:188](../src/training/checkpoints.py) |

`_read_payload` raises `CheckpointError` when the file is missing, unreadable, or
missing any of the ten keys ([:208](../src/training/checkpoints.py)).

## API model rebuild

`_rebuild_model` ([src/api/inference.py:308](../src/api/inference.py)) builds from
the **checkpoint's own stored config**, overriding `num_classes` and forcing
`pretrained: False` (weights arrive from the checkpoint), then
`load_state_dict`, `to(device)`, `eval()`.

Consequence: the API requires the `torch.hub` DINOv2 repo cache to construct the
architecture, but not the pretrained `.pth` and not `data/`.

## Versioning

| Identifier | Value / source | Where surfaced |
| --- | --- | --- |
| `model_version` | `project.version` = `1.0.0` | `/metadata`, `/health`, every prediction |
| Checkpoint SHA-256 | computed by `fingerprint_file` | `/metadata`, `evaluation.json` |
| Checkpoint epoch | `epoch` key | `/metadata`, `evaluation.json` |
| `best_value` | monitored metric at selection | `/metadata` |
| Distribution version | `1.0.0` in `pyproject.toml` | **not surfaced at runtime** |

There is no automatic link between `project.version` and the checkpoint contents;
changing weights without editing config leaves `model_version` unchanged.

## Artifacts

| Artifact | Producer | Source |
| --- | --- | --- |
| `results/model_summary.txt` | `python -m src.model` | [src/verification.py:162](../src/verification.py) |
| `results/model_verification.json` | `python -m src.model` | [src/verification.py:165](../src/verification.py) |
| `checkpoints/best_model.pt`, `last_model.pt` | training | [src/train.py](../src/train.py) |
| `results/history.csv` | training | [src/training/metrics.py:86](../src/training/metrics.py) |
| `results/loss_curve.png`, `accuracy_curve.png` | training | [src/visualization/plots.py](../src/visualization/plots.py) |
| Evaluation artifacts (9 files) | evaluation | see [METRICS.md](METRICS.md#exported-artifacts) |
