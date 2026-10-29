# Week 3 - Baseline U-Net

## Done
- `parsel/models/factory.py`: `build_model(cfg, num_classes)` for segmentation-models-pytorch
  architectures with timm/ImageNet encoders.
- `parsel/losses/losses.py`: weighted cross-entropy + Dice (`CombinedLoss`), ignore index 255.
- `parsel/utils/metrics.py`: confusion-matrix metrics (per-class IoU/F1, mIoU, pixel accuracy).
  mIoU is averaged over the classes present in the ground truth only.
- `parsel/train/trainer.py`: epoch loop, cosine schedule with warmup, gradient clipping,
  early stopping, `best.pt` / `last.pt` / `history.json`, resume.
- `scripts/train.py`: `--config`, `--name`, `--epochs`, `--overfit N`, `--set KEY=VALUE`,
  `--resume`. `parsel/utils/config.py` applies the overrides.
- `scripts/evaluate.py`: per-class metrics, confusion matrix and worst-tile figures for any
  split.
- `notebooks/colab_train.ipynb`: the same training on a Colab GPU, with runs stored on Drive.
- 32 more tests (metrics, losses, models, trainer, config); 62 in total.

## Baseline
U-Net, ResNet34 encoder (ImageNet), 24.4M parameters. Random 384 px crops, batch 8, AdamW
lr 3e-4, 15 epochs, CE + Dice with median-frequency class weights (power 0.5), seed 42.
Trained on the 7693 train tiles, evaluated on the 1460 test tiles (11 regions that appear in
neither train nor val). Checkpoint: epoch 14, the last one.

| Metric | Test |
|---|---|
| mIoU | 0.5789 |
| mF1 | 0.7190 |
| Pixel accuracy | 0.7850 |

| Class | IoU | F1 | GT pixels (M) |
|---|---|---|---|
| vegetation | 0.7509 | 0.8577 | 125.6 |
| buildings | 0.7332 | 0.8461 | 74.8 |
| agriculture | 0.6726 | 0.8042 | 43.5 |
| transport | 0.5798 | 0.7340 | 37.1 |
| water | 0.5755 | 0.7306 | 8.7 |
| background | 0.4833 | 0.6517 | 84.0 |
| vacant land | 0.2570 | 0.4089 | 3.9 |
| construction | - | - | 0 |
| energy | - | - | 0 |

Construction and energy have no labels in OpenEarthMap, so they are not scored and have
weight 0 in the loss. They arrive with the custom data in Week 5.

## Hardware
| Device | Throughput | Epoch time |
|---|---|---|
| Apple MPS (Mac) | 5.5-7.5 img/s | 17-24 min |
| Colab L4 (CUDA, AMP) | 63-78 img/s | about 2 min |

Mixed precision gave no speedup on MPS, so `amp: auto` enables it for CUDA only. The Colab
run took roughly 30 minutes, the same 15 epochs would have taken about 5 hours on the Mac.
A Mac run was stopped after 6 epochs (val mIoU 0.563) once Colab was working. Different
hardware gives slightly different numbers, so every Week 4 comparison runs on one device.

## Observations
- Vegetation and buildings are the strongest classes; vacant land is by far the weakest
  (IoU 0.26) and is only 1% of test pixels. The class weight for it is already 6.6.
- Background ("developed", paved yards and plazas) scores 0.48: it borders buildings and
  roads and has no clear visual definition. This was the least certain mapping in Week 2.
- The stopped Mac run at epoch 5 scored 0.563 on val and the final Colab model 0.579 on
  test: ten more epochs added little. Val and test regions differ, so this is a rough
  comparison, but it suggests the baseline is near its plateau and Week 4 gains should come
  from the model, loss and data rather than longer training.
- Not yet checked: whether the model predicts construction or energy anywhere. That needs the
  test confusion matrix (`runs/<run>/eval_test.json`).

## Bugs found and fixed
- `CrossEntropyLoss` returns NaN when every pixel of a batch is ignored; `CombinedLoss` now
  skips CE in that case.
- First-epoch "best" check and `min_delta` interacted badly; fixed in the trainer.
- Early stopping cut the overfit sanity run short; it is disabled in `--overfit` mode. With
  that fixed, 150 epochs on 16 tiles reached mIoU 0.894, so the pipeline can fit.
- YAML parses `3e-4` as a string; `config.py` now converts overrides itself.
- DeepLabV3+ fails with batch size 1 in BatchNorm; the test uses batch 2.

## Next (Week 4)
SegFormer-B2, UperNet + ConvNeXt-T and DeepLabV3+ against this baseline; loss, class-weight and
input-size ablations; a `compare_runs.py` script; boundary IoU.
