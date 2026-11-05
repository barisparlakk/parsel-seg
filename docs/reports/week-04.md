# Week 4 - Architecture comparison

## Done
- `segformer`, `upernet` registered in `parsel/models/factory.py`; configs `segformer_b2.yaml`,
  `upernet_convnext_t.yaml`, `deeplabv3plus_r50.yaml` next to the Week 3 `unet_r34.yaml`.
- `parsel/utils/metrics.py`: `BoundaryIoU`, per-class IoU inside a 3 px band around each
  region's contour (ground truth and prediction), averaged over the classes present.
- `scripts/evaluate.py` now also writes boundary IoU, parameter count and the confusion matrix
  to `eval_<split>.json`.
- `scripts/compare_runs.py`: one table of all evaluated runs, a per-class table and a chart;
  it also reports the share of predictions that land on classes absent from the ground truth.
- `notebooks/colab_train.ipynb`: `CONFIG`, `EXTRA` (config overrides) and `TAG` select the
  experiment; a last cell runs the comparison.
- 11 more tests (boundary IoU, new architectures, every train config builds, comparison
  script); 73 in total.

## Setup
All four models: same data and split (7693 train / 1443 val / 1460 test tiles), 384 px random
crops, batch 8, AdamW, 15 epochs with one warmup epoch and cosine decay, CE + Dice with
median-frequency weights (power 0.5), seed 42, mixed precision, one Colab L4. Early stopping
after 5 epochs without a val mIoU gain of 0.001. Test numbers use the checkpoint with the best
val mIoU.

The learning rate differs by family: 3e-4 for the two CNN encoders (as in the Week 3 baseline)
and 1e-4 for SegFormer and ConvNeXt, which usually want a smaller one. None of them was tuned,
so "architecture" and "learning rate" are confounded in this comparison.

## Results (test, 11 held-out regions)

| Model | Params (M) | mIoU | mF1 | Boundary mIoU | Pixel acc | Best val mIoU | Epochs run | Best epoch | Train min |
|---|---|---|---|---|---|---|---|---|---|
| SegFormer-B2 | 24.7 | **0.6237** | **0.7571** | 0.1962 | 0.8037 | 0.6834 | 14 | 8 | 39 |
| UPerNet + ConvNeXt-T | 37.0 | 0.6168 | 0.7519 | **0.2131** | **0.8081** | 0.6782 | 15 | 9 | 44 |
| DeepLabV3+ R50 | 26.7 | 0.5917 | 0.7325 | 0.1736 | 0.7840 | 0.6287 | 15 | 12 | 33 |
| U-Net R34 | 24.4 | 0.5789 | 0.7190 | 0.1881 | 0.7850 | 0.6502 | 15 | 14 | 30 |

| Class | SegFormer-B2 | UPerNet | DeepLabV3+ | U-Net |
|---|---|---|---|---|
| background | 0.5190 | **0.5384** | 0.4778 | 0.4833 |
| buildings | 0.7581 | **0.7746** | 0.7324 | 0.7332 |
| vacant land | **0.3282** | 0.3194 | 0.3147 | 0.2570 |
| agriculture | **0.7137** | 0.7124 | 0.6726 | 0.6726 |
| transport | 0.5883 | **0.6087** | 0.5754 | 0.5798 |
| water | **0.6955** | 0.6049 | 0.6176 | 0.5755 |
| vegetation | **0.7630** | 0.7595 | 0.7516 | 0.7509 |

Construction and energy have no ground truth yet. None of the four models predicts a single
pixel of either (0.00% of predictions).

## Observations
- SegFormer-B2 and UPerNet + ConvNeXt-T are about 4 to 4.5 points above the U-Net and 2.5 to
  3 above DeepLabV3+. The CNN-only models are close to each other. The gap between the two
  leaders is 0.007 mIoU, which is within what one seed and 11 regions can resolve.
- Val and test disagree on the order of the two weakest models: DeepLabV3+ has the lower best
  val mIoU (0.629 vs 0.650) but the higher test mIoU (0.592 vs 0.579). Differences below about
  0.02 between models are therefore not trustworthy on this split.
- Vacant land improved for every model other than the U-Net (0.257 -> 0.315 to 0.328), the
  first movement for that class after longer training did nothing. It is still the weakest class.
- Water is where SegFormer leads (0.696 vs 0.605 for UPerNet), but water is about 2% of test
  pixels, so a few regions decide it.
- Boundary IoU is low for all models (0.17 to 0.21) against mIoU 0.58 to 0.62, and the spread
  between models is small. UPerNet is best at edges; its multi-scale decoder may help. Edge
  quality is the clearest weakness shared by all of them.
- UPerNet's val mIoU dipped around epochs 3 to 4 at a high learning rate and recovered as the
  rate decayed; best epoch was 9 of 15. Early stopping did not trigger for it.
- Cost on an L4: 165 to 176 s per epoch for SegFormer and UPerNet, 122 s for the U-Net.

## Decision
SegFormer-B2 is the base for the Week 4 ablations and for Week 5. It ties UPerNet on test, has
fewer parameters (24.7M vs 37.0M) and trains faster. UPerNet stays as the comparison point for
edges.

## Still to do this week (not run yet)
Ablations on SegFormer-B2, one change at a time, same hardware. The baseline for each is the
`segformer_b2_colab` run above (CE + Dice, weights with power 0.5, 384 px crops).

| Question | `TAG` | `EXTRA` |
|---|---|---|
| CE only | `_ce_only` | `--set loss.dice_weight=0` |
| Focal + Dice | `_focal_dice` | `--set loss.type=focal_dice` |
| No class weights | `_no_weights` | `--set loss.class_weights=null` |
| Raw class weights | `_weights_full` | `--set loss.weight_power=1.0` |
| Full 512 px tiles as training input | `_crop512` | `--set data.crop_size=512` |

`focal_dice` uses focal loss (gamma 2, same class weights, same normalisation as CE) in place
of the CE term. `evaluate.py` now also measures forward-pass speed (512 px, batch 8, fp32) and
`compare_runs.py` lists it, so the inference-time column needs every run re-evaluated on the
same GPU.

### 768 px input: deferred to Week 6
The plan compares 512 and 768 px input. On OpenEarthMap that cannot be done cleanly with tiles:
72% of the images are 1024 px, so 768 px tiles start at 0 and 256 and overlap by two thirds
(each pixel is seen up to 4 times, centres more than edges), and the 49 images of about
440 x 406 px would be more than half padding and dropped. Train and test pixels would no
longer match the 512 px runs. The fair version evaluates whole scenes with a sliding window,
which Week 6 builds for inference anyway: then 512, 768 and full-scene windows can be scored
on identical pixels.
