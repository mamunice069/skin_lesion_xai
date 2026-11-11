# Report notes

Running notes for the final report: methodological decisions, deviations from the proposal,
and known limitations. Numbers here come from script outputs (file named next to each).

## Data

- **HAM10000:** 10,015 images and 7,470 lesions (`HAM10000_metadata.csv`, Harvard Dataverse).
  - Images: original 600x450 JPGs (`ISIC/diagnosis_confirm_type/`).
  - Lesion masks: Tschandl segmentations, one per image, 600x450. All are present
    (checked by `scripts/prepare_ham10000.py`); the sampled masks are strictly 0/255 (`tests/test_datasets.py`).
- **ISIC 2020:** the Kaggle copy "ISIC 2020 JPG 256x256 RESIZED", 33,126 images, 2,056 patients, 584 melanoma (1.76%).
  - The originals have varying aspect ratios and were resized to 256x256 squares without cropping, so lesion shapes are stretched.
  - Only the public training set is used, because test labels are not public. It is split into train/val/test ourselves.
  - There are no lesion masks, so the XAI evaluation on ISIC 2020 is qualitative only.
- **Splits:** 70/15/15, stratified by class.
  - HAM10000 is grouped by `lesion_id` and ISIC 2020 by `patient_id`, so no lesion or patient appears in two splits.
  - Counts: `data/splits/ham10000_summary.csv`, `data/splits/isic2020_summary.csv`.
  - Contrast with related work (corrected 2026-11-03 after reading the PDFs' text): GC-ViT-E (§3.1) and Mahesh et al. (§4.1) describe class-stratified 70/15/15 splits and do not state whether images of one lesion are kept in one split. The earlier wording "split by image" went beyond the papers; for Mahesh et al. it rested on the published code (project plan: "Repo ≠ paper"), not on the paper. GC-ViT-E names patient-level leakage as a known risk in its related work (§2).

## Hair removal (DullRazor)

- **Method:** a simplified OpenCV version of DullRazor (Lee et al., 1997), in `src/skin_lesion_xai/data/dullrazor.py`:
  1. Black-hat morphology (cross kernel).
  2. Threshold.
  3. Thin-structure check: keep a connected region only if it is long, i.e. bounding-box diagonal ≥ 8% of the
     image's shorter side, and thin, i.e. diagonal² / area ≥ 8.
  4. Widen the mask by 1 px.
  5. Telea inpainting.
- **Parameters** (in `configs/data/*.yaml`):
  - HAM10000: kernel 17, threshold 12.
  - ISIC 2020: kernel 11, threshold 12.
- **Tuning:** done visually on the first 50 images of each split CSV (`data/processed/<dataset>_dullrazor/_compare_old_new_*.png`).
  - Without the thin-structure check (threshold 10), dots, globules and pigment network inside lesions were marked as hair and smoothed away.
  - Mean marked area on the HAM10000 images fell from 7.83% / 7.38% to 2.69% / 3.03% (images 1–25 / 26–50).
- **Use:** training uses the original images by default. The hair-removed copies are used only for the
  with/without hair-removal ablation (Milestone 3).

### Known failure cases and limitations
- **Lesion border ring kept as hair:** a dark, thin, ring-shaped lesion border passes the long/thin test and is inpainted.
  - Example: ISIC_0025936, 22.4% of the image marked.
  - This touches the region the Grad-CAM/IoU evaluation is measured against, so the IoU of hair-removed models must be read with this in mind.
- **Connected pigment network kept as hair:** a large, sparse web of network lines passes the long/thin test.
  - Examples: ISIC_0022129 (21.6% marked), ISIC_0027615 (27.9%).
- **Light hair missed:** blond or white hair has too little contrast for black-hat detection.
  - Examples: ISIC_0074542, ISIC_0092726 (ISIC 2020), and the white hairs in ISIC_0027850 (HAM10000).
- **Dense thick hair:** inpainting leaves streaks when hair covers a large part of the image.
  - Example: ISIC_0068279, about 41% marked.
- **Dots touching a hair** join the hair's connected region and are removed with it (`tests/test_dullrazor.py`).
- **Scope:** these cases come from a visual check of 100 images, not a full audit. How often they occur over the whole dataset is not measured.

## Baselines on HAM10000 (Milestone 2)

- **Recipe (all baselines):** `configs/train/default.yaml` + `configs/train/local_4gb.yaml`, from GC-ViT-E §3.4 and ViT App. B.1.1.
  - Optimiser: AdamW, backbone lr 1e-4, head lr 1e-3, weight decay 0.05.
  - Schedule: 3 warmup epochs, then cosine decay, 30 epochs at most.
  - AMP and gradient clipping at 1.0.
  - EMA 0.999; checkpoints are selected on the EMA weights.
  - Early stopping: patience 7 on val macro-F1.
  - Loss: label smoothing 0.1 and class weights (1/freq)^0.5.
  - Batch: 8 × 4 accumulation steps (effective 32) on the local RTX 3050 4 GB.
- **ViT-B/16 deviation:** with backbone lr 1e-4, ViT-B peaked during warmup (val macro-F1 0.7594 at epoch 2, run `ham10000_vit_b16`).
  - After the lr reached 1e-4, training loss nearly stopped falling and it early-stopped at epoch 9.
  - It was rerun with backbone lr 3e-5 (`ham10000_vit_b16_lr3e-5`, `--backbone-lr 3e-5`); everything else was unchanged. This run is the ViT-B baseline.
  - The lr was chosen after seeing one run's val results; the test set was not used. So ViT-B had one tuning step the other baselines did not.
  - The lr 1e-4 run is kept as a validation-only record.
- **Interrupted run:** the first ConvNeXt-T run was killed by the system after 18 epochs (low RAM).
  - It was restarted from scratch (`outputs/runs/ham10000_convnext_t_killed`, kept for the record only).
  - Resume support (`--resume`) was added afterwards.

### Validation (best epoch; `outputs/runs/<run>/train_summary.json`)
| Model | Run | Best val macro-F1 | Best epoch | Epochs run | Train time |
|---|---|---|---|---|---|
| ConvNeXt-T | ham10000_convnext_t | 0.8384 | 16 | 23 | 59.8 min |
| Swin-T | ham10000_swin_t | 0.8255 | 6 | 13 | 42.2 min |
| ViT-B/16 (lr 3e-5) | ham10000_vit_b16_lr3e-5 | 0.8201 | 16 | 23 | 61.5 min |
| EfficientNetV2-S | ham10000_effnetv2_s | 0.7850 | 12 | 19 | 111.6 min |
| ViT-B/16 (lr 1e-4, val only) | ham10000_vit_b16 | 0.7594 | 2 | 9 | 25.8 min |
| ResNet-50 | ham10000_resnet50 | 0.7172 | 23 | 30 | ~55 min |

Train times come from `outputs/logs/baselines_ham10000_memory.log`; the ResNet-50 time is from its train.log timestamps.

### Test set (n = 1,503; `outputs/runs/<run>/eval/metrics.json`; 95% CI from 1,000 bootstrap resamples)
| Model | Macro-F1 [95% CI] | Bal. acc | Accuracy | AUC (OvR macro) | MCC | Kappa | Mel. sensitivity |
|---|---|---|---|---|---|---|---|
| ConvNeXt-T | 0.8118 [0.7775–0.8424] | 0.8362 | 0.8729 | 0.9595 | 0.7599 | 0.7596 | 0.679 |
| ViT-B/16 (lr 3e-5) | 0.8048 [0.7720–0.8346] | 0.8422 | 0.8689 | 0.9581 | 0.7541 | 0.7538 | 0.673 |
| Swin-T | 0.7710 [0.7304–0.8047] | 0.7837 | 0.8530 | 0.9592 | 0.7259 | 0.7253 | 0.655 |
| EfficientNetV2-S | 0.7563 [0.7152–0.7909] | 0.7927 | 0.8516 | 0.9609 | 0.7184 | 0.7180 | 0.571 |
| ResNet-50 | 0.7111 [0.6673–0.7509] | 0.7577 | 0.8150 | 0.9548 | 0.6568 | 0.6561 | 0.548 |

The macro-F1 ranking on test matches the val ranking, except Swin-T.

### Observations
- **Separation:** the macro-F1 CIs of ConvNeXt-T, ViT-B and Swin-T overlap heavily. ConvNeXt-T and ViT-B do not overlap with ResNet-50.
  - Overlapping CIs are not a formal test. A paired test on `eval/predictions.csv` is planned for Milestone 4.
- **AUC barely separates the models** (0.955–0.961). Macro-F1, balanced accuracy and melanoma sensitivity do separate them.
- **Melanoma** (168 test images) is the weakest of the clinically important classes for every model (sensitivity 0.548–0.679), followed by bkl (0.636–0.739).
- **Rare classes are too small to compare:** df has 17 test images and vasc 21, so their per-class values rest on a handful of images.
- **Swin-T's val→test drop is the largest** (0.8255 → 0.7710). Its checkpoint is from epoch 6, a single peak on a noisy val curve, so patience 7 on val macro-F1 may be tight.
- **Val score mismatch for EfficientNetV2-S:** evaluate.py gives val macro-F1 0.7842 versus 0.7850 during training, from the same `best.pt`.
  - Likely cause: eval batch 16 vs 8 plus fp16/cuDNN non-determinism. The other four runs match exactly.

### Input normalisation mismatch (found in Milestone 3)
- **The mismatch:** the data pipeline normalises every image with ImageNet mean/std.
  - `tf_efficientnetv2_s.in21k_ft_in1k` and `vit_base_patch16_224.augreg2_in21k_ft_in1k` were pretrained with mean = std = 0.5 (timm `pretrained_cfg`).
  - The ResNet-50, ConvNeXt-T and Swin-T checkpoints use ImageNet statistics.
  - So the EfficientNetV2-S and ViT-B/16 baselines above were fine-tuned on inputs shifted and scaled from their pretraining. The results stand as recorded, but these two baselines may be understated.
- **Fix:** `match_pretrained_norm: true` in a model config adds `InputRenorm` (`models/backbones.py`), which maps the input to the checkpoint's own statistics.
  - It is opt-in; runs without it reload and behave exactly as trained.
  - The hybrid uses it, so its EfficientNetV2 branch and Swin branch each get their own normalisation.
- **Retrains with the fix:** `ham10000_effnetv2_s_pnorm` and `ham10000_vit_b16_pnorm_lr3e-5` (configs `effnetv2_s_pnorm.yaml`, `vit_b16_pnorm.yaml`).
  - The hybrid-vs-single-branch comparison uses the corrected EfficientNetV2-S.

## Hybrid on HAM10000 (Milestone 3)

- **Model:** `configs/model/hybrid.yaml`, 55.61M parameters.
  - EfficientNetV2-S branch and Swin-T branch, each with its own input normalisation.
  - Bidirectional cross-attention fusion: d = 384, 2 layers, 8 heads.
  - Same recipe and `local_4gb` profile as the baselines.
- **Concat ablation:** `configs/model/hybrid_concat.yaml`, 47.71M parameters.
  - The same two branches, but each 7×7 grid is average-pooled and the two vectors are concatenated (1280 + 768); there is no attention between branches.
  - Pooled-feature concatenation of two image branches, with no attention between them. (Corrected 2026-11-03: this is not "the fusion of Mahesh et al. §3.5". Their image branch is sequential, EfficientNetV2-B0 features passed through a Transformer encoder (§3.3), and their §3.5 concatenates the image features with patient-metadata features from an MLP (§3.4).)
- **Runs:** `ham10000_hybrid` and `ham10000_hybrid_concat`, original images (no hair removal).

### Validation (`outputs/runs/<run>/train_summary.json`)
| Model | Run | Best val macro-F1 | Best epoch | Epochs run | Epoch time (sum) |
|---|---|---|---|---|---|
| Hybrid (cross-attention) | ham10000_hybrid | 0.8389 | 10 | 17 | 235 min |
| Hybrid (concat fusion) | ham10000_hybrid_concat | 0.8417 | 6 | 13 | 505 min (89 min without epoch 9) |

- **Early stop:** both runs stopped after 7 epochs without improvement (cross-attention after epoch 17, concat after epoch 13).
- **Overfitting:**
  - Cross-attention: after epoch 10, train loss kept falling (1.21 → 1.05) while val loss rose (1.42 → 1.46).
  - Concat: after epoch 6, train loss fell (1.25 → 1.08) and val loss rose (1.42 → 1.43).
- **Cross-attention vs concat:** concat is 0.003 higher on val (0.8417 vs 0.8389).
  - These are single runs; a gap this small does not show that either fusion is better.
  - So far, cross-attention fusion has not beaten the simpler concat fusion on val macro-F1. Test-set results and a paired test (Milestone 4) are needed before drawing a conclusion.
- **Against the baselines:** both hybrids are level with ConvNeXt-T (0.8384) on val. They are above Swin-T (0.8255), ViT-B/16 (0.8201) and EfficientNetV2-S (0.7850).
  - The EfficientNetV2-S figure is from the run without the normalisation fix. The `_pnorm` retrain scored 0.7777; see the Milestone 3 summary below.
  - These are single runs compared on val only; the test set has not been evaluated yet.

### Run history (`train.log`, `outputs/logs/m3_train_ham10000_memory.log`)
- **RAM paging:** epochs 2–4 took 2,147 s, 1,321 s and 3,817 s; the other epochs took 286–706 s.
  - Cause: low free RAM while other programs (Chrome) were open. The training process's working set was pushed out to the page file, and GPU utilisation fell to about 11%.
  - The run sped back up to about 520 s per epoch after Chrome was closed.
- **Restart:** the process was stopped by hand after epoch 12 and continued with `--resume` from `last.pt` (20:43).
  - The resume restores model, EMA, optimiser, scheduler, AMP scaler and RNG state (equivalence checked in `tests/test_trainer.py`).
  - The work lost was the partial epoch 13, which was then repeated.
- **Wall clock:** 17:15–21:16 (4.0 h). Epoch times depend on machine load, so they are not a clean speed comparison with the baselines.
- **Concat run, paging:** epoch 9 took 24,948 s (6.9 h); the other epochs took 276–535 s.
  - Cause: the same RAM paging, with Chrome/Edge open overnight.
  - It recovered without a restart, and the run finished in one process (21:16 – 05:41, 505.8 min).
  - The paging changes only wall time, not the training computation, so the result stands.

## Milestone 3 summary: all HAM10000 runs, validation only

- **Sources:**
  - Best val macro-F1, best epoch and epochs run come from `outputs/runs/<run>/train_summary.json`.
  - Val melanoma sensitivity comes from the `val` block of `outputs/runs/<run>/eval/metrics.json` for the baselines.
  - For the M3 runs it comes from `outputs/runs/<run>/eval_val/metrics.json` (`evaluate.py --val-only`, which does not load the test split; log: `outputs/logs/m3_eval_val_memory.log`).
- **Melanoma sensitivity:** n = 166 val melanomas. It is measured at the best (EMA) checkpoint.
  - For all six M3 runs, the macro-F1 from `--val-only` matches the training value to 4 decimals.
  - The EfficientNetV2-S baseline's val macro-F1 from `evaluate.py` is 0.7842 (vs 0.7850 in training; see Baselines). Its melanoma sensitivity comes from that evaluation.
- **Loop log:** `outputs/logs/m3_train_ham10000_memory.log` records all six M3 runs as OK; the last finished at 2026-11-01 09:59.

| Group | Model | Run | Best val macro-F1 | Val mel. sens. | Best epoch | Epochs run |
|---|---|---|---|---|---|---|
| Baseline | ResNet-50 | ham10000_resnet50 | 0.7172 | 0.602 | 23 | 30 |
| Baseline | ConvNeXt-T | ham10000_convnext_t | 0.8384 | 0.735 | 16 | 23 |
| Baseline | ViT-B/16 (lr 3e-5) | ham10000_vit_b16_lr3e-5 | 0.8201 | 0.687 | 16 | 23 |
| Baseline | Swin-T | ham10000_swin_t | 0.8255 | 0.699 | 6 | 13 |
| Baseline | EfficientNetV2-S | ham10000_effnetv2_s | 0.7850 | 0.620 | 12 | 19 |
| M3 fusion | Hybrid, cross-attention | ham10000_hybrid | 0.8389 | 0.723 | 10 | 17 |
| M3 fusion | Hybrid, concat | ham10000_hybrid_concat | 0.8417 | 0.669 | 6 | 13 |
| M3 norm fix | EfficientNetV2-S, pretrained norm | ham10000_effnetv2_s_pnorm | 0.7777 | 0.566 | 9 | 16 |
| M3 norm fix | ViT-B/16 (lr 3e-5), pretrained norm | ham10000_vit_b16_pnorm_lr3e-5 | 0.8311 | 0.699 | 8 | 15 |
| M3 hair | ConvNeXt-T, hair removed | ham10000_convnext_t_hair | 0.8527 | 0.717 | 19 | 26 |
| M3 hair | Hybrid (cross-attention), hair removed | ham10000_hybrid_hair | 0.8442 | 0.753 | 11 | 18 |

The ViT-B/16 lr 1e-4 run (val 0.7594) stays a validation-only record and is not listed.

### Observations (val, single runs, so differences of about 0.01 are within run-to-run noise)
- **Fusion:** cross-attention (0.8389) and concat (0.8417) are level. Neither clearly beats the best single backbone, ConvNeXt-T (0.8384).
- **Normalisation fix:** mixed.
  - ViT-B/16: +0.011 (0.8201 → 0.8311).
  - EfficientNetV2-S: −0.007 (0.7850 → 0.7777).
  - So the original mismatch did not clearly understate EfficientNetV2-S.
- **Hair removal:** higher val macro-F1 for both models.
  - ConvNeXt-T: +0.014 (0.8384 → 0.8527).
  - Hybrid: +0.005 (0.8389 → 0.8442).
  - Hair removal also changes the val images, so this compares each model on its own preprocessing. The DullRazor failure cases (see Hair removal) still apply to the XAI comparison.
- **Melanoma sensitivity (val):** the hair-removed hybrid is highest (0.753, 125/166), followed by ConvNeXt-T (0.735, 122/166) and the cross-attention hybrid (0.723, 120/166).
  - Concat fusion has the higher macro-F1 but lower melanoma sensitivity than cross-attention: 0.669 (111/166) vs 0.723.
  - The normalisation fix lowered EfficientNetV2-S melanoma sensitivity: 0.620 → 0.566.
  - Hair removal: hybrid +0.030 (0.723 → 0.753); ConvNeXt-T −0.018 (0.735 → 0.717).
  - With 166 melanomas, one image is 0.006, so a 0.03 gap is 5 images. These single-run gaps need the Milestone 4 paired test before any claim.
- **Hybrid vs single branches:** the hybrid (0.8389) is above both of its own branches trained alone.
  - Local branch: EfficientNetV2-S, 0.7777 with the norm fix.
  - Global branch: Swin-T, 0.8255.

## XAI localisation on HAM10000 val (Milestone 3b)

- **Sources:** `outputs/runs/<run>/xai_val/summary.json` (trained, best.pt) and `xai_val_random/summary.json` (same architecture, random weights).
  - Made by `scripts/explain.py` and `scripts/eval_xai.py` with `configs/xai/default.yaml`; log: `outputs/logs/xai_val_ham10000_memory.log`.
  - All 22 steps exited OK. Every row covers all 1,503 val images, and no row has a NaN IoU.
  - **Rescored on 2026-11-01** (`outputs/logs/xai_rescore_ham10000_memory.log`, 22 OK) after a scoring fix; the maps themselves were not recomputed.
    - Before the fix, an all-zero map scored 0 on IoU, Dice, box IoU and pointing, but NaN on the two energy metrics, so it dropped out of their averages.
    - It now scores 0 on every metric, and `summary.json` reports each map's share of empty maps (`empty_share`).
    - The numbers below are the rescored ones; the change is described under Observations.
- **Protocol** (fixed before the runs):
  - Maps are upsampled to 224 px and scaled by their max.
  - IoU / Dice of the map binarised at 15% of its max (Selvaraju et al., Grad-CAM §4.1; Dice as WA-NET Eq. 16).
  - Box IoU of the largest region (Grad-CAM §4.1).
  - Pointing game (Grad-CAM §4.3).
  - Energy in mask: the share of the map's total value inside the lesion.
  - Grad-CAM maps are for the predicted class. Attention maps (rollout, cross-attention) are class-agnostic.
- **Lesion size:** the masks after the eval centre crop cover 41.8% of the image on average (median 37.7%). In 60 of 1,503 images the lesion fills ≥ 90% of the crop, and in 8 it fills all of it.
  - So a uniform map already scores IoU 0.418. A map highlighting the image centre, a centred Gaussian with std 0.25 × image size, scores IoU 0.487 and pointing 0.961.

### Reference maps (identical for every run, same images and masks)
| Map | IoU | Dice | Box IoU | Pointing | Energy in mask |
|---|---|---|---|---|---|
| Centred Gaussian | 0.487 | 0.615 | 0.550 | 0.961 | 0.603 |
| Uniform | 0.418 | 0.545 | 0.544 | — | 0.418 |

### Trained models vs random weights
| Model | Map | IoU | Dice | Box IoU | Pointing | Energy in mask | Random: IoU | Random: pointing | Random: energy |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | Grad-CAM | **0.533** | 0.680 | 0.599 | 0.968 | **0.711** | 0.418 | 0.282 | 0.398 |
| ConvNeXt-T | Grad-CAM | 0.311 | 0.450 | 0.325 | 0.910 | 0.521 | 0.246 | 0.536 | 0.395 |
| Swin-T | Grad-CAM | 0.453 | 0.584 | 0.551 | 0.504 | 0.530 | 0.336 | 0.348 | 0.331 |
| EfficientNetV2-S | Grad-CAM | 0.442 | 0.573 | 0.553 | 0.901 | 0.550 | 0.285 | 0.393 | 0.403 |
| ViT-B/16 (lr 3e-5) | Grad-CAM | 0.483 | 0.611 | 0.573 | 0.983 | 0.654 | 0.130 | 0.365 | 0.361 |
| ViT-B/16 (lr 3e-5) | attention rollout | 0.418 | 0.545 | 0.544 | 0.399 | 0.425 | 0.418 | 0.504 | 0.414 |
| Hybrid, cross-attention | Grad-CAM, EffNetV2 branch | 0.490 | 0.622 | 0.544 | 0.884 | 0.706 | 0.223 | 0.402 | 0.388 |
| Hybrid, cross-attention | Grad-CAM, Swin branch | 0.410 | 0.534 | 0.539 | 0.611 | 0.468 | 0.285 | 0.598 | 0.443 |
| Hybrid, cross-attention | Grad-CAM, fused tokens | 0.422 | 0.549 | 0.544 | 0.849 | 0.446 | 0.418 | 0.808 | 0.461 |
| Hybrid, cross-attention | cross-attn, EffNetV2 → Swin | 0.277 | 0.416 | 0.358 | 0.866 | 0.557 | 0.418 | 0.589 | 0.417 |
| Hybrid, cross-attention | cross-attn, Swin → EffNetV2 | 0.228 | 0.355 | 0.244 | 0.967 | 0.655 | 0.418 | 0.275 | 0.418 |
| Hybrid, concat | Grad-CAM, EffNetV2 branch | 0.363 | 0.492 | 0.528 | 0.485 | 0.424 | 0.341 | 0.703 | 0.562 |
| Hybrid, concat | Grad-CAM, Swin branch | 0.459 | 0.585 | 0.561 | 0.976 | 0.574 | 0.411 | 0.657 | 0.396 |
| Hybrid, concat | Grad-CAM, combined | 0.449 | 0.575 | 0.555 | 0.969 | 0.546 | 0.411 | 0.664 | 0.396 |
| EfficientNetV2-S, pretrained norm | Grad-CAM | 0.446 | 0.580 | 0.555 | 0.870 | 0.570 | 0.373 | 0.446 | 0.452 |
| ViT-B/16, pretrained norm | Grad-CAM | 0.496 | 0.622 | 0.597 | 0.974 | 0.644 | 0.060 | 0.140 | 0.128 |
| ViT-B/16, pretrained norm | attention rollout | 0.418 | 0.545 | 0.544 | 0.659 | 0.461 | 0.418 | 0.335 | 0.413 |
| ConvNeXt-T, hair removed | Grad-CAM | 0.324 | 0.466 | 0.336 | 0.907 | 0.551 | 0.247 | 0.552 | 0.396 |
| Hybrid, hair removed | Grad-CAM, EffNetV2 branch | 0.400 | 0.535 | 0.485 | 0.790 | 0.610 | 0.222 | 0.398 | 0.387 |
| Hybrid, hair removed | Grad-CAM, Swin branch | 0.412 | 0.539 | 0.537 | 0.563 | 0.459 | 0.287 | 0.603 | 0.446 |
| Hybrid, hair removed | Grad-CAM, fused tokens | 0.418 | 0.545 | 0.544 | 0.808 | 0.432 | 0.418 | 0.813 | 0.460 |
| Hybrid, hair removed | cross-attn, EffNetV2 → Swin | 0.301 | 0.445 | 0.380 | 0.818 | 0.590 | 0.418 | 0.599 | 0.417 |
| Hybrid, hair removed | cross-attn, Swin → EffNetV2 | 0.276 | 0.421 | 0.328 | 0.900 | 0.636 | 0.418 | 0.272 | 0.418 |

### Observations (val, single runs, no confidence intervals yet; `per_image.csv` allows a paired bootstrap in Milestone 4)
- **The centred Gaussian is hard to beat.**
  - On IoU, only ResNet-50 (0.533), ViT-B/16 with pretrained norm (0.496) and the hybrid's EfficientNetV2 branch (0.490) score above it (0.487), and the last two only by 0.003–0.009.
  - On energy in mask (centre 0.603), seven maps score above it: ResNet-50 (0.711), the hybrid's EfficientNetV2 branch (0.706; hair removed 0.610), both ViT Grad-CAMs (0.654, 0.644), and the Swin → EffNetV2 cross-attention (0.655; hair removed 0.636).
  - So most maps localise the lesion no better than "look at the middle of the image". Reported IoUs without such a reference would overstate localisation.
- **Maps that equal the uniform map:** ViT attention rollout and the hybrid's fused-token Grad-CAM score exactly the uniform map's IoU (0.418) and box IoU (0.544), trained and random alike.
  - Their maps are so flat that the 15% threshold keeps the whole image. They carry no localisation information under this protocol.
- **Random-weights sanity check:** trained Grad-CAM beats random weights on IoU for every CNN, Swin, ViT and branch map.
  - On energy in mask it does so too, with two exceptions:
    - maps that are flat in both cases (rollout, fused tokens);
    - the concat hybrid's EfficientNetV2 branch (trained 0.424 vs random 0.562). Its trained map is also weak on pointing (0.485), so it is not evidence of learned localisation.
  - **ViT, before the scoring fix:** random weights appeared to beat trained on energy in mask (0.826 vs 0.654; pretrained norm 0.733 vs 0.644).
  - Checked read-only (scratchpad analysis of `maps.npz` and `per_image.csv`), the cause was the averaging:
    - A random ViT's Grad-CAM is all zero for 846 / 1,503 images (pretrained norm: 1,241 / 1,503), because the weighted sum is negative on every patch.
    - Those images were left out of the energy average only.
    - Counted as 0, like every other metric, random energy is 0.361 (pretrained norm 0.128) vs 0.654 / 0.644 trained.
  - **The non-empty random-ViT maps are sparse** (about 2–3 of 196 patches positive) and follow dark pixels.
    - Measured on 300 val images: correlation with image darkness +0.33 and with the lesion mask +0.27. The trained ViT's figures are +0.64 and +0.67.
    - Lesions are darker than skin, so a few dark patches often fall inside the lesion; 64% of the non-empty random maps have energy in mask > 0.9.
    - This matches the observation of Adebayo et al. 2018 ("Sanity Checks for Saliency Maps", `Related_papers/Sanity Checks for Saliency Maps/`; added to the folder after this section was first written, which called it external) that maps from randomly initialised networks can still follow input structure.
    - Why random-ViT Grad-CAM is so often entirely negative is not established.
  - **Pointing game:** random maps reach 0.81 (hybrid fused tokens) and 0.66–0.70 (concat), against a centre-pixel rate of 0.961. Pointing is weak evidence on this dataset.
- **Correct vs incorrect predictions:** only the hybrid's EfficientNetV2 branch localises clearly worse on misclassified images.
  - IoU 0.503 vs 0.383, pointing 0.906 vs 0.703 (165 incorrect); the hair-removed hybrid behaves the same way.
  - For the other models, IoU on misclassified images is equal or higher, so there localisation does not track correctness.
- **Hybrid vs baselines:** the hybrid's best map (EfficientNetV2 branch) ranks third on IoU (after ResNet-50 and the pretrained-norm ViT) and second on energy in mask (after ResNet-50).
  - Its Swin branch (0.410), fused map (0.422) and cross-attention maps (0.228–0.277 IoU) are below the uniform reference on IoU.
  - The cross-attention maps are peaked: high pointing (0.87–0.97) and energy in mask (0.56–0.66), but low IoU, because they highlight a small area.
- **Hair removal:**
  - ConvNeXt-T: slightly higher IoU (0.311 → 0.324).
  - Hybrid EfficientNetV2 branch: lower IoU (0.490 → 0.400) and energy in mask (0.706 → 0.610).
  - The DullRazor border-ring failure (see Hair removal) touches exactly this region.
- **Normalisation fix:** EfficientNetV2-S Grad-CAM unchanged (0.442 → 0.446); ViT-B/16 slightly higher (0.483 → 0.496).
- **All-zero maps** (every positive Grad-CAM value removed by the ReLU; they score 0 on every metric) are rare with trained weights (`empty_share`, predicted-class maps):
  - the hair-removed hybrid's EfficientNetV2 branch, 0.8%;
  - the hybrid's two branches, 0.1% each;
  - every other trained map, 0.
  - With random weights they dominate only for ViT: 56.3% (lr 3e-5) and 82.6% (pretrained norm).

## HAM10000 test set for the Milestone 3 runs, and paired classification tests (protocol written 2026-11-02, before the runs)
- **Runs evaluated on test once** (`evaluate.py`, same defaults as their val evaluation; wrapper `eval_test_m3.ps1`, log `outputs/logs/m3_eval_test_memory.log`): `ham10000_hybrid`, `ham10000_hybrid_concat`, `ham10000_effnetv2_s_pnorm`, `ham10000_vit_b16_pnorm_lr3e-5`, `ham10000_convnext_t_hair`, `ham10000_hybrid_hair`.
  - The M4 trust test run had already predicted the cross-attention hybrid's test labels (198 wrong, see M4 test results). No checkpoint or setting is changed because of that.
- **Paired tests** (`scripts/compare_paired.py`, multiclass mode, argmax; the same 1,000 resampled test sets for both models, seed 0; resamples missing a class are skipped):
  1. `ham10000_hybrid` minus `ham10000_convnext_t` (does the hybrid beat the best baseline?);
  2. `ham10000_hybrid` minus `ham10000_hybrid_concat` (does cross-attention fusion beat concatenation?).
- **Primary endpoint:** macro-F1 (the model-selection metric), Holm-corrected over the two comparisons. Balanced accuracy, accuracy, MCC, kappa, OvR AUC / AP and per-class sensitivity are exploratory.

### Results (test n = 1,503; run 2026-11-02 07:24–07:43, all six OK)
Sources: `outputs/runs/<run>/eval/metrics.json` (95% CI from 1,000 bootstrap resamples). The hair-removed runs are evaluated on their own preprocessing (DullRazor test images).

| Model | Run | Epoch | Val macro-F1 | Test macro-F1 [95% CI] | Bal. acc | Accuracy | AUC (OvR macro) | MCC | Kappa | Mel. sensitivity |
|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid, cross-attention | ham10000_hybrid | 10 | 0.8389 | 0.7903 [0.7501–0.8260] | 0.8027 | 0.8683 | 0.9575 | 0.7499 | 0.7495 | 0.601 |
| Hybrid, concat | ham10000_hybrid_concat | 6 | 0.8417 | 0.7823 [0.7433–0.8156] | 0.8000 | 0.8603 | 0.9712 | 0.7369 | 0.7366 | 0.625 |
| EfficientNetV2-S, pretrained norm | ham10000_effnetv2_s_pnorm | 9 | 0.7777 | 0.7655 [0.7255–0.7989] | 0.8315 | 0.8629 | 0.9619 | 0.7386 | 0.7379 | 0.565 |
| ViT-B/16 (lr 3e-5), pretrained norm | ham10000_vit_b16_pnorm_lr3e-5 | 8 | 0.8311 | 0.7698 [0.7242–0.8072] | 0.7881 | 0.8523 | 0.9564 | 0.7259 | 0.7250 | 0.649 |
| ConvNeXt-T, hair removed | ham10000_convnext_t_hair | 19 | 0.8527 | 0.8006 [0.7670–0.8293] | 0.8341 | 0.8656 | 0.9560 | 0.7448 | 0.7445 | 0.655 |
| Hybrid, hair removed | ham10000_hybrid_hair | 11 | 0.8442 | 0.8302 [0.7906–0.8603] | 0.8462 | 0.8842 | 0.9703 | 0.7832 | 0.7826 | 0.726 |
| (baseline, for reference) ConvNeXt-T | ham10000_convnext_t | 16 | 0.8384 | 0.8118 [0.7775–0.8424] | 0.8362 | 0.8729 | 0.9595 | 0.7599 | 0.7596 | 0.679 |

**Paired tests** (`outputs/comparisons/ham10000_test_ham10000_hybrid_vs_ham10000_convnext_t.json`, `..._vs_ham10000_hybrid_concat.json`; 1,000 of 1,000 resamples valid). Holm from `metrics.holm` over the two primary p-values.

| B − A | Macro-F1 (primary) [95% CI] | p (Holm) | Bal. acc | Accuracy | AUC | Mel. sensitivity |
|---|---|---|---|---|---|---|
| Hybrid − ConvNeXt-T | −0.022 [−0.054, +0.007] | 0.148 (0.296) | −0.034 [−0.072, +0.001], p 0.056 | −0.005, p 0.54 | −0.002, p 0.88 | −0.077 [−0.149, −0.006], p 0.042 |
| Hybrid − concat hybrid | +0.008 [−0.018, +0.032] | 0.480 (0.480) | +0.003, p 0.82 | +0.008, p 0.23 | −0.014 [−0.032, −0.000], p 0.048 | −0.024, p 0.53 |

- **Primary result: neither comparison is significant.** The hybrid is not shown to beat ConvNeXt-T (it is 0.022 lower on macro-F1, CI includes 0), and cross-attention is not shown to beat concatenation (+0.008, CI includes 0).
- **Exploratory, uncorrected** (several per-class tests, so single p ≈ 0.02–0.05 are weak evidence):
  - Against ConvNeXt-T the hybrid has lower melanoma sensitivity (0.601 vs 0.679; 101 vs 114 of 168) and df sensitivity (11 vs 15 of 17), and higher bcc sensitivity (0.897 vs 0.833).
  - Against concat, the cross-attention hybrid has lower OvR AUC (0.958 vs 0.971) and higher nv sensitivity.
- **Val → test:** the cross-attention hybrid dropped the most among the main runs (0.8389 → 0.7903), as did ViT-B/16 with pretrained norm (0.8311 → 0.7698). ConvNeXt-T dropped 0.027.
- **The hair-removed hybrid is the best run on test** (macro-F1 0.830, melanoma sensitivity 0.726). It was not a pre-specified comparison; its val score (0.8442) was close to the others. It is reported as an observation, not a finding, and no claim rests on it without a paired test planned in advance.

## ISIC 2020 (Milestone 3c), validation only

- **Run:** `isic2020_convnext_t`, ConvNeXt-T (`configs/model/convnext_t.yaml`), trained on Kaggle (2× Tesla T4, one session 09:26–10:42 on 2026-11-01, not resumed).
  - Profile `kaggle_16gb`: batch 32, no accumulation. Otherwise the same recipe as HAM10000, with early stopping on val AUC.
  - Loss: cross-entropy with sqrt-inverse-frequency class weights (0.236 benign, 1.764 melanoma) and label smoothing 0.1; no balanced sampler.
  - Split `data/splits/isic2020.csv` (grouped by patient): train 23,190 (408 melanoma), val 4,969 (87 melanoma). The file used on Kaggle has the same SHA-256 as the local one.
  - Versions on Kaggle: torch 2.11.0+cu128, timm 1.0.30, albumentations 2.0.8, scikit-learn 1.9.1 (`kaggle_sessions.json`).
- **Training** (`train.log`, `history.csv`): best val AUC 0.9085 at epoch 8; early stop after epoch 15.
  - Epoch 1 took 2,721 s, likely the first read of the dataset from Kaggle's input storage. Every later epoch took about 125 s.
  - After epoch 8, train loss kept falling (0.803 → 0.775) while val AUC drifted down (0.9085 → 0.8816).
- **Val evaluation**, done locally: `scripts/evaluate.py --val-only --data configs/data/isic2020.yaml`, output in `outputs/runs/isic2020_convnext_t/eval_val/`. The test split was not loaded.
  - It matches the Kaggle `history.csv` epoch-8 row to 4 decimals except AUC: 0.90847 locally vs 0.90852 logged on Kaggle. The 0.00005 gap is attributed to the torch version or image decoding (not investigated further).

| Metric (val, n = 4,969, argmax = threshold 0.5) | Value |
|---|---|
| AUC | 0.9085 |
| Average precision | 0.2036 |
| Accuracy | 0.9779 |
| Balanced accuracy | 0.6049 |
| Macro-F1 | 0.6228 |
| MCC / kappa | 0.2499 / 0.2459 |
| Melanoma sensitivity | 0.218 (19 / 87) |
| Specificity | 0.991 (42 false positives / 4,882 benign) |

- **Observations:**
  - The model ranks melanomas well (AUC 0.91), but at argmax it calls only 61 images melanoma and misses 68 of the 87.
  - With 1.75% melanoma in val, accuracy (0.978) is close to the all-benign rate (0.982) and says little.
  - AP (0.20) is far below AUC: at any useful sensitivity, most positive calls are false positives.
  - So the decision threshold matters more here than on HAM10000. It is chosen on val and fixed before any test run (below).

### Decision threshold, fixed on val (`eval_val/threshold.json`, `scripts/select_threshold.py`)
- **Rule** (the same for every ISIC 2020 run, including the hybrid): predict melanoma when P(melanoma) ≥ cutoff. Two cutoffs are chosen from `eval_val/predictions_val.csv` only:
  - **Youden:** maximises sensitivity + specificity − 1 (Youden's J). The rule is a standard choice, not taken from the papers in `Related_papers/`.
  - **Sensitivity ≥ 0.80:** the highest cutoff whose val sensitivity is at least 0.80, i.e. the best specificity among those. With 87 val melanomas this means at least 70 caught.
- **Fixing before test:**
  - `threshold.json` records both cutoffs, the checkpoint epoch and the SHA-256 of the predictions file.
  - The test evaluation (`evaluate.py`) reads it and stops before loading anything if it is missing; it also stops if the checkpoint epoch differs.
  - `select_threshold.py` refuses to run once `eval/` (test) exists.
- **Caveat:** the val numbers below are measured on the same images the cutoffs were chosen on, so they are optimistic. Only the test numbers at these fixed cutoffs will be an unbiased estimate.

| ConvNeXt-T, val (n = 4,969, 87 melanoma) | Cutoff | Called melanoma | Sensitivity | Specificity | Balanced acc. | Melanoma F1 | MCC | Confusion [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|---|---|---|
| Argmax (for reference) | 0.5 | 61 | 0.218 | 0.991 | 0.605 | 0.257 | 0.250 | [[4840, 42], [68, 19]] |
| Youden | 0.3277 | 958 | 0.885 | 0.820 | 0.852 | 0.147 | 0.234 | [[4001, 881], [10, 77]] |
| Sensitivity ≥ 0.80 | 0.3318 | 822 | 0.805 | 0.846 | 0.825 | 0.154 | 0.230 | [[4130, 752], [17, 70]] |

- **Observations:**
  - Both cutoffs sit far below 0.5: P(melanoma) is compressed into a narrow band.
    - Benign: median 0.301, 5th–95th percentile 0.285–0.372, minimum 0.282.
    - Melanoma: median 0.375, 5th–95th percentile 0.319–0.837.
    - The cause (for example label smoothing combined with the class-weighted loss) has not been checked.
  - The two cutoffs differ by only 0.004, yet they call 958 vs 822 images melanoma. The operating point is therefore fragile: a small shift in the probabilities on test (or between machines) can move sensitivity noticeably.
  - At either cutoff, about 1 in 12 melanoma calls is correct (precision 77/958 = 0.080 and 70/822 = 0.085). MCC barely changes (0.25 → 0.23), while sensitivity rises from 0.22 to 0.80–0.89.

### Hybrid on ISIC 2020, validation only
- **Run:** `isic2020_hybrid`, `configs/model/hybrid.yaml` (cross-attention fusion).
  - Trained on Kaggle: 2× Tesla T4, one session 10:47–12:24 on 2026-11-01, not resumed.
  - Same split, recipe, loss, class weights, `kaggle_16gb` profile and versions as `isic2020_convnext_t` (`config.json`, `kaggle_sessions.json`).
- **Training** (`train_summary.json`, `history.csv`): best val AUC 0.9118 at epoch 7; early stop after epoch 14 (7 epochs without improvement).
  - Epoch 1 took 1,628 s and every later epoch about 314 s; GPU peak 5.51 GiB.
  - After epoch 7, train loss kept falling (0.797 → 0.753) while val AUC fell to 0.8593–0.8679 in epochs 12–14.
- **Val evaluation**, done locally: `evaluate.py --val-only --data configs/data/isic2020.yaml --batch-size 8 --num-workers 0`.
  - It matches the Kaggle epoch-7 row to 4 decimals except AUC: 0.91191 locally vs 0.91180 on Kaggle.
  - Thresholds: `select_threshold.py --min-sensitivity 0.80`, the same rule as ConvNeXt-T. They are fixed in `eval_val/threshold.json`. The test split was not loaded.

### ConvNeXt-T vs hybrid on ISIC 2020 val (n = 4,969, 87 melanoma; `eval_val/metrics.json`, `eval_val/threshold.json`)
Val metrics at the val-chosen cutoffs are optimistic (see above). These are single runs without confidence intervals.

| Metric (val) | ConvNeXt-T (epoch 8) | Hybrid (epoch 7) |
|---|---|---|
| AUC | 0.9085 | 0.9119 |
| Average precision | 0.204 | 0.322 |
| **Argmax**: sensitivity / specificity | 0.218 (19/87) / 0.991 | 0.368 (32/87) / 0.991 |
| Argmax: precision | 0.311 (19/61) | 0.421 (32/76) |
| **Youden** cutoff | 0.3277 | 0.3145 |
| Youden: sensitivity / specificity | 0.885 (77/87) / 0.820 | 0.839 (73/87) / 0.872 |
| Youden: balanced accuracy | 0.852 | 0.855 |
| Youden: precision | 0.080 (77/958) | 0.104 (73/700) |
| **Sensitivity ≥ 0.80** cutoff | 0.3318 | 0.3162 |
| Sens ≥ 0.80: sensitivity / specificity | 0.805 (70/87) / 0.846 | 0.805 (70/87) / 0.879 |
| Sens ≥ 0.80: precision | 0.085 (70/822) | 0.106 (70/662) |
| MCC (argmax / Youden / sens ≥ 0.80) | 0.250 / 0.234 / 0.230 | 0.383 / 0.268 / 0.264 |

- **Observations:**
  - **Ranking:** AUC is almost equal (+0.003 for the hybrid). With 87 positives that gap is within the noise of a single run; a paired test is left for Milestone 4.
  - **The clearer differences:**
    - AP is 0.322 vs 0.204.
    - At the same sensitivity (70/87), the hybrid makes 160 fewer false positives (592 vs 752; specificity 0.879 vs 0.846).
    - At argmax it catches 13 more melanomas (32 vs 19) at the same specificity.
    - So the hybrid separates the high-scoring end better, even though the overall ranking (AUC) is about the same.
  - **Youden:** the two models land at different operating points. ConvNeXt-T trades specificity for sensitivity (0.885 / 0.820), the hybrid the other way (0.839 / 0.872). Balanced accuracy is the same (0.852 vs 0.855).
  - **Compressed probabilities** (same pattern as ConvNeXt-T):
    - Hybrid benign: median 0.291, 5th–95th percentile 0.284–0.351.
    - Hybrid melanoma: median 0.370, 5th–95th percentile 0.296–0.902.
    - The two hybrid cutoffs differ by 0.002 (700 vs 662 calls), so its operating point is just as fragile.
  - **Precision stays low** for both models at the useful cutoffs (0.08–0.11, about 1 in 10 melanoma calls correct), as expected at 1.75% prevalence.

### ISIC 2020 test set: final, run once (2026-11-01)
- **Runs:** `evaluate.py --data configs/data/isic2020.yaml --batch-size 8 --num-workers 0`, once per run.
  - Output: `outputs/runs/<run>/eval/` (`metrics.json`, `predictions.csv`).
  - Logs: `outputs/logs/isic2020_<model>_evaltest_*.log`.
- **Thresholds:** read from the `eval_val/threshold.json` files written before the test run. The `eval/metrics.json` files record their SHA-256 and creation time (17:13 for ConvNeXt-T and 19:22 for the hybrid; the test runs started at 19:25). No threshold was changed after the test results were seen.
- **Data:** test n = 4,967, 89 melanoma.
- **Confidence intervals:** 95% percentile bootstrap, 1,000 resamples.
  - AUC, AP, balanced accuracy and MCC from `evaluate.py` (`eval/metrics.json`).
  - Sensitivity and specificity from `scripts/compare_paired.py` (`outputs/comparisons/isic2020_test_isic2020_hybrid_vs_isic2020_convnext_t.json`), because `evaluate.py`'s bootstrap does not include the binary sensitivity/specificity.

| Test | ConvNeXt-T (epoch 8) | Hybrid (epoch 7) |
|---|---|---|
| AUC | 0.891 [0.865, 0.916] | 0.899 [0.874, 0.923] |
| Average precision | 0.201 [0.125, 0.296] | 0.242 [0.155, 0.336] |
| **Argmax** sensitivity | 0.213 (19/89) [0.129, 0.304] | 0.213 (19/89) [0.126, 0.304] |
| Argmax specificity | 0.988 [0.985, 0.991] | 0.991 [0.988, 0.994] |
| Argmax balanced acc. / MCC | 0.601 [0.559, 0.646] / 0.218 [0.131, 0.311] | 0.602 [0.559, 0.648] / 0.242 [0.147, 0.342] |
| Argmax precision | 0.250 (19/76) | 0.302 (19/63) |
| **Youden** cutoff (fixed on val) | 0.3277 | 0.3145 |
| Youden sensitivity | 0.854 (76/89) [0.776, 0.925] | 0.730 (65/89) [0.641, 0.823] |
| Youden specificity | 0.796 [0.785, 0.807] | 0.857 [0.848, 0.867] |
| Youden balanced acc. / MCC | 0.825 [0.786, 0.860] / 0.209 [0.178, 0.241] | 0.794 [0.749, 0.839] / 0.216 [0.177, 0.257] |
| Youden precision | 0.071 (76/1,073) | 0.085 (65/761) |
| **Sensitivity ≥ 0.80** cutoff (fixed on val) | 0.3318 | 0.3162 |
| Sens ≥ 0.80 sensitivity | 0.809 (72/89) [0.716, 0.890] | 0.730 (65/89) [0.641, 0.823] |
| Sens ≥ 0.80 specificity | 0.826 (0.8255 counting ties, see below) [0.816, 0.836] | 0.868 [0.859, 0.877] |
| Sens ≥ 0.80 balanced acc. / MCC | 0.818 [0.770, 0.859] / 0.217 [0.181, 0.251] | 0.799 [0.754, 0.845] / 0.227 [0.186, 0.269] |
| Sens ≥ 0.80 precision | 0.078 (72/919) | 0.092 (65/709) |

**Paired bootstrap, hybrid minus ConvNeXt-T** (the same 1,000 resampled test sets for both models, seed 0; `scripts/compare_paired.py`).
The p-value is two-sided, 2 × min(share of resamples with difference ≤ 0, share with difference ≥ 0); with 1,000 resamples the smallest non-zero value is 0.002.

| Difference (B − A) | Point | 95% CI | p |
|---|---|---|---|
| AUC | +0.009 | [−0.011, +0.028] | 0.36 |
| Average precision | +0.041 | [−0.011, +0.091] | 0.13 |
| Specificity at the sens ≥ 0.80 cutoff | +0.042 | [+0.034, +0.051] | < 0.002 (0 of 1,000) |
| Sensitivity at the sens ≥ 0.80 cutoff | −0.079 | [−0.184, +0.031] | 0.21 |
| Specificity at Youden | +0.062 | [+0.052, +0.072] | < 0.002 (0 of 1,000) |
| Sensitivity at Youden | −0.124 | [−0.219, −0.024] | 0.016 |
| Specificity at argmax | +0.003 | [0.000, +0.006] | 0.07 |
| Sensitivity at argmax | 0.000 | [−0.075, +0.075] | 1.00 |

- **Observations:**
  - **Ranking:** the hybrid is not significantly better at ranking. AUC +0.009 and AP +0.041 both have CIs that include 0. The val AP gap (0.322 vs 0.204) shrank on test (0.242 vs 0.201).
  - **Val → test:** AUC fell for both (ConvNeXt-T 0.9085 → 0.891, hybrid 0.9119 → 0.899); AP held for ConvNeXt-T (0.204 → 0.201) and fell for the hybrid (0.322 → 0.242).
  - **The sens ≥ 0.80 cutoff did not transfer for the hybrid:** on test it catches 65/89 (0.730), below the 0.80 target, while ConvNeXt-T's cutoff holds (72/89, 0.809).
    - This is the fragility noted on val: the cutoff sits inside a narrow, crowded probability band.
    - So the hybrid's higher specificity at this cutoff (+0.042, CI excludes 0) is **not like-for-like**. It comes at a sensitivity cost (−0.079, CI includes 0), and at Youden the trade is clearer (specificity +0.062, sensitivity −0.124 with CI excluding 0).
    - Neither model is shown to be better at a fixed sensitivity on test. A fair comparison at equal sensitivity would need a cutoff chosen per model on test, which is not allowed here.
  - **Precision** at the useful cutoffs is 0.07–0.09 for both models; at argmax both catch only 19/89.
  - **Single runs, single seed.** The CIs cover test-sample variation only, not training variation.
- **Tied probabilities at a cutoff:**
  - With AMP, probabilities are quantised: ConvNeXt-T's test set has only 1,144 distinct P(melanoma) values among 4,967 images.
  - ConvNeXt-T's sens ≥ 0.80 cutoff (0.33176893) is exactly the probability of 3 val images, including the melanoma that brings val sensitivity to 70/87. It is also exactly the probability of 4 benign test images.
  - `select_threshold.py` and `compare_paired.py` compare the CSV values, so ties count as melanoma, which is consistent with how the cutoff was chosen on val. This gives specificity 0.8255 (851 false positives).
  - `evaluate.py` compares the in-memory float32 value (0.33176892996), which falls just below the decimal cutoff, so it excludes the ties: 0.8264 (847 false positives).
  - Among ConvNeXt-T and the hybrid, only this one model/cutoff pair is affected: the other three agree to 4 decimals between the two scripts. The paired difference uses 0.8255. (Updated 2026-11-03: the EfficientNetV2-S run below is affected at both of its cutoffs.)
  - Fixing the comparison precision in `predict_labels` (comparing in float32) would make the two scripts agree. That would change how the fixed rule is applied, not the threshold, and has not been done.

### ISIC 2020: EfficientNetV2-S with pretrained normalisation (the hybrid's local branch alone), run once (2026-11-03)
- **Purpose** (Deviations): isolate what the Swin branch and the fusion add on ISIC 2020. Pre-specified comparison: hybrid minus EfficientNetV2-S, paired, on the test set.
- **Training on Kaggle** (`outputs/runs/isic2020_effnetv2_s_pnorm/`, from the output zip of 2026-11-03; `kaggle_sessions.json`): 2× Tesla T4, one session 09:28–10:44, not resumed, status finished. torch 2.11.0+cu128, timm 1.0.30, albumentations 2.0.8, scikit-learn 1.9.1.
  - `config.json`: model `effnetv2_s_pnorm` (`tf_efficientnetv2_s.in21k_ft_in1k`, `match_pretrained_norm: true`); data and training settings identical to `isic2020_hybrid` (same split CSV, `kaggle_16gb` profile, loss, class weights).
  - Best val AUC 0.8852 at epoch 5; early stop after epoch 12 (7 epochs without improvement). Epoch 1 took 2,643 s, later epochs about 163 s; GPU peak 2.74 GiB.
  - After epoch 5, train loss kept falling (0.801 → 0.756) while val AUC fell to 0.849 at epoch 12 (`history.csv`).
- **Evaluation, locally, in this order** (2026-11-03, each step exit 0): `evaluate.py --val-only` (17:09), `select_threshold.py --min-sensitivity 0.80` (`eval_val/threshold.json`, 17:09:16), `evaluate.py` on test once (17:13; `eval/metrics.json` records the threshold file's SHA-256, which matches), `compare_paired.py --runs outputs/runs/isic2020_effnetv2_s_pnorm outputs/runs/isic2020_hybrid`.
  - Val (n = 4,969, 87 melanoma): AUC 0.8852, accuracy 0.9783, balanced accuracy 0.6390, MCC 0.3072; argmax melanoma sensitivity 0.287.
  - Val cutoffs (optimistic, chosen on these images): Youden 0.3109 (1,334 called, sensitivity 0.885, specificity 0.743); sensitivity ≥ 0.80: 0.3222 (982 called, 0.805 / 0.813).
- **Test** (n = 4,967, 89 melanoma; 95% CI from 1,000 bootstrap resamples, `eval/metrics.json`):

| Test | EfficientNetV2-S (epoch 5) |
|---|---|
| AUC | 0.8946 [0.8658, 0.9224] |
| Average precision | 0.2175 [0.1414, 0.3210] |
| Argmax sensitivity / specificity | 0.2247 (20/89) / 0.9910 |
| Argmax balanced acc. / MCC | 0.6078 [0.5643, 0.6525] / 0.2537 [0.1583, 0.3504] |
| Youden cutoff 0.3109: sensitivity / specificity | 0.888 (79/89) / 0.719 (0.718 counting ties) |
| Sens ≥ 0.80 cutoff 0.3222: sensitivity / specificity | 0.809 (72/89) / 0.795 (0.795 counting ties: 0.7946 vs 0.7952) |
| Precision at Youden / sens ≥ 0.80 (ties counted) | 0.054 (79/1,454) / 0.067 (72/1,074) |

- **Ties at the cutoffs:** 3 test images have exactly the Youden cutoff's probability and 3 the sens ≥ 0.80 cutoff's. `evaluate.py` (float32 comparison) gives specificity 0.7187 and 0.7952; `compare_paired.py` (CSV values, ties counted as melanoma, as on val) gives 0.7181 and 0.7946. Sensitivity agrees. The paired test uses the latter. Same cause and same decision (not fixed after the test run) as for ConvNeXt-T above.
- **Paired bootstrap, hybrid minus EfficientNetV2-S** (`outputs/comparisons/isic2020_test_isic2020_hybrid_vs_isic2020_effnetv2_s_pnorm.json`; 1,000 of 1,000 resamples valid):

| Difference (B − A) | Point | 95% CI | p |
|---|---|---|---|
| AUC | +0.0045 | [−0.0130, +0.0218] | 0.662 |
| Average precision | +0.0244 | [−0.0217, +0.0643] | 0.390 |
| Specificity at sens ≥ 0.80 cutoff | +0.0734 | [+0.0634, +0.0829] | < 0.002 |
| Sensitivity at sens ≥ 0.80 cutoff | −0.0787 | [−0.1507, −0.0115] | 0.040 |
| Specificity at Youden | +0.1392 | [+0.1281, +0.1507] | < 0.002 |
| Sensitivity at Youden | −0.1573 | [−0.2445, −0.0818] | < 0.002 |
| Specificity at argmax | +0.0000 | [−0.0027, +0.0025] | 1.000 |
| Sensitivity at argmax | −0.0112 | [−0.0780, +0.0602] | 0.908 |

- **Observations:**
  - **Ranking:** the hybrid is not significantly better than its local branch alone: AUC +0.0045 (p 0.662), AP +0.0244 (p 0.390). On ISIC 2020 the Swin branch and the fusion add no measurable ranking ability. The three ISIC test AUCs are close: EfficientNetV2-S 0.895, ConvNeXt-T 0.891, hybrid 0.899.
  - **Val → test:** EfficientNetV2-S had the lowest val AUC of the three (0.8852 vs 0.9085 ConvNeXt-T, 0.9118 hybrid) but its test AUC rose to 0.8946, while the other two fell.
  - **Cutoffs:** EfficientNetV2-S's sens ≥ 0.80 cutoff transferred to test (72/89, 0.809), like ConvNeXt-T's and unlike the hybrid's (65/89, 0.730). At each model's own cutoffs the hybrid again trades sensitivity for specificity, so this is not a like-for-like comparison at equal sensitivity.
  - Single run, single seed, as for the other ISIC 2020 runs.

## M4 explanation trust on HAM10000: protocol frozen on val (2026-11-02)

- **Question:** are the hybrid's explanations more trustworthy than ConvNeXt-T's?
  - Map A: ConvNeXt-T Grad-CAM (`ham10000_convnext_t`, epoch 16).
  - Maps B: the hybrid's EfficientNetV2-branch Grad-CAM (`gradcam_local`) and fused-token Grad-CAM (`gradcam_fused`) (`ham10000_hybrid`, cross-attention, epoch 10).
  - All maps are for the predicted class, on the native 7×7 grid, upsampled to 224 px.
- **Code and settings:** `scripts/eval_trust.py`, `scripts/compare_xai_paired.py`, `configs/xai/trust.yaml`. Run with the scratchpad wrapper `trust_ham.ps1 -Split val|test`, each step under the memory guard; log `outputs/logs/trust_<split>_ham10000_memory.log`.
- **This protocol is now frozen.** The test run uses it unchanged, and nothing will be changed after the test results are seen.

### Metrics (all per image, then averaged)
- **Faithfulness: deletion / insertion** (Petsiuk et al. 2018, RISE; not in `Related_papers/`, cited as external).
  - Pixels are ranked by the map and changed 1/50 at a time (50 steps).
  - Deletion replaces them with the dataset-mean colour (0 after normalisation). Insertion starts from a Gaussian blur of the image (kernel 11, σ 5) and puts the original pixels back.
  - Score: softmax probability of the originally predicted class. AUC by the trapezoid rule over the 51 points.
  - **`*_vs_random`:** the map's AUC minus the AUC of that run's own random map (uniform noise on a 7×7 grid, seeded per image, upsampled like the real maps). Deletion: lower is better. Insertion: higher is better.
  - A centred-Gaussian map (σ 0.25 × image size) is scored as a second reference.
- **Sanity: cascading parameter randomisation** (Adebayo et al. 2018, `Related_papers/Sanity Checks for Saliency Maps/`).
  - Blocks are re-initialised top down and cumulatively. ConvNeXt-T: head, then the four stages from last to first, then the stem. Hybrid: head, fusion, then both branches stage by stage, then both stems.
  - Re-initialisation follows the paper's code `code/src/utils.py:300-314`: the layers' own initialisers are re-run, and BatchNorm running statistics are kept.
  - After each step: Spearman rank correlation, SSIM (`gaussian_weights=True`) and HOG Spearman (16 px per cell) between the trained and the randomised map, as in `code/notebooks/inceptionv3_cascading_randomization.ipynb`.
  - Checked 2026-11-03 from the PDF's text: cascading randomisation is the paper's §3.1.
  - Main number: similarity after the last step (`rand_full_*`). Lower is better: the map should depend on the learned weights.
- **Stability:** Spearman and SSIM between the map of the original image and the map after each of four perturbations, with the target class held at the original prediction.
  - Perturbations: horizontal flip (the map is flipped back), Gaussian noise σ 0.03 on the 0–1 pixel scale, brightness ×0.9 and ×1.1.
  - `stab_*_mean` averages only the perturbations that did **not** change the prediction (`stab_n_unchanged` counts them). Higher is better.
- **Localisation** (from `eval_xai.py`, protocol as in Milestone 3b): IoU at 15% of the max (Grad-CAM §4.1), energy in mask, pointing game (Grad-CAM §4.3).
  - **`iou_gap`:** mean IoU on correctly predicted images minus mean IoU on wrong ones, each model judged by its own correctness. Higher is better: the explanation should be worse when the model is wrong.
- **Paired bootstrap:** the same 1,000 resampled image sets for both models (seed 0); B − A with percentile 95% CI.
  - Two-sided p = 2 × min(share of resamples with difference ≤ 0, share with difference ≥ 0). The smallest non-zero value is 0.002; zero is written "< 0.002".
  - `iou_gap` is recomputed inside each resample.
  - An image without a value (stability, when all four perturbations changed the prediction) is skipped.

### Endpoints and decision rule (fixed before the val run; unchanged)
- **Primary**, for each of the two hybrid maps (12 tests, Holm-corrected together): `deletion_vs_random`, `insertion_vs_random`, `rand_full_ssim`, `stab_spearman_mean`, `iou`, `iou_gap`. Everything else is exploratory.
- **Decision rule:** "the hybrid map is more trustworthy than ConvNeXt-T's" holds only if both of these are true (Holm-adjusted p < 0.05):
  - it is significantly better on faithfulness (`deletion_vs_random` or `insertion_vs_random`) **and** on the sanity endpoint (`rand_full_ssim`);
  - it is not significantly worse on any other primary endpoint.

### Changes made after looking at val data (all before any test run)
1. **Faithfulness enters relative to the random map** (after the 99-image val smoke run). The planned primary endpoints were the raw deletion / insertion AUCs.
   - The fully deleted image still gets 0.26 (ConvNeXt-T) vs 0.43 (hybrid) probability, so raw AUCs mix each model's calibration into the comparison.
2. **Stability averages only perturbations that left the prediction unchanged** (after the smoke run). Noise σ 0.03 changed 16–19% of predictions, and a map that follows a changed decision is not unstable.
3. **SSIM caveat** (after the smoke run): flat maps score a high SSIM by themselves.
   - `rand_full_ssim` stays primary, as planned. `rand_full_*spearman` and `rand_full_*hog` are reported next to it, and Spearman (not SSIM) is the primary stability metric.
4. **|ρ| for the randomisation rank correlations** (after the full val run): `rand_full_spearman` / `rand_full_hog` became `rand_full_abs_spearman` / `rand_full_abs_hog`.
   - The signed values rewarded a map for becoming anti-correlated with the trained map, when the check wants no correlation. Both are exploratory, so the primary set did not change.
- **Considered and not changed:** the fused map's IoU is higher than ConvNeXt-T's while its energy in mask and pointing are lower. This suggests a more diffuse map, which IoU at the 15% threshold rewards on large lesions (Milestone 3b: a uniform map already scores IoU 0.418).
  - IoU stays primary, because changing it after seeing results that go against the hybrid would be outcome-driven.
  - `energy_in_mask` and `pointing` are reported next to it, with this caveat.

### Val results (n = 1,503; not confirmatory: these data informed changes 1–4)
- **Source:** `outputs/comparisons/ham10000_xai_val_ham10000_hybrid_vs_ham10000_convnext_t.json`, rerun after change 4 (the `trust_val/` maps and scores were not recomputed).
- **Run time:** ConvNeXt-T 21.7 min, hybrid 60.1 min (`trust_val_ham10000_memory.log`). GPU peak 1.1 / 1.6 GB.

**Hybrid map minus ConvNeXt-T Grad-CAM, val.** P = primary endpoint; Holm is over the 12 primary tests. A raw p of < 0.002 (no resample crossed zero) has a Holm-adjusted bound of < 12 × 0.002 = 0.024 (corrected 2026-11-03 from "< 0.002"; no verdict changes).

| Metric | Better | ConvNeXt-T | Local | Local − ConvNeXt-T [95% CI] | p (Holm) | Fused | Fused − ConvNeXt-T [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| deletion_vs_random (P) | lower | −0.084 | −0.041 | +0.043 [+0.032, +0.054] | < 0.002 (< 0.024) | −0.086 | −0.002 [−0.013, +0.009] | 0.69 (1.00) |
| insertion_vs_random (P) | higher | +0.065 | +0.031 | −0.034 [−0.041, −0.027] | < 0.002 (< 0.024) | +0.048 | −0.017 [−0.023, −0.011] | < 0.002 (< 0.024) |
| rand_full_ssim (P) | lower | 0.267 | 0.264 | −0.002 [−0.010, +0.006] | 0.62 (1.00) | 0.294 | +0.027 [+0.016, +0.039] | < 0.002 (< 0.024) |
| stab_spearman_mean (P) | higher | 0.938 | 0.701 | −0.237 [−0.252, −0.222] | < 0.002 (< 0.024) | 0.890 | −0.048 [−0.053, −0.042] | < 0.002 (< 0.024) |
| iou (P) | higher | 0.311 | 0.490 | +0.179 [+0.164, +0.193] | < 0.002 (< 0.024) | 0.422 | +0.111 [+0.100, +0.123] | < 0.002 (< 0.024) |
| iou_gap (P) | higher | −0.049 | +0.120 | +0.169 [+0.125, +0.214] | < 0.002 (< 0.024) | −0.046 | +0.003 [−0.049, +0.055] | 0.93 (1.00) |
| deletion_auc | lower | 0.437 | 0.524 | +0.087 [+0.077, +0.097] | < 0.002 | 0.479 | +0.042 [+0.032, +0.052] | < 0.002 |
| insertion_auc | higher | 0.719 | 0.640 | −0.079 [−0.086, −0.070] | < 0.002 | 0.657 | −0.062 [−0.069, −0.054] | < 0.002 |
| rand_full_abs_spearman | lower | 0.241 | 0.218 | −0.023 [−0.036, −0.010] | < 0.002 | 0.204 | −0.037 [−0.049, −0.024] | < 0.002 |
| rand_full_abs_hog | lower | 0.080 | 0.058 | −0.022 [−0.026, −0.018] | < 0.002 | 0.057 | −0.023 [−0.028, −0.019] | < 0.002 |
| stab_ssim_mean | higher | 0.925 | 0.698 | −0.228 [−0.236, −0.219] | < 0.002 | 0.975 | +0.049 [+0.047, +0.052] | < 0.002 |
| energy_in_mask | higher | 0.521 | 0.706 | +0.186 [+0.169, +0.202] | < 0.002 | 0.446 | −0.075 [−0.085, −0.065] | < 0.002 |
| pointing | higher | 0.910 | 0.884 | −0.027 [−0.047, −0.003] | 0.028 | 0.849 | −0.061 [−0.083, −0.039] | < 0.002 |

- **Reference maps** (deletion / insertion AUC): ConvNeXt-T random 0.521 / 0.654, centre 0.510 / 0.702; hybrid random 0.565 / 0.609, centre 0.512 / 0.633.
- **Predictions changed by each perturbation:**
  - ConvNeXt-T: flip 3.9%, noise 17.6%, brightness ×0.9 1.7%, ×1.1 1.3%.
  - Hybrid: 2.8%, 17.7%, 1.7%, 1.7%.
  - On average 3.76 of the 4 perturbations are kept. One ConvNeXt-T image (all four perturbations changed its prediction) has no stability value.
- **Wrong predictions** (for `iou_gap`): ConvNeXt-T 167, hybrid 165.
- **Signed randomisation correlations** before change 4: Spearman −0.006 (ConvNeXt-T), −0.035 (local), −0.060 (fused); HOG 0.071, 0.040, 0.045.
  - The per-image |ρ| (0.20–0.24) is much larger than the signed mean: individual randomised maps keep some correlation in either direction, which averages out.
- **Observations (val; the test run decides):**
  - **Under the decision rule, neither hybrid map would qualify on val.** Both are significantly *worse* on insertion, and neither is significantly better on any faithfulness endpoint.
  - Local map: no difference on `rand_full_ssim`; much less stable (Spearman 0.70 vs 0.94).
  - Fused map: no difference on deletion; worse on `rand_full_ssim` (more similar after full randomisation) and on stability Spearman.
  - **Where the hybrid is better is localisation:** IoU for both maps, and `iou_gap` for the local map. ConvNeXt-T localises slightly *better* on its wrong predictions (gap −0.049), while the hybrid's local map localises clearly worse when the model is wrong (+0.120), which is the behaviour a trustworthy explanation should show.
  - **Faithfulness vs references:** ConvNeXt-T Grad-CAM beats its own centre map on deletion (0.437 vs 0.510) and insertion (0.719 vs 0.702). The hybrid's local map does not beat the hybrid's centre map on deletion (0.524 vs 0.512) and only barely on insertion (0.640 vs 0.633).
  - The fused map's stability SSIM (0.975) exceeds its Spearman (0.890), in line with change 3.

### Test run (once, after this section was written)
- Command: `trust_ham.ps1 -Split test`. This runs `explain.py` + `eval_xai.py` on the test split for both runs, then `eval_trust.py`, then `compare_xai_paired.py`.
- Results will go below this section, read against the decision rule above.

## M4 explanation trust on HAM10000: test results (final, run once, 2026-11-02)
- **Run:** `trust_ham.ps1 -Split test`, 01:01–02:26 on 2026-11-02; every step exited 0 (`outputs/logs/trust_test_ham10000_memory.log`).
  - `explain.py` + `eval_xai.py` on the test split: 0.8 + 0.8 min (ConvNeXt-T), 0.9 + 1.1 min (hybrid).
  - `eval_trust.py`: 21.3 min (ConvNeXt-T), 60.0 min (hybrid). GPU peak 1.1 / 1.6 GB.
  - Same checkpoints as on val (ConvNeXt-T epoch 16, hybrid epoch 10).
  - Protocol, endpoints and decision rule exactly as frozen above; nothing was changed after the results were seen.
- **Sources:** `outputs/comparisons/ham10000_xai_test_ham10000_hybrid_vs_ham10000_convnext_t.json` (+ `.log`); `outputs/runs/<run>/trust_test/`, `xai_test/`.

**Hybrid map minus ConvNeXt-T Grad-CAM, test (n = 1,503).** P = primary endpoint; Holm is over the 12 primary tests. A raw p of < 0.002 (no resample crossed zero) has a Holm-adjusted bound of < 12 × 0.002 = 0.024 (corrected 2026-11-03 from "< 0.002"; no verdict changes, every such value is below 0.05).

| Metric | Better | ConvNeXt-T | Local | Local − ConvNeXt-T [95% CI] | p (Holm) | Fused | Fused − ConvNeXt-T [95% CI] | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| deletion_vs_random (P) | lower | −0.078 | −0.028 | +0.051 [+0.041, +0.061] | < 0.002 (< 0.024) | −0.075 | +0.003 [−0.007, +0.012] | 0.64 (0.64) |
| insertion_vs_random (P) | higher | +0.058 | +0.032 | −0.027 [−0.034, −0.020] | < 0.002 (< 0.024) | +0.048 | −0.011 [−0.017, −0.004] | 0.002 (0.008) |
| rand_full_ssim (P) | lower | 0.268 | 0.262 | −0.006 [−0.015, +0.002] | 0.18 (0.36) | 0.292 | +0.023 [+0.011, +0.035] | < 0.002 (< 0.024) |
| stab_spearman_mean (P) | higher | 0.938 | 0.707 | −0.232 [−0.246, −0.218] | < 0.002 (< 0.024) | 0.892 | −0.047 [−0.053, −0.041] | < 0.002 (< 0.024) |
| iou (P) | higher | 0.315 | 0.499 | +0.184 [+0.169, +0.199] | < 0.002 (< 0.024) | 0.437 | +0.122 [+0.110, +0.133] | < 0.002 (< 0.024) |
| iou_gap (P) | higher | −0.032 | +0.084 | +0.115 [+0.076, +0.156] | < 0.002 (< 0.024) | −0.071 | −0.040 [−0.080, +0.002] | 0.068 (0.20) |
| deletion_auc | lower | 0.438 | 0.527 | +0.089 [+0.079, +0.099] | < 0.002 | 0.479 | +0.041 [+0.031, +0.051] | < 0.002 |
| insertion_auc | higher | 0.711 | 0.632 | −0.079 [−0.088, −0.071] | < 0.002 | 0.648 | −0.063 [−0.071, −0.055] | < 0.002 |
| rand_full_abs_spearman | lower | 0.237 | 0.215 | −0.022 [−0.035, −0.009] | < 0.002 | 0.194 | −0.043 [−0.054, −0.030] | < 0.002 |
| rand_full_abs_hog | lower | 0.082 | 0.058 | −0.024 [−0.029, −0.020] | < 0.002 | 0.057 | −0.026 [−0.030, −0.021] | < 0.002 |
| stab_ssim_mean | higher | 0.926 | 0.702 | −0.223 [−0.232, −0.215] | < 0.002 | 0.975 | +0.050 [+0.047, +0.052] | < 0.002 |
| energy_in_mask | higher | 0.525 | 0.716 | +0.191 [+0.175, +0.206] | < 0.002 | 0.459 | −0.066 [−0.075, −0.057] | < 0.002 |
| pointing | higher | 0.906 | 0.890 | −0.016 [−0.038, +0.005] | 0.16 | 0.848 | −0.057 [−0.080, −0.036] | < 0.002 |

- **Reference maps** (deletion / insertion AUC): ConvNeXt-T random 0.516 / 0.653, centre 0.514 / 0.695; hybrid random 0.555 / 0.601, centre 0.515 / 0.624.
- **Predictions changed by each perturbation:**
  - ConvNeXt-T: flip 2.9%, noise 17.6%, brightness ×0.9 1.0%, ×1.1 1.2%.
  - Hybrid: 5.1%, 19.5%, 2.2%, 1.9%.
  - On average 3.77 (ConvNeXt-T) and 3.71 (hybrid) of the 4 perturbations are kept; no image lost its stability value.
- **Wrong predictions** (for `iou_gap`): ConvNeXt-T 191 (matches `eval/metrics.json`, accuracy 0.8729), hybrid 198.

### Verdict under the frozen decision rule
- **The claim "the hybrid's explanations are more trustworthy than ConvNeXt-T's" is not supported, for either hybrid map.**
  - **EfficientNetV2-branch (local) Grad-CAM:** significantly *worse* on both faithfulness endpoints (deletion +0.051, insertion −0.027 vs random). No difference on the sanity endpoint (`rand_full_ssim`, Holm 0.36). Much less stable (Spearman 0.71 vs 0.94).
  - **Fused-token Grad-CAM:** no difference on deletion (Holm 0.64); significantly worse on insertion (Holm 0.008), on `rand_full_ssim` (more similar to the trained map after full randomisation) and on stability (0.89 vs 0.94).
- **Where the hybrid is better is localisation against the lesion masks:**
  - IoU is higher for both maps (+0.184 local, +0.122 fused).
  - The local map's IoU drops when the model is wrong (`iou_gap` +0.084), while ConvNeXt-T's does not (−0.032). This is the only trust property on which a hybrid map is clearly ahead, and it replicates the val result (+0.169 val, +0.115 test).
  - Caveat (recorded before the test run): the fused map's IoU gain comes with lower energy in mask (−0.066) and pointing (−0.057), consistent with a more diffuse map that the 15% threshold rewards on large lesions. Its stability SSIM (0.975) is above its Spearman (0.892), the flat-map effect.
- **Replication of val:** every primary endpoint has the same direction as on val, and every significant val result stays significant.
  - Fused `iou_gap` moved from +0.003 to −0.040 (Holm 0.20, still not significant).
  - The local map's pointing difference (exploratory) is no longer significant (p 0.16 vs 0.028 on val).
- **Exploratory randomisation results point the other way to SSIM:** both hybrid maps have lower |Spearman| and |HOG| after full randomisation than ConvNeXt-T. So the sanity verdict depends on the similarity measure. With SSIM as the pre-registered endpoint, the fused map fails and the local map ties.
- **Limits:**
  - Single training run per model, single seed; the CIs cover test-image variation only.
  - All maps are Grad-CAM on 7×7 grids; other attribution methods (attention, rollout) were not part of this comparison.
  - Faithfulness uses one perturbation baseline per mode (mean colour, blur). Deletion / insertion are known to depend on that choice (Petsiuk et al. 2018, external).
- **Seen here for the first time:** the trust run predicts the hybrid's HAM10000 test labels, so its test accuracy follows (198 wrong → 1,305 / 1,503 = 0.868). `evaluate.py` on the hybrid's HAM10000 test split has not been run yet; that number is informal until it is, and nothing about the hybrid was or will be changed in response.

## Part B (EXPLORATORY): can the hybrid's explanations become more trustworthy? Protocol frozen 2026-11-02
- **Everything in Part B is exploratory.** The HAM10000 test set was already used once by M4 (and by the classification test above). The M4 verdict stays as it is; nothing in Part B can change it.
- **Frozen before any Part B val run:** the settings below, the configs `configs/xai/trust_smoothgrad.yaml`, `trust_scorecam.yaml`, `trust_mask_screen.yaml`, `trust_mask_final.yaml`, `configs/train/mask_finetune.yaml`, and the decision rules. Then val, then test once.
- **Smoke runs before freezing, val only** (folders `trust_val_smoothgrad_smoke/`, `trust_val_scorecam_smoke/`, run `smoke_ham10000_hybrid_mask`): 8–34 val images per model, to check the code and measure speed.
  - Two smoke comparison logs were printed (34 and 11 val images) and then deleted. No setting was changed in response to their values.
  - Code changes after the smoke runs: SmoothGrad noise is drawn on the GPU, not copied from the CPU (same distribution, different random numbers; for speed); a metric with no valid resample (e.g. `iou_gap` when a model has no wrong prediction on a tiny subset) now reports NaN and is left out of Holm, instead of crashing.
- **Same trust metrics and paired comparison as M4** (`eval_trust.py`, `compare_xai_paired.py`), with these differences, all fixed in the configs:
  - Localisation (IoU, energy in mask, pointing) is scored inside `eval_trust.py` with the `configs/xai/default.yaml` protocol of `eval_xai.py`.
  - Randomisation is one step (`randomisation: full`): every cascade block re-initialised at once. This is the end state of the M4 cascade, the only randomisation state with a primary endpoint (`rand_full_ssim`); the random draws differ from M4's.
  - Image seeds (random map, stability noise, SmoothGrad noise) come from the image's position in the full split, so a subset run uses the same seeds per image as a full run.
  - Exploratory addition: `stab_spearman_no_noise`, stability Spearman over the flip and brightness perturbations only, because SmoothGrad averages over input noise by design and the noise perturbation would favour it.

### Step 4: other explanation methods, no retraining (`xai/cam_variants.py`)
- **SmoothGrad-CAM** (Smilkov et al. 2017, SmoothGrad; external): Grad-CAM averaged over 50 noisy copies of the input, noise std 0.15 × (max − min) of the input tensor. Both settings come from the code of Adebayo et al. 2018 (`Related_papers/Sanity Checks for Saliency Maps/code/src/randomize_inceptionv3.py:78-82`). The target class is resolved on the clean image; post-ReLU maps are averaged. All 1,503 images per split.
- **Score-CAM** (Wang et al. 2020; external): each channel of the layer, upsampled and min-max scaled, masks the normalised input (masked pixels take the dataset-mean colour, the deletion-curve fill). The channel weight is the softmax probability of the target class for that masked input; map = ReLU(Σ w_k A_k). Constant channels get weight 0.
  - Fused map: the 2 × 384 channels of both fusion token streams together, as Grad-CAM's fused sum.
  - One forward pass per channel (ConvNeXt-T 768, hybrid local 1,280 + fused 768), so a **fixed stratified subset of 150 images per split** (seed 42, the same images for both runs): per class akiec 5, bcc 8, bkl 16, df 2, mel 17, nv 100, vasc 2. With 2 df and 2 vasc images, per-class statements are impossible on this subset.
- Same layers and map names as Grad-CAM (`gradcam` for ConvNeXt-T, `gradcam_local` / `gradcam_fused` for the hybrid).
- **Comparisons per method, each Holm-corrected over its own 6 primary endpoints × maps:**
  - (a) hybrid map vs ConvNeXt-T map, both with the method: is the hybrid more trustworthy under this method?
  - (b) hybrid: the method vs its own Grad-CAM (the M4 `trust_<split>/` run; Score-CAM: the same 150 images): does the method make the hybrid's maps more trustworthy?
  - (c) ConvNeXt-T: the method vs its own Grad-CAM, for context on (b).
- **Decision rule, the M4 rule applied to (a) and (b) separately:** B counts as more trustworthy than A only if it is significantly better (Holm p < 0.05) on `deletion_vs_random` or `insertion_vs_random` and on `rand_full_ssim`, and significantly worse on no other primary endpoint. (c) is descriptive.
- **Measured cost** (smoke runs; the machine stalls when RAM runs short, so these are lower bounds): SmoothGrad about 1.6 h ConvNeXt-T + 3.6 h hybrid ≈ 5 h per split; Score-CAM about 0.6 h + 5 h ≈ 5.5 h per split. My earlier estimates (2.3 h and 2.8 h) were too low.

#### Step 4 val results (EXPLORATORY; 2026-11-02 18:15 – 2026-11-03 03:17, `outputs/logs/partb_trust_val_memory.log`)
- Wall time: SmoothGrad-CAM 161 min ConvNeXt-T + 122 min hybrid; Score-CAM 24 min + 156 min. All trust runs and comparisons (a)–(c) finished without errors.
- Files: `outputs/runs/<run>/trust_val_smoothgrad/`, `trust_val_scorecam/`; `outputs/comparisons/ham10000_xai_val_*-smoothgrad_vs_*.json`, `*-scorecam_vs_*.json`.
- **Primary endpoints, means.** SmoothGrad-CAM on all 1,503 val images; Score-CAM on the fixed 150-image subset, with the Grad-CAM values of the same 150 images beside it (from the M4 `trust_val/` run):

| Primary endpoint (better) | Map | Grad-CAM (n 1,503) | SmoothGrad-CAM (n 1,503) | Grad-CAM (n 150) | Score-CAM (n 150) |
|---|---|---|---|---|---|
| `deletion_vs_random` (lower) | ConvNeXt-T | −0.0841 | −0.0507 | −0.0718 | +0.0072 |
| | hybrid local | −0.0411 | −0.0401 | −0.0395 | −0.0609 |
| | hybrid fused | −0.0863 | −0.0733 | −0.0714 | −0.0707 |
| `insertion_vs_random` (higher) | ConvNeXt-T | 0.0652 | 0.0327 | 0.0627 | 0.0563 |
| | hybrid local | 0.0313 | 0.0228 | 0.0193 | 0.0263 |
| | hybrid fused | 0.0483 | 0.0313 | 0.0434 | 0.0405 |
| `rand_full_ssim` (lower) | ConvNeXt-T | 0.2665 | 0.4175 | 0.2588 | 0.4024 |
| | hybrid local | 0.2643 | 0.1783 | 0.2786 | 0.3749 |
| | hybrid fused | 0.2939 | 0.3020 | 0.3008 | 0.3602 |
| `stab_spearman_mean` (higher) | ConvNeXt-T | 0.9377 | 0.9769 | 0.9429 | 0.7562 |
| | hybrid local | 0.7005 | 0.8663 | 0.7097 | 0.9715 |
| | hybrid fused | 0.8900 | 0.9282 | 0.9047 | 0.8029 |
| `iou` (higher) | ConvNeXt-T | 0.3110 | 0.3514 | 0.3034 | 0.1690 |
| | hybrid local | 0.4898 | 0.5120 | 0.4876 | 0.5533 |
| | hybrid fused | 0.4224 | 0.3916 | 0.4118 | 0.4258 |
| `iou_gap` (higher) | ConvNeXt-T | −0.0488 | −0.0090 | −0.0328 | −0.2281 |
| | hybrid local | 0.1197 | 0.0444 | 0.0530 | −0.0317 |
| | hybrid fused | −0.0456 | 0.0312 | −0.1603 | −0.1128 |

- **Paired differences, B − A** (1,000 paired resamples; Holm-adjusted p in brackets, Holm over the 6 primary endpoints × maps of each comparison; \*\* = no resample crossed zero, p < 0.002, Holm < 0.024 for (a) and (b) (12 tests) and < 0.012 for (c) (6 tests); corrected 2026-11-03 from "Holm < 0.012" for all, no verdict changes). (a) hybrid vs ConvNeXt-T, both with the method; (b) hybrid, method vs its own Grad-CAM; (c) ConvNeXt-T, method vs its own Grad-CAM (descriptive):

| Primary endpoint (better) | SG (a) local | SG (a) fused | SG (b) local | SG (b) fused | SG (c) | SC (a) local | SC (a) fused | SC (b) local | SC (b) fused | SC (c) |
|---|---|---|---|---|---|---|---|---|---|---|
| `deletion_vs_random` (lower) | +0.0106 (0.144) | −0.0226 \*\* better | +0.0010 (0.740) | +0.0130 \*\* worse | +0.0334 \*\* worse | −0.0681 \*\* better | −0.0779 \*\* better | −0.0214 (0.084) | +0.0008 (1.000) | +0.0790 \*\* worse |
| `insertion_vs_random` (higher) | −0.0098 (0.010) worse | −0.0013 (0.668) | −0.0085 \*\* worse | −0.0170 \*\* worse | −0.0326 \*\* worse | −0.0301 (0.060) | −0.0159 (0.414) | +0.0069 (1.000) | −0.0029 (1.000) | −0.0063 (0.432) |
| `rand_full_ssim` (lower) | −0.2392 \*\* better | −0.1156 \*\* better | −0.0860 \*\* better | +0.0081 (0.060) | +0.1510 \*\* worse | −0.0275 (0.420) | −0.0422 (0.312) | +0.0963 \*\* worse | +0.0594 \*\* worse | +0.1436 \*\* worse |
| `stab_spearman_mean` (higher) | −0.1106 \*\* worse | −0.0487 \*\* worse | +0.1657 \*\* better | +0.0382 \*\* better | +0.0392 \*\* better | +0.2153 \*\* better | +0.0467 (0.140) | +0.2618 \*\* better | −0.1018 \*\* worse | −0.1867 \*\* worse |
| `iou` (higher) | +0.1605 \*\* better | +0.0402 \*\* better | +0.0221 \*\* better | −0.0308 \*\* worse | +0.0405 \*\* better | +0.3843 \*\* better | +0.2568 \*\* better | +0.0657 \*\* better | +0.0140 (1.000) | −0.1345 \*\* worse |
| `iou_gap` (higher) | +0.0534 (0.056) | +0.0402 (0.144) | −0.0753 (0.012) worse | +0.0768 \*\* better | +0.0398 (0.002) better | +0.1964 (0.028) better | +0.1153 (0.420) | −0.0846 (0.720) | +0.0475 (1.000) | −0.1952 \*\* worse |
| **Frozen rule** | not met | not met | not met | not met | (descriptive) | not met | not met | not met | not met | (descriptive) |

- **Under the frozen decision rule, none of the eight comparisons (a) and (b) is met on val:**
  - SmoothGrad (a) local: worse on insertion, no difference on deletion. (a) fused: better on deletion and `rand_full_ssim`, but significantly *worse* on stability, so it fails the "worse on no other primary endpoint" condition. This is the closest case in Part B so far.
  - SmoothGrad (b): local no better on deletion, worse on insertion; fused worse on both.
  - Score-CAM (a): both maps better on deletion, but no difference on `rand_full_ssim`. (b): no faithfulness gain, and worse on `rand_full_ssim` for both maps.
- **Observations (val, exploratory):**
  - SmoothGrad-CAM's wins in (a) come largely from ConvNeXt-T getting worse: its `rand_full_ssim` rose from 0.267 to 0.418 with SmoothGrad (c), so the randomised-model maps look more like the trained-model maps. Noise averaging seems to smooth both towards the same blob.
  - Score-CAM on ConvNeXt-T performs badly on this subset: `deletion_vs_random` +0.0072 (no better than a random map), pointing 0.427 (Grad-CAM 0.873), IoU 0.169.
  - **Diagnostic after the val results (2026-11-03, val only, nothing changed;** scripts and figure in the session scratchpad, not part of the project):
    - On 10 subset images, Score-CAM was all zero for 3 and almost zero for 1; ConvNeXt-T's Grad-CAM was never blank. On the 150 subset images, 27 Score-CAM maps were all zero (Grad-CAM 0).
    - Cause: ConvNeXt-T's hooked layer (`stages[-1]` output, no activation after it; the head normalises after pooling) is signed, about 51% of its activations negative. The Score-CAM weights (softmax probabilities of the masked inputs) are all positive and nearly equal (spread typically 4–13% of their mean), so the map is close to ReLU of the plain channel mean (r ≈ 0.95 where non-zero) and is blank wherever that mean is negative. A blank map puts its maximum on a corner pixel (pointing miss, near-zero IoU) and gives deletion no order, hence random-like.
    - This is Score-CAM's linear combination of raw activations meeting a signed layer, not a coding error in the formula as implemented; Score-CAM as usually applied combines post-ReLU convolutional activations. Grad-CAM is not affected because its gradient weights can be negative.
    - The hybrid's fused map is hardly affected: 1 of 150 maps all zero (fusion token streams about 49% negative, Grad-CAM 0 blank). The hybrid's local map was not counted (its layer is after SiLU, whose negative values are small; 85% of values are negative but bounded by −0.28).
    - **Consequence:** the ConvNeXt-T side of Score-CAM comparison (a) and comparison (c) mostly measure this blank-map failure, not Score-CAM's quality on a well-suited layer. The hybrid-only comparison (b) is not affected by it.
  - Score-CAM's local hybrid map hardly changes under full randomisation by rank (`rand_full_abs_spearman` 0.828 vs 0.205 for Grad-CAM), a strong sanity-check failure on a secondary measure.
  - Score-CAM results rest on 150 images, with 2 df and 2 vasc images, so they are less precise than SmoothGrad's.
- **No setting was changed after these results.** Test run once: `partb_trust.ps1 -Split test`.

#### Step 4 test results (EXPLORATORY, run once, 2026-11-03 06:47–12:37, `outputs/logs/partb_trust_test_memory.log`)
- Settings exactly as frozen; the Score-CAM blank-map diagnosis above was written before the test run, and nothing was changed after it (no fix applied). Wall time: SmoothGrad-CAM 57 min ConvNeXt-T + 122 min hybrid; Score-CAM 24 min + 146 min. All trust runs and comparisons finished without errors.
- Files: `outputs/runs/<run>/trust_test_smoothgrad/`, `trust_test_scorecam/`; `outputs/comparisons/ham10000_xai_test_*-smoothgrad_vs_*.json`, `*-scorecam_vs_*.json`.
- **Primary endpoints, test means** (Grad-CAM from the M4 `trust_test/` runs; Score-CAM on the fixed stratified 150-image test subset, same class counts as val):

| Primary endpoint (better) | Map | Grad-CAM (n 1,503) | SmoothGrad-CAM (n 1,503) | Grad-CAM (n 150) | Score-CAM (n 150) |
|---|---|---|---|---|---|
| `deletion_vs_random` (lower) | ConvNeXt-T | −0.0782 | −0.0525 | −0.0588 | +0.0408 |
| | hybrid local | −0.0276 | −0.0222 | −0.0210 | −0.0439 |
| | hybrid fused | −0.0754 | −0.0552 | −0.0692 | −0.0487 |
| `insertion_vs_random` (higher) | ConvNeXt-T | 0.0584 | 0.0290 | 0.0766 | 0.0636 |
| | hybrid local | 0.0315 | 0.0201 | 0.0227 | 0.0203 |
| | hybrid fused | 0.0477 | 0.0307 | 0.0420 | 0.0482 |
| `rand_full_ssim` (lower) | ConvNeXt-T | 0.2684 | 0.4188 | 0.2578 | 0.4298 |
| | hybrid local | 0.2622 | 0.1809 | 0.2470 | 0.3665 |
| | hybrid fused | 0.2917 | 0.2920 | 0.2855 | 0.3494 |
| `stab_spearman_mean` (higher) | ConvNeXt-T | 0.9382 | 0.9762 | 0.9367 | 0.7101 |
| | hybrid local | 0.7066 | 0.8630 | 0.6721 | 0.9678 |
| | hybrid fused | 0.8916 | 0.9320 | 0.8826 | 0.7442 |
| `iou` (higher) | ConvNeXt-T | 0.3147 | 0.3501 | 0.3157 | 0.1598 |
| | hybrid local | 0.4986 | 0.5209 | 0.4917 | 0.5720 |
| | hybrid fused | 0.4369 | 0.4057 | 0.4287 | 0.4006 |
| `iou_gap` (higher) | ConvNeXt-T | −0.0315 | −0.0038 | −0.0282 | −0.1794 |
| | hybrid local | 0.0836 | 0.0601 | 0.0886 | −0.0144 |
| | hybrid fused | −0.0712 | −0.0036 | −0.0250 | −0.0195 |

- **Paired differences B − A, val / test.** ▲ = significantly better, ▼ = significantly worse (Holm p < 0.05 within each comparison), no mark = not significant. Comparisons (a), (b), (c) as defined above:

| Comparison | `deletion_vs_random` | `insertion_vs_random` | `rand_full_ssim` | `stab_spearman_mean` | `iou` | `iou_gap` | Rule val / test |
|---|---|---|---|---|---|---|---|
| SG (a) local | +0.011 / ▼+0.030 | ▼−0.010 / ▼−0.009 | ▲−0.239 / ▲−0.238 | ▼−0.111 / ▼−0.113 | ▲+0.161 / ▲+0.171 | +0.053 / ▲+0.064 | not met / not met |
| SG (a) fused | ▲−0.023 / −0.003 | −0.001 / +0.002 | ▲−0.116 / ▲−0.127 | ▼−0.049 / ▼−0.044 | ▲+0.040 / ▲+0.056 | +0.040 / +0.000 | not met / not met |
| SG (b) local | +0.001 / +0.005 | ▼−0.009 / ▼−0.011 | ▲−0.086 / ▲−0.081 | ▲+0.166 / ▲+0.156 | ▲+0.022 / ▲+0.022 | ▼−0.075 / −0.023 | not met / not met |
| SG (b) fused | ▼+0.013 / ▼+0.020 | ▼−0.017 / ▼−0.017 | +0.008 / +0.000 | ▲+0.038 / ▲+0.040 | ▼−0.031 / ▼−0.031 | ▲+0.077 / ▲+0.068 | not met / not met |
| SG (c) ConvNeXt-T | ▼+0.033 / ▼+0.026 | ▼−0.033 / ▼−0.029 | ▼+0.151 / ▼+0.150 | ▲+0.039 / ▲+0.038 | ▲+0.041 / ▲+0.035 | ▲+0.040 / ▲+0.028 | descriptive |
| SC (a) local | ▲−0.068 / ▲−0.085 | −0.030 / ▼−0.043 | −0.027 / ▲−0.063 | ▲+0.215 / ▲+0.258 | ▲+0.384 / ▲+0.412 | ▲+0.196 / ▲+0.165 | not met / not met |
| SC (a) fused | ▲−0.078 / ▲−0.090 | −0.016 / −0.015 | −0.042 / ▲−0.080 | +0.047 / +0.034 | ▲+0.257 / ▲+0.241 | +0.115 / +0.160 | not met / **met** |
| SC (b) local | −0.021 / ▲−0.023 | +0.007 / −0.002 | ▼+0.096 / ▼+0.119 | ▲+0.262 / ▲+0.296 | ▲+0.066 / ▲+0.080 | −0.085 / −0.103 | not met / not met |
| SC (b) fused | +0.001 / +0.021 | −0.003 / +0.006 | ▼+0.059 / +0.064 | ▼−0.102 / ▼−0.138 | +0.014 / −0.028 | +0.047 / +0.005 | not met / not met |
| SC (c) ConvNeXt-T | ▼+0.079 / ▼+0.100 | −0.006 / −0.013 | ▼+0.144 / ▼+0.172 | ▼−0.187 / ▼−0.227 | ▼−0.134 / ▼−0.156 | ▼−0.195 / ▼−0.151 | descriptive |

  SG = SmoothGrad-CAM (n 1,503), SC = Score-CAM (n 150). Third decimals as generated by `scripts/compare_results.py` from the unrounded values (7 cells corrected 2026-11-03 by 0.001, from rounding the 4-decimal log values). Holm-adjusted p of the test cells near 0.05: SG (a) local insertion 0.048; SC (a) local `rand_full_ssim` 0.024, `iou_gap` 0.048; SC (a) fused `rand_full_ssim` 0.024, insertion 0.284, stability 0.284, `iou_gap` 0.192; SC (b) local deletion 0.032; SC (b) fused deletion 0.190, `rand_full_ssim` 0.056; SC (c) `iou_gap` 0.032.
- **Verdict under the frozen decision rule (test):**
  - SmoothGrad-CAM: not met in any of the four comparisons. (a) fused, the closest case on val, loses its deletion advantage on test (−0.003, Holm 1.000) and is still worse on stability.
  - Score-CAM: not met in (a) local (better on deletion and `rand_full_ssim`, but worse on insertion), (b) local (better on deletion, worse on `rand_full_ssim`) and (b) fused (no faithfulness gain, worse on stability).
  - **Score-CAM (a) fused formally meets the rule on test:** better on deletion (−0.090, Holm < 0.024) and `rand_full_ssim` (−0.080, Holm 0.024), and significantly worse on no primary endpoint. It did not meet it on val (`rand_full_ssim` not significant, Holm 0.312).
  - **This one formal pass is not evidence that the hybrid's fused map is more trustworthy.** Its reference, ConvNeXt-T Score-CAM, is broken by the blank-map failure diagnosed on val before the test run (below): on test it is worse than a random map on deletion (`deletion_vs_random` +0.041) and its `rand_full_ssim` (0.430) is far above ConvNeXt-T Grad-CAM's (0.258). The pass is one of eight exploratory rule checks on test, on 150 images, and was not reproduced on val.
- **Blank-map check on test** (same diagnostic script, ConvNeXt-T, the 150-image test subset; run after the test results): 24 of 150 Score-CAM maps all zero (val 27/150); Grad-CAM 0/150; 51.4% of the hooked activations negative (val 51.4%). The failure appears on test as on val, and ConvNeXt-T Score-CAM's test pointing (0.407) and IoU (0.160) are as low as on val.
- **Observations (exploratory, val and test agree unless stated):**
  - SmoothGrad-CAM makes the hybrid's local map more stable (0.71 → 0.86) and less similar to its randomised-model map (`rand_full_ssim` 0.262 → 0.181), but less faithful on insertion; on the fused map it costs faithfulness on both curves and changes `rand_full_ssim` little. On ConvNeXt-T it worsens faithfulness and `rand_full_ssim` (0.268 → 0.419), so SmoothGrad's apparent wins for the hybrid in (a) on `rand_full_ssim` come mostly from ConvNeXt-T getting worse.
  - Score-CAM's local hybrid map is highly stable (0.97) and well localised (IoU 0.572, pointing 0.973 on test), but it hardly changes under full randomisation by rank (`rand_full_abs_spearman` 0.823 on test, 0.828 on val; Grad-CAM about 0.21), a strong sanity-check failure on a secondary measure, and its `rand_full_ssim` is worse than Grad-CAM's on both splits.
  - No method tried in step 4 makes either hybrid map more trustworthy than its own Grad-CAM under the frozen rule (comparison (b): 0 of 4 on val, 0 of 4 on test).

### Step 5: mask-guided fine-tuning (`xai/mask_guidance.py`, `train/mask_trainer.py`, `scripts/finetune_mask.py`)
- **Map trained on: the fused CAM.** The hybrid's head is linear on the token means of the two fusion streams, so the fused Grad-CAM is exactly CAM / 49 (class activation mapping, Zhou et al. 2016; external) and can be trained at forward-pass cost. Checked by `tests/test_trust.py::test_fused_cam_equals_fused_gradcam_and_mask_losses_by_hand`.
- **Loss:** cross-entropy (the run's own, class-weighted, label smoothing 0.1) + λ · outside-mask share + μ · consistency.
  - Outside-mask share: Σ ReLU(CAM_y) · (1 − m) / Σ ReLU(CAM_y), with CAM_y the true-class map and m the lesion mask area-pooled to 7×7. At the start of the smoke run it was 0.64 (64% of the fused map outside the lesion).
  - Consistency: 1 − cosine similarity of ReLU(CAM_y) for the image and for the image + Gaussian noise (std 0.03 on the 0–1 pixel scale, as in the stability check).
  - The idea is "right for the right reasons" (Ross et al. 2017, external) applied to a CAM rather than input gradients; supervising a CAM-type map is the approach of GAIN (Li et al. 2018, external).
- **Masks from the train split only**; val is loaded without masks.
- **Recipe** (`configs/train/mask_finetune.yaml` on `default.yaml` + `local_4gb.yaml`): from `ham10000_hybrid/best.pt` (epoch 10), 5 epochs, backbone lr 1e-5, fusion + head lr 1e-4, no warmup, cosine, EMA 0.999, best.pt = best val macro-F1 epoch.
- **Candidates and selection, val only** (`scripts/select_mask_candidate.py`, refuses if a candidate has been evaluated on test):
  1. λ ∈ {0.1, 0.3, 1} with μ = 0 (3 runs).
  2. The λ chosen in step 1, with μ ∈ {0.1, 1} (2 runs); the step-1 winner stays a candidate.
  - **Eligible:** best val macro-F1 ≥ 0.8289 (the source's 0.8389 − 0.01). This is the "classification does not drop" check.
  - **Chosen:** the highest val `insertion_vs_random` of the fused map (rounded to 3 decimals, ties broken by `stab_spearman_mean`), on a fixed stratified 300-image val subset (`trust_mask_screen.yaml`: Grad-CAM, one-step randomisation; akiec 10, bcc 15, bkl 33, df 4, mel 33, nv 201, vasc 4). Insertion is where both hybrid maps failed in M4. The source hybrid is screened too, for reference.
  - If no candidate is eligible, step 5 stops there and that is the result.
- **Then, for the chosen run only:** full val trust run with the M4 protocol (`trust_mask_final.yaml`: Grad-CAM, cascade, all images, localisation inside `eval_trust.py`), then on test once: the trust run, `evaluate.py`, and the paired classification test vs the source hybrid.
- **Comparisons:** (d) chosen run vs the source hybrid, map by map (local vs local, fused vs fused); (e) chosen run vs ConvNeXt-T Grad-CAM. Same decision rule as M4 for both.
- **Classification on test** (paired, `compare_paired.py`): reported; the fine-tuned model is called "not worse" only if the lower 95% CI bound of its macro-F1 difference to the source is above −0.01.
- **Measured cost:** about 45 min per run with μ = 0 and about 70 min with μ > 0 (smoke: GPU peak 1.52 GB with consistency), about 10 min per screening, about 1 h for the final val trust run: ≈ 6.5 h for the val stage, ≈ 1.2 h for the test stage.
#### Step 5 val results (EXPLORATORY; run 2026-11-02, `outputs/logs/mask_guided_val_memory.log`, about 3.6 h)
- **Selection** (`outputs/comparisons/ham10000_mask_guided_lambda.json`, `_consistency.json`; val macro-F1 = best epoch, screen = 300-image val subset, fused map):

| Run | Val macro-F1 | Eligible (≥ 0.8289) | Screen `insertion_vs_random` |
|---|---|---|---|
| `ham10000_hybrid` (source, reference) | 0.8389 | – | 0.047 |
| λ 0.1, μ 0 | 0.8367 | yes | 0.049 |
| λ 0.3, μ 0 | 0.8257 | no | 0.045 |
| λ 1, μ 0 | 0.8166 | no | 0.041 |
| λ 0.1, μ 0.1 | 0.8102 | no | 0.059 |
| λ 0.1, μ 1 | 0.8143 | no | 0.036 |

  - **Chosen: `ham10000_hybrid_mask_l0.1_m0`**, the only eligible run in both stages. The run with the highest screened insertion (μ 0.1) was excluded by the macro-F1 rule, as frozen.
- **Paired comparisons, full val** (n = 1,503, 1,000 paired resamples; `outputs/comparisons/ham10000_xai_val_ham10000_hybrid_mask_l0.1_m0_vs_*.json`). B − A, Holm-adjusted p in brackets; \*\* = no resample crossed zero (p < 0.002, Holm < 12 × 0.002 = 0.024; corrected 2026-11-03 from "< 0.012", no verdict changes):

| Primary endpoint (better) | ConvNeXt-T | Hybrid local | Mask local | Hybrid fused | Mask fused | (d) local | (d) fused | (e) local | (e) fused |
|---|---|---|---|---|---|---|---|---|---|
| `deletion_vs_random` (lower) | −0.0841 | −0.0411 | −0.0488 | −0.0863 | −0.0656 | −0.0077 (0.128) | +0.0207 \*\* worse | +0.0353 \*\* worse | +0.0185 \*\* worse |
| `insertion_vs_random` (higher) | 0.0652 | 0.0313 | 0.0403 | 0.0483 | 0.0503 | +0.0089 \*\* better | +0.0020 (0.668) | −0.0249 \*\* worse | −0.0149 \*\* worse |
| `rand_full_ssim` (lower) | 0.2665 | 0.2643 | 0.2813 | 0.2939 | 0.2981 | +0.0170 \*\* worse | +0.0042 (0.294) | +0.0148 \*\* worse | +0.0316 \*\* worse |
| `stab_spearman_mean` (higher) | 0.9377 | 0.7005 | 0.8647 | 0.8900 | 0.9557 | +0.1642 \*\* better | +0.0657 \*\* better | −0.0731 \*\* worse | +0.0180 \*\* better |
| `iou` (higher) | 0.3110 | 0.4898 | 0.5456 | 0.4224 | 0.5257 | +0.0558 \*\* better | +0.1033 \*\* better | +0.2347 \*\* better | +0.2147 \*\* better |
| `iou_gap` (higher) | −0.0488 | 0.1197 | 0.1034 | −0.0456 | 0.0126 | −0.0163 (0.668) | +0.0582 (0.040) better | +0.1522 \*\* better | +0.0615 (0.004) better |

- **Under the frozen decision rule, none of the four comparisons is met on val:**
  - (d) local: better on insertion, but significantly *worse* on `rand_full_ssim`.
  - (d) fused: significantly worse on deletion; no difference on insertion or `rand_full_ssim`.
  - (e) both maps: significantly worse than ConvNeXt-T on deletion, insertion and `rand_full_ssim` (local also on stability).
- **Observations (val, exploratory):** the fine-tuning mainly changed localisation and stability, not faithfulness.
  - Fused map: IoU 0.422 → 0.526, energy in mask 0.446 → 0.790, pointing 0.849 → 0.964, `iou_gap` turned positive (−0.046 → +0.013).
  - Stability Spearman rose for both maps (local 0.70 → 0.86, fused 0.89 → 0.96), but the secondary fused stability SSIM fell (0.975 → 0.919).
  - Randomisation is mixed: `rand_full_ssim` worse, while the secondary |ρ| measures `rand_full_abs_spearman` and `rand_full_abs_hog` were lower (better) for the local map and lower or unchanged for the fused map; the M4 SSIM caveat applies.
- **Selection-bias caveat:** the run was chosen on fused `insertion_vs_random` of the 300-image screening subset, which is part of the 1,503 val images used above, and the λ/μ grid and checkpoint were also chosen on val. The val comparisons of the chosen run are therefore optimistic, most of all for fused insertion — and even that difference is not significant (+0.0020, Holm 0.668). Only the single test run below is free of this selection.
- **Test run:** once, `mask_guided.ps1 -Stage test`, with the rule and the "not worse" classification criterion above unchanged.

#### Step 5 test results (EXPLORATORY, run once, 2026-11-02 17:01–18:04, `outputs/logs/mask_guided_test_memory.log`)
- Run `ham10000_hybrid_mask_l0.1_m0`, chosen on val above; nothing was changed after the val results or after these results were seen. Reference values for the source hybrid and ConvNeXt-T are their M4 test trust runs.
- **Paired comparisons** (test n = 1,503, 1,000 paired resamples; `outputs/comparisons/ham10000_xai_test_ham10000_hybrid_mask_l0.1_m0_vs_*.json`). B − A, Holm-adjusted p in brackets; \*\* = no resample crossed zero (p < 0.002, Holm < 12 × 0.002 = 0.024; corrected 2026-11-03 from "< 0.012", no verdict changes):

| Primary endpoint (better) | ConvNeXt-T | Hybrid local | Mask local | Hybrid fused | Mask fused | (d) local | (d) fused | (e) local | (e) fused |
|---|---|---|---|---|---|---|---|---|---|
| `deletion_vs_random` (lower) | −0.0782 | −0.0276 | −0.0396 | −0.0754 | −0.0554 | −0.0120 \*\* better | +0.0201 \*\* worse | +0.0386 \*\* worse | +0.0228 \*\* worse |
| `insertion_vs_random` (higher) | 0.0584 | 0.0315 | 0.0398 | 0.0477 | 0.0514 | +0.0083 \*\* better | +0.0037 (0.208) | −0.0186 \*\* worse | −0.0070 (0.092) |
| `rand_full_ssim` (lower) | 0.2684 | 0.2622 | 0.2804 | 0.2917 | 0.2980 | +0.0182 \*\* worse | +0.0063 (0.080) | +0.0119 (0.006) worse | +0.0295 \*\* worse |
| `stab_spearman_mean` (higher) | 0.9382 | 0.7066 | 0.8686 | 0.8916 | 0.9558 | +0.1620 \*\* better | +0.0643 \*\* better | −0.0696 \*\* worse | +0.0176 \*\* better |
| `iou` (higher) | 0.3147 | 0.4986 | 0.5509 | 0.4369 | 0.5258 | +0.0523 \*\* better | +0.0889 \*\* better | +0.2362 \*\* better | +0.2111 \*\* better |
| `iou_gap` (higher) | −0.0315 | 0.0836 | 0.0983 | −0.0712 | −0.0242 | +0.0147 (0.406) | +0.0471 (0.080) | +0.1298 \*\* better | +0.0073 (0.690) |

- **Verdict under the frozen decision rule: not met in any of the four comparisons.**
  - (d) local: significantly better on deletion and insertion (more faithful than the source's local map), but significantly *worse* on `rand_full_ssim`.
  - (d) fused: significantly worse on deletion; no difference on insertion or `rand_full_ssim`.
  - (e) local: worse than ConvNeXt-T on deletion, insertion, `rand_full_ssim` and stability.
  - (e) fused: worse on deletion and `rand_full_ssim`; no difference on insertion.
- **Classification vs the source hybrid** (`outputs/comparisons/ham10000_test_ham10000_hybrid_mask_l0.1_m0_vs_ham10000_hybrid.json`, 1,000 paired resamples, all valid; `outputs/runs/ham10000_hybrid_mask_l0.1_m0/eval/metrics.json`):

| Metric | Source hybrid | Mask-guided | B − A [95% CI] | p |
|---|---|---|---|---|
| Macro-F1 | 0.7903 | 0.7972 | +0.0069 [−0.0097, +0.0239] | 0.434 |
| Balanced accuracy | 0.8027 | 0.8100 | +0.0073 [−0.0074, +0.0227] | 0.360 |
| Accuracy | 0.8683 | 0.8736 | +0.0053 [−0.0053, +0.0173] | 0.362 |
| Melanoma sensitivity | 0.6012 | 0.5774 | −0.0238 [−0.0737, +0.0221] | 0.394 |

  - **"Not worse" by the frozen criterion, borderline:** the lower CI bound (−0.0097) is above −0.01 by 0.0003. No classification metric differs significantly.
- **Observations (exploratory):**
  - The val pattern holds on test: better localisation (fused IoU 0.437 → 0.526, energy in mask 0.459 → 0.801, pointing 0.848 → 0.975) and stability Spearman, worse `rand_full_ssim`, and still behind ConvNeXt-T on faithfulness.
  - Differences from val, none changing a verdict: (d) local deletion is significant on test (Holm < 0.024; val 0.128); (d) fused `iou_gap` is not (Holm 0.080; val 0.040); (e) fused insertion is not significant on test (Holm 0.092; val < 0.024 worse).
  - Secondary measures as on val: the fused stability SSIM fell (0.9751 → 0.9175); the |ρ| randomisation measures were lower (better) for the local map (`rand_full_abs_spearman` −0.010, `rand_full_abs_hog` −0.005) and about unchanged for the fused map, so the SSIM caveat of M4 applies again.
  - Summary: supervising the fused CAM with lesion masks moved the maps onto the lesion and made them more stable, at no measurable classification cost, but did not make them pass the trust rule; the localisation gain does not translate into faithfulness against ConvNeXt-T.
- **Maps for the report figures (2026-11-03, after the results above were recorded; no reported number changed):**
  - `explain.py` + `eval_xai.py --split test` on `ham10000_hybrid_mask_l0.1_m0` (`xai_test/`, for figure F6). Its scores agree with the step 5 trust test run: IoU local 0.551 vs 0.5509, fused 0.526 vs 0.5258.
  - `scripts/explain_randomised.py --split test` on `ham10000_convnext_t` and `ham10000_hybrid` (`xai_test_randomised_examples/`, figure F9): Grad-CAM of the 14 `explain.py` figure images before and after re-initialising every cascade block at once (seeded with the trust seed). This illustrates the randomisation check; its random draw differs from the M4 trust runs, whose scores stay the reported ones.
- The M4 confirmatory verdict is unchanged.

- **Not run (future work):** the same supervision on the EfficientNetV2-branch Grad-CAM. That map passes through the cross-attention, so its loss needs second-order gradients (about 2–3× the cost, and it may not fit in 4 GB); it was left for a Kaggle GPU and not done. This matters because the local map is the one that localises well in M4 and fails on faithfulness and stability.

## Future work
Sources outside `Related_papers/` are marked "external"; dataset sizes are as published and should be rechecked on download. (Update 2026-11-03: all related-paper PDFs are now readable; the report's reference list was cut to 15 entries copied from the related papers and their reference lists, so several sources named here are no longer cited in the report.)

1. **Mask-guided training on a larger masked dataset, over several seeds.**
   - Why: in Part B step 5, supervising the fused CAM with HAM10000 train masks improved localisation (fused IoU 0.437 → 0.526 on test) and stability, but not faithfulness against ConvNeXt-T, and `rand_full_ssim` got worse. This rests on one run, one seed and a val selection.
   - What: repeat with ISIC 2018 Task 1 (lesion segmentation, 2,594 training images with masks; Codella et al. 2019, external) as a second, larger mask source; at least 3 seeds per setting, reporting mean ± SD of the trust endpoints and macro-F1. Add the EffNetV2-branch supervision left undone in step 5 (second-order gradients, larger GPU).
   - Care: ISIC 2018 Task 1 images come from the ISIC Archive and may overlap HAM10000, so deduplicate (image ID and perceptual hash) first. Fix settings and the decision rule before a fresh test set is used: the HAM10000 test set has now been used by M4 and Part B.
2. **PH2 as an external test set.**
   - Why: every HAM10000 result comes from one dataset. WA-NET (Hu et al. 2025) uses PH2 alongside ISIC 2017/2018 for segmentation.
   - What: PH2 (Mendonça et al. 2013, external): 200 dermoscopic images (common nevi, atypical nevi, melanomas) with lesion masks. Its classes do not map one-to-one onto the 7 HAM10000 classes, so classification would be melanoma vs nevus; the masks allow the M4 localisation and trust metrics on unseen data.
   - Care: with 200 images the CIs will be wide; a transfer check, not a ranking.
3. **Derm7pt: do the maps follow the 7-point checklist?**
   - Why: IoU against a lesion mask only asks whether the map lies on the lesion, not whether it highlights the structures dermatologists use.
   - What: Derm7pt (Kawahara et al. 2019, external) labels each case with 7-point-checklist criteria (pigment network, streaks, dots/globules, blue-whitish veil, regression structures, …).
   - Care: the labels are per image, not per region, so they cannot be scored against a map directly. Options: test whether a criterion is linearly decodable from the features the map highlights (e.g. TCAV, Kim et al. 2018, external), or get region annotations for a subset (item 6).
4. **ISIC 2024 as a domain-shift study.**
   - Why: ISIC 2024 (SLICE-3D) consists of lesion crops from 3D total-body photographs, not dermoscopy: lower resolution, different lighting, and very few malignant cases (several hundred among roughly 400,000 crops; recheck on download).
   - What: evaluate the HAM10000 / ISIC 2020 models zero-shot, then fine-tune with a patient-grouped split, and measure how classification and the trust metrics degrade.
   - Care: no lesion masks, so only the faithfulness, stability and randomisation metrics apply; the imbalance needs the val-fixed threshold protocol used for ISIC 2020.
5. **Self-supervised / contrastive pretraining.**
   - Why: every model starts from ImageNet weights, and the randomisation results (figure F9; Score-CAM's local hybrid map at `rand_full_abs_spearman` 0.82) show maps that partly follow image structure rather than learned features.
   - What: pretrain both branches on unlabelled dermoscopy (e.g. the ISIC Archive without labels and without any test image) with SimCLR (Chen et al. 2020), MoCo v2 (Chen et al. 2020) or DINO (Caron et al. 2021), all external; fine-tune with the unchanged recipe and repeat the M4 protocol.
   - Care: deduplicate against every test set used here before pretraining.
6. **Clinician review of the maps.**
   - Why: every trust metric here is computational; none measures whether a map is useful to a clinician.
   - What: a blinded review of a fixed, stratified image set (model and map type hidden, random order). Dermatologists rate plausibility and mark the region they would examine; report inter-rater agreement (Cohen's or Fleiss' κ) and agreement between marked regions and the maps. Rating protocol and analysis plan written before the review.
   - Care: needs clinician time, and possibly ethics approval depending on the institution.

**Carried over from earlier sections:**
- ISIC 2020 baselines ResNet-50, ViT-B and Swin-T (Deviations, compute limit). (The EfficientNetV2-S run listed here before was done on 2026-11-03; see "ISIC 2020: EfficientNetV2-S".)
- Score-CAM on ConvNeXt-T on a non-negative layer, or a variant with signed weights, so comparisons (a) and (c) measure Score-CAM itself and not the blank-map failure (Part B step 4).
- The EffNetV2-branch mask supervision of Part B step 5 (item 1).

## Limitations
Collected from the sections above (named in brackets), plus the two items deliberately left undone: the float32 tie fix and more training seeds.

**Statistical design**
1. **Single runs, single seed (not done).** Every model was trained once (training seed 42).
   - The 95% CIs and paired tests cover only variation over test images, not over training runs.
   - Val differences of about 0.01 macro-F1 are within run-to-run noise ("Milestone 3 summary", observations). The hybrid vs ConvNeXt-T and cross-attention vs concat differences are not significant on test, and their direction could change with another seed.
   - More seeds were skipped for time: about 1–4 h per HAM10000 run on the RTX 3050.
2. **The HAM10000 test set has been used more than once:** the M4 trust test (confirmatory), then the classification test of the Milestone 3 runs, then Part B.
   - The M4 trust run had predicted the hybrid's test labels before the classification test; nothing was chosen on them.
   - Everything after M4 is labelled EXPLORATORY. A claim that a method makes the maps more trustworthy would need a fresh test set (Future work 1–2).
3. **Changes after seeing val** (no test data involved):
   - four changes to the M4 protocol after its val run ("M4 … changes made after looking at val data");
   - one learning-rate change for ViT-B/16 ("Baselines");
   - the mask-guided run chosen on a val subset, so its val comparisons are optimistic (Part B step 5).
4. **Not pre-specified:** the hair-removed hybrid's best test macro-F1 (0.830) was not a planned comparison, and per-class sensitivities (e.g. the hybrid's lower melanoma sensitivity vs ConvNeXt-T, p 0.042 uncorrected) are exploratory.
5. **Rare classes:** the HAM10000 test set has 17 df and 21 vasc images, and the Score-CAM subset 2 of each; per-class statements about them are not possible.

**Numerical precision**
6. **Tied probabilities at a cutoff (fix not applied).** With AMP the predicted probabilities are quantised. `evaluate.py` compares in float32, while `select_threshold.py` and `compare_paired.py` compare the CSV values (ties count as melanoma).
   - For ConvNeXt-T at the ISIC sensitivity ≥ 0.80 cutoff this gives specificity 0.8264 vs 0.8255 (4 test images), and for EfficientNetV2-S 0.7187 vs 0.7181 (Youden) and 0.7952 vs 0.7946 (sens ≥ 0.80; 3 test images each); the hybrid's two cutoffs and ConvNeXt-T's Youden cutoff agree to 4 decimals.
   - One comparison precision in `predict_labels` would fix it, but that would change how a fixed rule is applied after the test run, so it was not done ("ISIC 2020 test set", tied probabilities).
7. **fp16 / cuDNN non-determinism:** EfficientNetV2-S val macro-F1 is 0.7842 from `evaluate.py` vs 0.7850 during training, from the same checkpoint ("Baselines", observations).

**Data**
8. **ISIC 2020:**
   - a pre-resized 256×256 copy, so non-square images are stretched;
   - only the public training set, split by us, so results are not comparable with the challenge leaderboard;
   - no lesion masks, so no localisation;
   - the hybrid's sensitivity ≥ 0.80 cutoff did not transfer to test (sensitivity 0.730).
9. **ISIC 2020 baselines:** ResNet-50, ViT-B and Swin-T were not trained (Deviations). Only ConvNeXt-T, EfficientNetV2-S (pretrained norm) and the hybrid are compared on ISIC 2020.
10. **One dermoscopy source for all localisation results:** HAM10000 images come from two sites (Tschandl et al. 2018, external), and skin type is not in the metadata. Behaviour on other skin types or devices is untested (Future work 2 and 4).
11. **Large lesions make localisation easy to match:** lesions cover 41.8% of the val crop on average, so a uniform map already scores IoU 0.418 and a centred Gaussian 0.487 ("XAI localisation", lesion size). Without these references IoU overstates localisation; pointing is weak evidence for the same reason (centre pixel 0.961).
12. **Hair removal:** DullRazor sometimes removes lesion borders and pigment network ("Hair removal", known failure cases), in the region the IoU is measured against. Its parameters were tuned visually on 100 images; there was no full audit.

**Explanation metrics**
13. **Blurred fill in deletion / insertion** creates inputs unlike the training images, so part of each score reflects how the model reacts to out-of-distribution inputs, not only how faithful the map is. The vs-random endpoints reduce this but do not remove it.
14. **SSIM and flat maps:** a flat map can score a high SSIM to its randomised version, so `rand_full_ssim` is read next to |ρ| (`rand_full_abs_spearman`, `rand_full_abs_hog`), and the two sometimes disagree (M4, SSIM caveat).
15. **One random draw per randomisation:** variation between re-initialisations is not measured (the figure F9 examples use yet another draw).
16. **Few stability perturbations:** flip, Gaussian noise and two brightness levels. The unchanged-prediction rule drops the images whose prediction the noise changes (M4 test: 17.6% ConvNeXt-T, 19.5% hybrid).
17. **Method-specific failures:**
    - the hybrid's fused Grad-CAM is often flat enough to score the uniform map's IoU ("XAI localisation", observations);
    - Score-CAM gives all-zero maps for 16–18% of the ConvNeXt-T subset images (signed layer; Part B step 4) and was run on 150 images per split only.
18. **No human evaluation:** every trust metric is computational, and no clinician has rated a map (Future work 6).

**Compute and environment**
19. **Hardware:** all HAM10000 runs used one RTX 3050 (4 GB) with batch 8 × 4 accumulation and gradient checkpointing. RAM paging (7.3 GB system RAM) stretched some epochs to hours, which changed wall time, not the computation ("Hybrid", run history); timings in these notes are lower bounds.
20. **Timeline:** the proposal's dates had passed before implementation began; the milestone order was kept (Deviations).

## Deviations from the proposal
- ISIC 2020 is used at 256x256, a pre-resized copy chosen for the 4 GB GPU and download size; see Data above.
- ViT-B/16 uses backbone lr 3e-5 instead of the shared 1e-4; see Baselines above.
- The proposal timeline (Jul 15 – Sep 15) had passed by the time of implementation. The milestone order is kept.
- ISIC 2020 baselines: only ConvNeXt-T (the strongest HAM10000 baseline) and EfficientNetV2-S with pretrained normalisation (the hybrid's local branch alone) are trained on ISIC 2020, next to the hybrid. ResNet-50, ViT-B/16 and Swin-T, which the proposal also lists, are not, because of compute limits: ISIC 2020 runs need Kaggle GPU sessions (1.5–2 h each so far, with a weekly GPU quota). EfficientNetV2-S was chosen because it is the proposal's local branch, so it isolates what the Swin branch and fusion add; ConvNeXt-T because it was the best HAM10000 baseline. ViT-B/16 was close behind it on HAM10000 test macro-F1 (0.805 vs 0.812), so leaving it out is a real gap, not a pick of weak models. Their ISIC 2020 results are not reported.
- The proposal evaluates explanations by IoU against the lesion masks only. Milestone 4 adds faithfulness, randomisation and stability tests with a paired bootstrap (see "M4 explanation trust"). IoU stays one of the endpoints.
