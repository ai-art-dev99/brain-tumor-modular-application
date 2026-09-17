# Leakage, Redundancy and Source Shortcuts in a Widely Used Brain Tumour MRI Benchmark: A Provenance-Reconstructed, Group-Disjoint Re-evaluation of Hybrid Deep-Feature Classifiers

**Authors:** Amirparsa Rouhi¹, Fatemeh Parsakordasiabi¹, Timothy Albiges¹, Zoheir Sabeur¹\*

¹ Department of Computing and Informatics, Bournemouth University, Poole, Dorset, BH15 5BB, United Kingdom

\* Corresponding author: zsabeur@bournemouth.ac.uk

---

## Abstract

**Background.** A composite dataset of approximately 7,000 brain MRI slices, assembled from three public repositories, has become the de facto benchmark for four-class brain tumour classification. Accuracies of 98–99% are routinely reported on it. Whether those figures reflect diagnostic capability has not been examined.

**Methods.** In a retrospective methodological re-analysis of public MRI datasets, we reconstructed the provenance of this benchmark from its source repositories, recovering patient identifiers and tumour masks from the original MATLAB archives of the Figshare collection and matching them to the redistributed benchmark by dual perceptual hashing. We removed exact and near-duplicate images, assigned every image to a leakage-control group, and re-evaluated a fine-tuned EfficientNetB0 and five hybrid deep-feature classifiers under nested group-disjoint cross-validation. We measured evaluation inflation by three complementary designs, probed whether repository identity is recoverable from non-pathological image content, quantified attribution against annotated tumour masks using occlusion sensitivity on the deployed pipeline, and screened two candidate external cohorts for contamination before any model was applied to them.

**Results.** The 6,402 curated images reduce to 3,813 after duplicate removal and resolve to 721 leakage-control groups; 1,452 near-duplicate clusters span two of the three repositories, which are therefore not independent sources. In the published train/test partition, 123 of 212 recoverable patients contribute images to both sides. Under image-level splitting our pipeline reaches 0.9756 accuracy (EfficientNetB0+SVM); under group-disjoint splitting, with the same architecture, training pipeline, hyperparameter search space, code and seed, it reaches 0.9405, and on a single-repository three-class cohort 0.9230. Inflation attributable to image-level splitting is +3.1 to +5.7 percentage points, and between-fold standard deviation falls threefold under leakage, so the flawed protocol inflates the estimate and understates its uncertainty simultaneously. On a third-party release, images matching our development data were classified 5.6–6.8 points more accurately than the same images scored by the one backbone fitted without their counterpart (n = 2,009; all p = 0.0005, the smallest value attainable with 4,000 resamples), while a negative control on 3,699 non-overlapping images with randomised fold labels produced differences of −0.22 to +0.25 points. A linear probe recovers repository identity from the region outside an estimated head mask with 0.9997 balanced accuracy, and with 0.9843 when the diagnostic class is held constant. The hybrid head improves accuracy over the standalone network by 1.16 points (95% CI 0.26–2.08; p = 0.011), but no downstream classifier dominates and relative ranking is not stable to the evaluation protocol. Occlusion attribution on the deployed pipeline exceeds the uniform baseline for all three tumour classes and is three- to sevenfold higher on correct than on incorrect predictions. Neither candidate external cohort was usable: 33.8% and 37.7% of their images match the source repositories, rising to 56.1% and 67.7% against the composite redistribution on which the published literature is trained, the latter including 3,315 byte-identical files.

**Conclusions.** Performance on this benchmark is materially overestimated by the evaluation protocol in common use, one class is separable from non-pathological image characteristics, and we did not identify a verifiably independent public cohort for external testing among the candidates screened. We report leakage-controlled estimates, release the audit tooling, and argue that cross-repository provenance screening should be a routine reporting requirement for redistributed medical imaging benchmarks.

**Keywords:** brain tumour, magnetic resonance imaging, data leakage, dataset provenance, shortcut learning, model evaluation, transfer learning

---

## 1. Introduction

Brain and central nervous system tumours account for under 2% of malignancies worldwide but carry disproportionate mortality and cost [1–3]. Magnetic resonance imaging is the primary diagnostic modality, and the interpretation burden on neuroradiologists has motivated a large literature on automated classification [4–8].

Much of that literature is built on one dataset. A composite collection of approximately 7,000 brain MRI slices, assembled from the Figshare collection of Cheng et al. [25], the SARTAJ Kaggle dataset [26] and the BR35H detection dataset [27], and redistributed as a four-class benchmark, has become the standard testbed. Reported accuracies cluster between 98% and 99% [21–23]. Our own earlier work on this benchmark reported 96%.

Two features of this literature invite scrutiny. First, the reported figures are close to ceiling, tightly clustered across architectures of very different capacity, and accompanied by narrow confidence intervals. Second, the benchmark is a redistribution: the relationship between its constituent repositories, the presence of duplicate images, and the patient structure underlying its slices are not documented anywhere in the chain from original acquisition to published result.

Data leakage is a recognised failure mode in medical imaging. Where several slices of one patient are distributed across a random split, a model can succeed by recognising the patient rather than the pathology, and the resulting estimate does not describe performance on new patients. Related work has documented inflation of several to tens of percentage points in other modalities. Shortcut learning is a second, distinct failure mode: where a class label correlates with an acquisition or processing characteristic, a model may learn that characteristic instead of the clinical finding.

This study began as a revision of a conventional comparison between a fine-tuned EfficientNetB0 and hybrid deep-feature classifiers. Reconstructing the dataset to the standard required for patient-level evaluation invalidated the original result and produced a different study. We report:

1. **A provenance reconstruction** of the benchmark, recovering patient identifiers and tumour masks that every redistribution discards, and establishing the relationships between its constituent repositories.
2. **Three independent measurements of evaluation inflation**, by designs that differ in what they hold constant and therefore in which confounds they exclude.
3. **Evidence of a source-associated shortcut** affecting one class, obtained from regions outside the estimated head mask, which contain no brain tissue.
4. **A leakage-controlled re-evaluation** of the standalone and hybrid architectures, with complete metrics, group-level intervals and paired significance testing.
5. **A quantitative attribution analysis** against annotated tumour masks, using a method that explains the deployed hybrid rather than a discarded classification head.
6. **A contamination screening** of two candidate external cohorts, conducted before any model was applied to them, and reported in place of an external evaluation.

All code, configurations, split assignments, per-image predictions and audit manifests are released so that every figure below can be reproduced from the source repositories.

---

## 2. Related work

### 2.1 Classification on this benchmark

Custom convolutional networks on this task report accuracies from 94% upwards [19, 20]. EfficientNet variants are the most common recent choice, with reported accuracies of 98–99% [21–23], and explainability-augmented variants report 98.72% [29]. Hybrid designs that use a deep network as a feature extractor and a conventional classifier as the decision layer are a recurring motif.

Table 1 characterises the studies most directly comparable to ours by the properties that determine whether their figures are comparable with leakage-controlled ones.

**Table 1. Evaluation practice in comparable studies on this benchmark.**

| Study | Dataset | Patient-level split | Duplicate removal | External test | Reported accuracy |
|---|---|---|---|---|---|
| Das et al. [19] | Figshare | Not reported | Not reported | No | 94.39% |
| Ullah et al. [20] | Composite | Not reported | Not reported | No | ~98% |
| Aksoy & Dasgupta [21] | Composite | Not reported | Not reported | No | 99% |
| Iqbal et al. [22] | Composite | Not reported | Not reported | No | 99% |
| Hastomo et al. [23] | Composite | Not reported | Not reported | No | ~98% |
| R et al. [29] | Composite | Not reported | Not reported | No | 98.72% |
| This study | Reconstructed | Yes, where identifiers exist | Yes, documented | Screened, none usable | 0.9405 |

We do not suggest that these studies contain errors of execution. They follow the prevailing convention, which is to accept a redistributed dataset as delivered and to split it at the level of individual images. Section 4.2 shows that our own pipeline reaches 0.9756 under that convention and 0.9405 without it, which is why the two columns are not comparable.

### 2.2 Leakage and shortcut learning

Patient-level partitioning is recommended by current reporting guidance for medical imaging AI, which also distinguishes internal testing from external testing on data from a separate institution. Shortcut learning, in which a model exploits a spurious correlate of the label, has been documented in several imaging domains. The two failure modes interact here: a benchmark assembled from repositories that differ in acquisition can encode class identity in acquisition characteristics, and a split that does not respect patient boundaries conceals how much of the resulting performance is transferable.

---

## 3. Materials and methods

### 3.1 Source repositories and provenance reconstruction

We worked from the three source repositories rather than from the redistributed benchmark. The redistribution is a mutable record: the release used in our original submission contained 7,023 images, and the release current at the time of writing contains 7,200, with the publisher reporting subsequent duplicate removal and elimination of train/test overlap.

Every image in each repository was indexed programmatically, recording the SHA-256 file hash, two perceptual hashes (pHash and dHash, 64 bits each), image dimensions and originating directory. For the Figshare collection we read the original MATLAB v7.3 archives, recovering for each of the 3,064 slices the patient identifier, the diagnostic label and the binary tumour mask. These fields are absent from every JPEG or PNG redistribution of that collection.

**Table 2. Source composition and available metadata.**

| Source | Files distributed | Retained after deduplication | Patients | Sequence | Contrast | Plane | Scanner | Native size | Patient ID | Mask |
|---|---|---|---|---|---|---|---|---|---|---|
| Figshare (Cheng) | 3,064 | 3,038 | 233 | T1-weighted | Contrast-enhanced (Gd-DTPA) | Axial, coronal, sagittal | Two hospitals, 2005–2010 | 512×512 | Yes | Yes |
| SARTAJ | 3,264 | 168 | NR | NR | NR | NR | NR | Variable (440 distinct) | No | No |
| BR35H | 3,000 | 607 | NR | NR | NR | NR | NR | Variable (291 distinct) | No | No |

NR = not reported by the originating repository. We have not inferred acquisition parameters from image appearance.

For the Figshare collection the originating publication additionally reports in-plane pixel spacing of 0.49 × 0.49 mm², slice thickness 6 mm, inter-slice gap 1 mm, and gadopentetate dimeglumine at 0.1 mmol/kg administered at 2 mL/s, with acquisition at two hospitals between 2005 and 2010 [25, 28]. No comparable information is published for the other two repositories. The Figshare subset is reported by its repository as T1-weighted contrast-enhanced; neither SARTAJ nor BR35H reports sequence metadata, and we assign none. The BR35H images exhibit visibly different intensity and contrast characteristics from the Figshare subset, but we do not translate that observation into sequence labels; what can be established quantitatively is reported in Section 4.4.

Within the Figshare collection, the 3,064 slices derive from 233 patients: 89 with glioma (1,426 slices), 82 with meningioma (708 slices) and 62 with pituitary tumour (930 slices). The number of slices per patient ranges from 1 to 38, with a median of 13 and a mean of 13.2. This distribution is the reason an image-level split cannot yield an independent test set on this benchmark.

### 3.2 Inclusion and exclusion

**Table 3. Exclusion criteria and supporting evidence.**

| Excluded | n | Reason |
|---|---|---|
| SARTAJ glioma | 926 | Documented labelling inconsistency in this subset; gliomas drawn from Figshare instead |
| SARTAJ no-tumour | 500 | 500 files reduce to 352 unique file hashes (29.6% internally duplicated); 208 of the files, representing 120 distinct images, are byte-identical to images in BR35H |
| BR35H `yes` | 1,500 | Tumours of unspecified histology; cannot be assigned to one of the four diagnostic classes |
| BR35H `Br35H-Mask-RCNN` | 801 | Byte-identical repackaging of the `yes` folder for an object-detection task |
| BR35H `pred` | 60 | No ground-truth labels |

The curated dataset comprises 6,402 images: glioma 1,426 (Figshare only), meningioma 1,645 (Figshare 708, SARTAJ 937), pituitary 1,831 (Figshare 930, SARTAJ 901) and no-tumour 1,500 (BR35H only).

### 3.3 Deduplication and leakage-control grouping

Two distinct thresholds are used and are not interchangeable.

**Deduplication threshold.** Near-duplicates are identified by agreement between both perceptual hash families, requiring each to fall within 2 bits. Exact hashing is insufficient: no byte-identical duplicate exists between the Figshare renders and their SARTAJ counterparts, yet 1,452 near-duplicate clusters span the two repositories.

The threshold was calibrated against an internal negative control. Glioma is drawn exclusively from Figshare and consists of distinct consecutive slices, so a correctly set threshold should remove almost none of it. At 2 bits, 8 of 1,426 glioma images (0.6%) are removed while the mixed-source classes lose 48–60% (Table 4). That contrast is the evidence that the removals reflect genuine redundancy rather than perceptual-hash collision.

Where a cluster spans repositories, the Figshare member is retained: it alone carries the patient identifier and tumour mask, and it is a lossless render of the original 16-bit slice rather than a recompressed derivative.

**Grouping threshold.** After deduplication, a looser threshold of 8 bits is applied to keep visually correlated but non-identical images within a single partition. Its value was fixed before any model was fitted, from a sensitivity sweep over 4 to 12 bits (Supplementary Table S3), on two criteria stated in advance: the group count of the no-tumour class, the only class whose grouping depends entirely on similarity, should have entered the flat portion of the sweep, and the glioma control should be unchanged. Successive reductions in the no-tumour group count are 91, 58 and 35 groups for thresholds 6, 8 and 10, and rise again to 50 at 12; 8 lies within the plateau and before that rise. We note that this is a judgement made on aggregate statistics rather than an optimisation, and we report the full sweep so that readers can see how the choice would have affected the partition. We note that the glioma group count is invariant across that sweep as a consequence of the patient-identifier constraint described below, and is therefore not independent evidence for the grouping threshold; the calibration evidence is the deduplication control above.

**Patient-level grouping.** Recovered identifiers provide true patient grouping for 233 patients. Identifiers propagate across duplicate clusters, so a SARTAJ image that is a re-encoding of a Figshare slice inherits that patient; this raises the proportion of images carrying a metadata-backed identifier from 47.9% to 71.2%. For images with no identifier in any release, near-duplicate clusters serve as pseudo-patient groups. Where metadata states that two images belong to different patients, that statement overrides hash similarity: 185 similarity edges were blocked on this basis. We describe these units as leakage-control groups, not patients, throughout.

**Table 4. Deduplication by class, and the resulting group structure.**

| Class | Curated | Retained | Removed (%) | Groups | Images per group |
|---|---|---|---|---|---|
| Glioma | 1,426 | 1,418 | 0.6 | 89 | 15.9 |
| Meningioma | 1,645 | 835 | 49.2 | 204 | 4.1 |
| Pituitary | 1,831 | 953 | 48.0 | 89 | 10.7 |
| No tumour | 1,500 | 607 | 59.5 | 339 | 1.8 |
| **Total** | **6,402** | **3,813** | **40.4** | **721** | **5.3** |

The 3,813 retained images resolve to 721 leakage-control groups. Of them, 3,038 (79.7%) carry a tumour mask, and 168 SARTAJ images survive as non-duplicates of Figshare (133 meningioma, 35 pituitary).

### 3.4 Experimental configurations and partitioning

Three configurations are evaluated.

**Main.** All 3,813 images, four classes, 721 leakage-control groups.

**Single-repository (Figshare-only).** The 3,038 images carrying a metadata-backed patient identifier, three classes (no no-tumour images exist in the Figshare collection), 233 true patients. This is the only configuration in which grouping rests entirely on metadata and all images share one repository and one imaging sequence.

**Image-level baseline.** The main configuration partitioned by `StratifiedKFold` at the level of individual images, ignoring groups. This reproduces the prevailing convention and exists only to be compared against.

All configurations use 5-fold outer cross-validation with a 3-fold inner loop, seed 42. For the grouped configurations both loops use `StratifiedGroupKFold`. Split generation aborts if any group or any patient spans two outer folds, or if any inner validation fold intersects its outer test fold.

The image-level baseline quantifies what it discards: 370 of 721 groups (51.3%) and 222 of 233 patients (95.3%) span more than one fold, and 3,437 of 3,813 test images (90.1%) have a same-group sibling in the corresponding training partition. Under the grouped configurations all three figures are zero by construction.

We also examined the partition distributed with the benchmark itself. Matching the redistributed images to the Figshare collection used a distinct threshold from deduplication, because the task is different: deduplication removes redundancy within our own cohort and is deliberately strict, whereas provenance matching must survive the re-encoding and rescaling that redistribution applies. The distance distribution for that matching is bimodal with an empty region between 3 and 11 bits, and a threshold of 4 bits raised agreement between the recovered and the distributed class label from 88.7% (at 8 bits) to 98.7%, which we took as the operational criterion. Matching at 4 bits recovered 1,829 images with 98.7% agreement between the recovered and the distributed class label, covering 212 patients. Of those, 123 (58%) contribute images to both the published training and testing folders, involving 1,306 images. This is a lower bound: only Figshare-derived images carry identifiers, so leakage among SARTAJ- and BR35H-derived images is not measurable rather than absent. The current release of the redistribution reports subsequent removal of duplicate images and of train–test overlap; we do not assert that this was its cause.

### 3.5 Models and implementation

The backbone is EfficientNetB0 as distributed by `timm`, initialised from ImageNet weights. Input resolution (224×224) and normalisation statistics are resolved from the model's own published data configuration rather than hard-coded. All layers are fine-tuned.

Fine-tuning uses AdamW (learning rate 3×10⁻⁴, weight decay 10⁻⁴), batch size 32, cosine annealing, a maximum of 20 epochs, and early stopping with patience 5 on balanced accuracy over an inner validation fold. The loss is cross-entropy with inverse-frequency class weights and label smoothing 0.05. Class weighting rather than oversampling is used, because duplicating images within a group would further amplify the few patients contributing many slices.

The backbone is retrained from ImageNet initialisation inside each outer fold, on that fold's training portion alone. Fine-tuning once on all data and then extracting features would place test-fold information in the feature extractor and would invalidate the comparison between the standalone and hybrid models.

Training augmentation is applied on the fly and only to the training portion: random affine transformation (±15°, 5% translation, 0.9–1.1 scale), horizontal flip with probability 0.5, and brightness and contrast jitter of 0.15. No augmented image is written to disk, so class counts before and after augmentation are identical by construction and no synthetic image can enter a validation or test partition (Supplementary Table S4). We note that horizontal flipping is not strictly anatomy-preserving for brain MRI; we retain it for continuity with prior work and state the choice rather than leaving it implicit. Evaluation applies resizing and normalisation only, with no stochastic component.

Features for the hybrid models are the 1,280-dimensional global-average-pooled output of the fine-tuned backbone. Five heads are compared: support vector machine, k-nearest neighbours, random forest, multilayer perceptron, and logistic regression as an additional baseline. Each is tuned by grid search over the group-disjoint inner folds (grids in Supplementary Table S5).

Two implementation choices bear on published practice. The multilayer perceptron does not use the library's internal early stopping, which constructs a validation split at random and does not respect the patient grouping; the stopping point would otherwise be chosen against partially memorised data. The support vector machine does not use the library's internal probability estimation, which fits Platt scaling through a non-group-aware five-fold cross-validation; calibration is performed explicitly over the same group-disjoint inner folds used for tuning, since the selective-prediction analysis depends on those posteriors.

Seeds are fixed for Python, NumPy and PyTorch with deterministic backend settings. Re-running split generation after a complete rewrite of the codebase reproduced every fold assignment, group count and class distribution identically.

### 3.6 Metrics and statistical analysis

Reported per configuration: accuracy, balanced accuracy, macro and weighted F1, macro precision, recall and specificity, macro one-vs-rest ROC-AUC, multiclass Brier score and expected calibration error. Reported per class: precision, sensitivity, specificity, F1, one-vs-rest ROC-AUC, support, and raw TP/FP/FN/TN counts. Confusion matrices with real counts and fold-level results are given in the supplement.

**Confidence intervals resample groups, not images.** With approximately 16 correlated images per glioma patient, resampling images would treat 16 views of one brain as 16 independent observations. All intervals are 95% percentile bootstrap intervals over leakage-control groups with 2,000 resamples, covering every reported metric including per-class values.

The comparison between the standalone network and the EfficientNetB0+SVM hybrid was designated primary in advance and is reported unadjusted; remaining pairwise comparisons are exploratory and carry Holm-adjusted values. Both exact McNemar and a group-level paired bootstrap are reported, with the bootstrap designated primary: McNemar assumes independent observations, which four to sixteen correlated images per patient violates.

We note that in a four-class one-vs-rest formulation each class is evaluated against a comparatively large negative pool, so specificity can remain numerically high even where sensitivity and precision differ appreciably between models. It is reported for completeness but carries no comparative argument.

### 3.7 Source-shortcut probes

To test whether repository identity is recoverable independently of pathology, a logistic-regression probe is trained on frozen ImageNet features to predict the source repository, using the same group-disjoint folds as the diagnostic task. Three variants are used:

- **Full image.** Source and diagnosis are confounded in this benchmark, so this variant alone cannot distinguish acquisition signature from tumour phenotype.
- **Background only.** Everything inside an Otsu-derived head mask is removed. No brain tissue remains, so tumour phenotype cannot explain a positive result. The region does retain the skull contour, field of view, cropping, padding, resampling and compression, so what is established is source-specific non-pathological signal rather than scanner acquisition in isolation.
- **Brain only.** The complement, optionally cropped to the head bounding box and rescaled, which removes framing and field-of-view cues as well as the background.

A class-conditional variant holds the diagnosis fixed at meningioma (702 Figshare against 133 SARTAJ images), removing the confound by design.

### 3.8 Attribution analysis

Attribution is scored against the annotated tumour masks rather than illustrated. Masks are used only for images rendered from the original archives; SARTAJ copies were independently rescaled before redistribution and their masks would not register.

For each image we compute the fraction of attribution mass falling inside the mask, normalised by the mask's own area fraction. This ratio, which we call **concentration**, equals 1.0 for a uniform and therefore uninformative map. Normalisation matters because a large tumour captures a large share of a diffuse map. We also report the pointing game and an area-matched IoU in which the salient region is thresholded to the same size as the ground-truth mask, so the overlap measure does not depend on an arbitrary threshold.

Grad-CAM differentiates through the network's own classification head, which the hybrid models discard. For the hybrid we therefore use occlusion sensitivity on the complete pipeline — image, backbone, pooled features, classical head — with the head refitted per fold exactly as in the main analysis. The occluding value is the dataset mean in normalised space rather than black, which would itself be out of distribution. Figures are computed on a stratified random sample of 100 images per class per configuration.

### 3.9 External contamination screening

Candidate external cohorts were screened before any model was applied to them, using the dual-hash criterion calibrated in Section 3.3, with structural similarity on a common 256×256 grid as an independent confirmation. Overlap is counted over near-duplicate clusters rather than files, since several re-encodings of one scan matching one development image constitute one contaminated unit. The screening protocol was committed to the public repository before the screening was run.

Three comparison scopes are distinguished, because they answer different questions: against the 3,813 images actually used for fitting; against the source repositories before deduplication, excluding the composite redistribution; and against the composite redistribution itself, which is not an acquisition source.

### 3.10 Leakage contrasts on a third-party release

The screening in Section 3.9 identifies, for every image in a candidate release, whether it matches our development data. Two contrasts exploit that labelling. Neither treats the release as an external validation cohort.

**Observational contrast.** Images in the publisher's test partition are stratified as *direct* (matching the images used for fitting), *source-only* (matching the source repositories but not the fitted set) and *clean* (matching neither), with a *strict-clean* subset restricted to nearest distances of 12 bits or more. Because class mixture differs between strata, balanced accuracy is the primary measure and per-class recall is reported separately for each stratum. The strata are different observation sets, so no paired test applies; the difference in balanced accuracy is bootstrapped with resampling within class within stratum, which holds class sizes fixed across iterations.

**Within-image paired contrast.** Because outer folds are group-disjoint, the leakage-control group containing a matched image's counterpart was held out of exactly one of the five fold-specific backbones. For each matched external image, four backbones were therefore fitted with that group present and one without it. Exposure is determined from *all* development images within 2 bits of the external image, not the nearest one alone: an image whose counterparts appear in more than one fold would otherwise be misclassified as unexposed for a backbone that had seen a different counterpart. Images with no unexposed backbone are excluded; in this release none were, since near-duplicate clustering places all counterparts of one scan in a single group and groups do not cross folds.

For each image, correctness is averaged over the exposed backbones and over the unexposed backbone, giving two per-image scalars. The bootstrap unit is therefore the image, not the prediction. Intervals are 95% percentile intervals over 4,000 resamples, and *p*-values use (extreme + 1)/(resamples + 1), so the smallest attainable two-sided value is 0.0005. Holm adjustment is applied across the model family.

**Negative control.** The identical procedure is applied to images that match nothing in the development repositories, with the unexposed fold assigned at random. Every mechanical feature is the same — the same images scored by all five backbones, the same leave-one-fold-out arithmetic, the same bootstrap — but the grouping variable carries no information. A non-null result would indicate that the estimator or ordinary variation between backbones produces a difference of its own, and would invalidate the paired analysis.

---

## 4. Results

### 4.1 Leakage-controlled performance

**Table 5. Four-class group-disjoint evaluation (3,813 images, 721 groups). 95% group-bootstrap intervals.**

| Model | Accuracy | Balanced accuracy | Macro F1 | Macro AUC | ECE |
|---|---|---|---|---|---|
| EfficientNetB0 | 0.9287 [0.9063, 0.9481] | 0.9381 | 0.9320 | 0.9865 | 0.031 |
| + SVM | 0.9405 [0.9215, 0.9580] | 0.9446 | 0.9419 | 0.9877 | 0.017 |
| + MLP | 0.9415 [0.9231, 0.9587] | 0.9445 | 0.9424 | 0.9907 | 0.044 |
| + Logistic regression | 0.9397 [0.9208, 0.9570] | 0.9438 | 0.9409 | 0.9887 | 0.036 |
| + Random forest | 0.9344 [0.9149, 0.9523] | 0.9394 | 0.9358 | 0.9892 | 0.147 |
| + k-nearest neighbours | 0.9184 [0.8979, 0.9371] | 0.9185 | 0.9176 | 0.9686 | 0.041 |

**Table 6. Single-repository three-class evaluation (3,038 images, 233 patients).**

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---|---|
| EfficientNetB0 | 0.9042 [0.8748, 0.9312] | 0.9103 | 0.8973 |
| + MLP | 0.9273 [0.9032, 0.9487] | 0.9233 | 0.9186 |
| + SVM | 0.9230 [0.8973, 0.9454] | 0.9232 | 0.9158 |
| + Logistic regression | 0.9217 [0.8973, 0.9431] | 0.9184 | 0.9127 |
| + Random forest | 0.9161 [0.8881, 0.9395] | 0.9135 | 0.9078 |
| + k-nearest neighbours | 0.9042 [0.8781, 0.9285] | 0.8841 | 0.8887 |

### 4.2 Evaluation inflation

**Table 7. Effect of partitioning. Same architecture, training pipeline, hyperparameter search space, code and seed; the backbone is retrained within each partitioning, so weights are not shared between the two columns.**

| Model | Group-disjoint | Image-level | Inflation (pp) |
|---|---|---|---|
| + k-nearest neighbours | 0.9184 | 0.9759 | **+5.74** |
| EfficientNetB0 | 0.9287 | 0.9691 | **+4.04** |
| + SVM | 0.9405 | 0.9756 | +3.51 |
| + Logistic regression | 0.9397 | 0.9722 | +3.25 |
| + Random forest | 0.9344 | 0.9667 | +3.23 |
| + MLP | 0.9415 | 0.9725 | +3.09 |

Two features of this table are worth separating from the headline numbers.

**Sensitivity to leakage is classifier-dependent.** The nearest-neighbour head shows the largest inflation, which follows from its mechanism: under image-level splitting it selects k = 1 in every fold and becomes the best-performing model overall, and under group-disjoint splitting it selects k = 5 or k = 11 and becomes the worst. Relative model ranking on this benchmark is therefore not stable to the evaluation protocol.

**Leakage narrows intervals as well as raising means.** Between-fold standard deviation of accuracy, across the six model configurations, is 0.021 to 0.029 under group-disjoint splitting and 0.007 to 0.009 under image-level splitting. A single random image-level split can land anywhere in a range of several points while appearing precise.

### 4.3 Comparison of architectures

Under four-class group-disjoint evaluation the hybrid improves **accuracy** over the standalone network by 1.16 percentage points (95% CI 0.26 to 2.08; group-level paired bootstrap p = 0.011; exact McNemar p = 0.0002). The corresponding difference in balanced accuracy is 0.65 points; both tests operate on per-image correctness, so accuracy is the quantity they address, and we report the balanced figure alongside rather than substituting one for the other.

Under single-repository three-class evaluation the improvement in accuracy is 1.86 points (95% CI 0.62 to 3.24; patient-level paired bootstrap p = 0.003).

**We do not claim that the SVM is the best classifier.** The MLP head reaches marginally higher accuracy and the two are not statistically distinguishable, as are the SVM and logistic-regression heads. Nearest-neighbour and random-forest heads do not improve on the standalone network. The defensible claim is that replacing the end-to-end softmax layer with a conventional classifier fitted on pooled features yields a small but detectable improvement, and that no single downstream classifier dominates.

**Table 8. Per-class performance, four-class group-disjoint evaluation.**

| Model | Class | Support | Sensitivity | F1 |
|---|---|---|---|---|
| EfficientNetB0 | Glioma | 1,418 | 0.897 [0.849, 0.938] | 0.933 |
| | Meningioma | 835 | 0.909 [0.856, 0.949] | 0.859 |
| | No tumour | 607 | 0.997 [0.991, 1.000] | 0.985 |
| | Pituitary | 953 | 0.950 [0.915, 0.978] | 0.951 |
| + SVM | Glioma | 1,418 | 0.936 [0.898, 0.966] | 0.949 |
| | Meningioma | 835 | 0.890 [0.834, 0.930] | 0.877 |
| | No tumour | 607 | 0.997 [0.991, 1.000] | 0.987 |
| | Pituitary | 953 | 0.956 [0.922, 0.983] | 0.954 |

Meningioma is the weakest class in both models, with the dominant confusions in both directions between glioma and meningioma. This is consistent with the clinical difficulty of the distinction: both present as intracranial lesions of variable appearance, whereas pituitary tumours occupy a fixed anatomical position. The no-tumour class is the strongest by a wide margin, and Section 4.4 bears on why.

### 4.4 Source-associated shortcut

**Table 9. Recovery of repository identity from image content.**

| Variant | Comparison | Accuracy | Balanced accuracy |
|---|---|---|---|
| Whole image | Three repositories | 0.9830 | 0.9256 |
| Background only | Three repositories | 0.9746 | 0.8784 |
| Background only | Figshare vs BR35H | **0.9995** | **0.9997** |
| Background only | Meningioma only, Figshare vs SARTAJ | 0.9940 | 0.9843 |
| Brain only, cropped | Three repositories | 0.9722 | 0.8744 |

Restricted to the region outside an estimated head mask, a linear probe separates the Figshare and BR35H repositories with 0.9997 balanced accuracy, making two errors in 3,645 images and recovering every BR35H image. The region contains no brain tissue, so tumour phenotype cannot explain the result. Holding the diagnosis constant at meningioma and comparing Figshare against SARTAJ gives 0.9843, which removes the confound by design and corroborates the finding independently.

The three-way balanced figures are lower only because SARTAJ recall is 0.696–0.798: SARTAJ is largely a re-encoding of Figshare, so confusion between the two is expected and is itself evidence of that relationship.

Because the no-tumour class is drawn exclusively from BR35H, its label is strongly associated with source-specific, non-pathological image characteristics. We do not attribute these to scanner or acquisition parameters specifically, since cropping, padding, resampling and compression are equally consistent with the finding. This bears directly on the ~99% figures routinely reported for that class, including in our own earlier work.

Masking the background does not remove the confound: the repository remains recoverable from the head region alone at 0.980 recall for BR35H, indicating that source-specific signal persists within the head and cannot be explained by external framing alone.

### 4.5 Attribution

**Table 10. Occlusion attribution on the deployed hybrid pipeline (EfficientNetB0+SVM), Figshare-only patient-level cohort, 100 images per class.**

| Class | Concentration, correct | Concentration, incorrect | Pointing | IoU |
|---|---|---|---|---|
| Meningioma | 6.54 | 2.28 | 0.310 | 0.183 |
| Glioma | 2.29 | 0.51 | 0.071 | 0.044 |
| Pituitary | 1.47 | 0.21 | 0.010 | 0.014 |

Concentration exceeds the uniform-map baseline of 1.0 for all three classes and is three- to sevenfold higher on correct than on incorrect predictions. This is consistent with greater dependence on lesion-containing regions when the prediction is correct. We are careful not to overstate it: an attribution map does not establish clinically valid reasoning, and localisation remains incomplete, particularly for pituitary lesions whose masks occupy under 1% of the image.

Gradient-based attribution on the backbone and occlusion on the deployed pipeline disagreed sharply for pituitary tumours. On the four-class cohort, pooling correct and incorrect predictions so that the two methods are compared on the same images, mean concentration is 0.28 under Grad-CAM against 2.25 under occlusion; on the Figshare-only cohort the corresponding values are 0.79 and 1.41. The discrepancy likely reflects, at least in part, the coarse 7×7 grid on which Grad-CAM for this architecture is computed — a single cell of which is larger than a pituitary mask — together with the fact that Grad-CAM targets the network's own classifier rather than the hybrid decision boundary. We report the occlusion figures for the hybrid and describe the limitation rather than reconciling the two methods.

For the same reason we do not report the absolute spatial spread of attribution. The share of attribution falling outside the head is approximately 0.50 for every class, but the background occupies 49.9% of the resized image; at this grid resolution the metric carries no information and is withdrawn.

### 4.6 Calibration and selective prediction

Aggregating slice-level posteriors within a leakage-control group raises accuracy of the standalone network from 0.9287 to 0.9459 with no other change, which is relevant because the unit of clinical decision is the examination rather than the slice.

Group-level expected calibration error is 0.019 for the SVM head and 0.021 for logistic regression, against 0.049 for the network's softmax; the random-forest head reaches 0.185 and its scores should not be treated as probabilities. Calibration and ranking quality are distinct properties: by area under the risk–coverage curve the logistic head ranks uncertainty best (0.0076), the network is intermediate (0.0100), and the SVM is worst among the linear heads (0.0155). The nearest-neighbour head selects k = 1 in some folds, yielding degenerate confidences, and is excluded from this analysis.

As an exploratory analysis, referring the least-confident 20% of patients in the Figshare-only patient-level cohort raises accuracy on the retained 80% from 0.910 to 0.968, transferring 482 of the 3,038 images to a radiologist. We report this as selective prediction and make no claim of clinical utility on that basis; accuracy on retained cases is conditional on the model's own confidence ordering and does not transfer to an unselected population.

### 4.7 External contamination screening

Neither candidate cohort was usable.

**PMRAM** is described by its repository as 1,600 raw images, 400 per class, collected from hospitals in Bangladesh. The archive we downloaded contained 1,505 image files in the raw directory, distributed as 373 glioma, 363 meningioma, 396 no-tumour and 373 pituitary; 184 of them, in 89 groups, are byte-identical duplicates of one another. We report both the published and the observed counts rather than substituting one for the other.

Against the source repositories, 509 images (33.8%) match at ≤2 bits, 408 of them at distance zero across both hashes, with class-wise overlap ranging from 0.6% for meningioma (2 of 363) to 67.7% for no-tumour (268 of 396). Against the composite redistribution — the dataset on which the published literature on this benchmark is trained — 844 images (56.1%) match, including all 373 pituitary images. No byte-identical file was found in either comparison, which is consistent with re-encoding on redistribution. Matched pairs agree on diagnostic label in 100.0% of cases against a chance rate of 25%, and of 400 pairs verified by structural similarity, 93.8% reach SSIM ≥ 0.95 with a median of 0.997.

Excluding the matched groups against the source repositories would leave 342 glioma, 361 meningioma, 165 pituitary and 128 no-tumour images. A four-class cohort therefore survives arithmetically, but we do not use it: more than half the release matches the benchmark the literature trains on, the two classes most depleted are precisely those most contaminated, and the provenance of the remainder is unestablished rather than established.

**BDNeuro-MRI v7** is described as clinical data from named hospitals in Bangladesh. The archive contains 5,944 image files, of which 3 lie outside any class directory and are excluded, leaving 5,941 study images in publisher-defined partitions of 4,160 / 892 / 889 — matching the composition the repository reports. Of these, 2,242 (37.7%) match the source repositories and 4,027 (67.7%) match the composite redistribution, with 1,831 and 3,315 of those respectively being byte-identical files. Against the 3,813 images actually used for fitting, 2,009 images (33.8%) match, 165 of them byte-identical. Matched pairs agree on diagnostic label in 100.0% of cases against a chance rate of 20%. Of 400 matched pairs verified by structural similarity, 90.0% reach SSIM ≥ 0.95 with a median of 1.000. The publisher-defined test partition is contaminated at the same rate as the release as a whole (37.1%), so restriction to it does not help.

One finding here is of general relevance. BDNeuro-MRI states that duplicates were removed between its partitions, and our audit largely corroborates this: one two-image near-duplicate group out of 5,939 spans publisher partitions, and no test image has a near-duplicate counterpart in the publisher's training or validation partition. The same release nevertheless shares 1,831 byte-identical files with the source repositories of this benchmark. Internal partition hygiene and independence from other public repositories are distinct properties established by different audits, and only the first appears to be routine.

We report substantial content-level overlap and make no inference about how it arose.

### 4.8 Leakage measured on a third-party release

The BDNeuro-MRI test partition was used not as an external cohort but as a natural experiment. Its 889 images stratify by relationship to our development data: 288 match the images actually used for fitting, 42 match the source repositories but not the fitted set, and 559 match neither. Of the 559, 70.7% lie at a nearest distance of 12 bits or more.

**Observational contrast.** Balanced accuracy is higher on the matching stratum than on the non-matching one for all six models, by +4.18 to +8.23 points, with all intervals excluding zero (Supplementary Table S9). Restricting the comparison group to images at nearest distance ≥12 bits attenuates the difference to approximately 2 points, with intervals spanning zero. The attenuation may reflect both the reduced sample size of that subset (395 images, interval widths of approximately ±4 points) and heterogeneity within the non-overlapping cohort; we do not interpret it as demonstrating a monotonic relationship between hash distance and performance.

**Within-image paired contrast.** Because folds are group-disjoint, the leakage-control group containing a matched image's counterpart was held out of exactly one of the five backbones. For each of 2,009 matched images across the whole release, four backbones were fitted with that group present and one without it. The image is then its own control: class, image quality, preprocessing and architecture are identical across the comparison, and each backbone is fitted on four fifths of the development data. Fold-specific training sets are not otherwise identical, which is what the negative control addresses.

**Table 11. Within-image paired contrast, and negative control.**

| Model | Exposed | Unexposed | Difference (pp) | 95% CI | p |
|---|---|---|---|---|---|
| EfficientNetB0 | 0.9373 | 0.8810 | +5.62 | [4.48, 6.74] | 0.0005 |
| + SVM | 0.9454 | 0.8850 | +6.04 | [4.90, 7.22] | 0.0005 |
| + Logistic regression | 0.9505 | 0.8825 | +6.79 | [5.64, 8.01] | 0.0005 |
| *Control:* EfficientNetB0 | 0.8527 | 0.8519 | +0.08 | [−0.80, +0.99] | 0.871 |
| *Control:* + SVM | 0.8617 | 0.8592 | +0.25 | [−0.56, +1.05] | 0.557 |
| *Control:* + Logistic regression | 0.8692 | 0.8713 | −0.22 | [−0.99, +0.55] | 0.587 |

The value 0.0005 is the smallest two-sided *p* attainable with 4,000 bootstrap resamples under the (extreme + 1)/(resamples + 1) convention; it is reported as an attained bound rather than as an exact value. Holm-adjusted values across the three models are 0.0015.

Mean confidence is also higher on the exposed side (0.922 against 0.876 for the standalone network), which is consistent with memorisation but does not on its own establish it.

The negative control repeats the identical procedure on 3,699 non-overlapping images with randomly assigned fold labels. All intervals span zero with widths of approximately 1.8 points, so the design would detect an effect several times smaller than the one observed. The leave-one-fold-out estimator does not manufacture a difference of its own, and variation between backbones does not account for the effect.

We describe this as leakage-associated inflation rather than inflation caused by leakage. Overlap status is not randomly assigned across images; what the paired design establishes is that the association survives holding the image itself constant.

---

## 5. Discussion

### 5.1 What the three measurements show together

Table 12 assembles the three estimates. They differ in what they hold constant, and therefore in which alternative explanations they exclude.

**Table 12. Three estimates of evaluation inflation.**

| Design | Contrast | Confounds held constant | Estimate |
|---|---|---|---|
| Internal, pre-specified | Image-level vs group-disjoint partitioning | Model, data, code, seed | +3.1 to +5.7 pp |
| Third-party, observational | Overlapping vs non-overlapping images | Publisher, preprocessing, model | +4.2 to +8.2 pp |
| Third-party, within-image paired | Same image, backbones fitted with vs without its counterpart | All of the above, plus the image itself | +5.6 to +6.8 pp |

The first estimate is open to the objection that the two partitions differ in more than leakage. The second is open to the objection that overlapping and non-overlapping images may differ systematically. The third substantially reduces both objections by holding the evaluated image constant, and the negative control did not reproduce the effect under randomised fold labels. It agrees with the first two.

### 5.2 Implications for the literature on this benchmark

Accuracies of 98–99% are reported on this benchmark with image-level splitting and without duplicate removal. Our pipeline reaches 0.9756 under that protocol and 0.9405 with group-disjoint partitioning, using the same architecture, training pipeline, hyperparameter search space, code and seed, with the backbone retrained within each partitioning. We do not claim that published figures are wrong as computed; we claim that they estimate a different quantity from the one they are taken to estimate.

Two further observations bear on comparative claims. First, relative model ranking is not stable to the protocol: the nearest-neighbour head is the best model under image-level splitting and the worst under group-disjoint splitting. Comparisons between architectures conducted under the flawed protocol may not survive the corrected one. Second, leakage narrows confidence intervals as well as raising means, so the apparent precision of published figures is itself partly an artefact.

### 5.3 The no-tumour class

The no-tumour class derives from a single repository with strong source-specific image characteristics distinguishable from those of the tumour repositories, at 0.9997 balanced accuracy from background pixels alone. Its per-class figures — 0.997 sensitivity here, ~99% in the wider literature — cannot be interpreted as diagnostic performance without that qualification. The benchmark therefore permits high performance on this class from source-correlated cues alone, which makes a diagnostic reading of the no-tumour result non-identifiable without source-balanced data. This is a property of the benchmark, not of any model trained on it, and it cannot be removed by better modelling. Masking the background does not help, since source signal persists within the head. Removing the confound would require the classes to be resampled across repositories, which these repositories do not permit.

### 5.4 Hybrid classifiers

The hybrid improvement over the standalone network is real but small, and it is detectable only under the corrected protocol: under image-level splitting it attenuates to 0.66 points (p = 0.024). The practical case for the hybrid rests less on accuracy than on calibration, where the linear heads produce posteriors two to three times better calibrated than the network's softmax. Calibration and ranking quality are nonetheless distinct, and the two do not agree on which head is preferable.

The original framing of this work — that a specific downstream classifier is superior — is not supported. What is supported is that the choice of decision layer matters somewhat, that no single choice dominates, and that claims of superiority for a particular classifier require paired testing under a leakage-controlled protocol.

### 5.5 Provenance as a reporting requirement

Several assumptions suggested by repository descriptions or by common use were contradicted by content-level provenance screening: that the benchmark's three repositories are independent sources; that a revised release had eliminated train–test overlap at the level that matters; and that two further datasets constitute independent clinical cohorts. In each case the audit took minutes once the tooling existed.

The BDNeuro-MRI case is the clearest illustration of the gap. Its publisher performed exactly the audit that is conventional — internal duplicate removal between partitions — and performed it well. What was not done, and appears not to be routine anywhere, is a check against other public repositories. We suggest that cross-repository provenance screening should be reported alongside internal deduplication for any redistributed medical imaging dataset, and we release tooling so that the check is inexpensive.

---

## 6. Limitations

- **No independent external validation was performed.** We screened the candidate public four-class cohorts we could identify and found substantial content-level overlap in each. This is the principal limitation of the work, and, we would argue, of the literature built on this benchmark.
- **Patient identifiers exist for one of three repositories.** Elsewhere, leakage-control groups are inferred from image similarity and are not patients. Intervals for those classes remain optimistic if the repositories contain multiple non-duplicate slices per patient.
- **The no-tumour class is confounded with source**, as set out in Section 5.3.
- **Glioma contributes 1,418 images from only 89 patients.** Class weighting operates at the image level and therefore overstates the diversity of that class.
- **Hyperparameter selection for the classical heads carries residual optimism.** They are tuned by cross-validation inside the outer training portion, part of which the backbone saw during fine-tuning. The outer test fold remained untouched by model selection in either case.
- **The analysis is retrospective and slice-based rather than volumetric**, and no comparison against radiologist performance was performed.
- **Attribution does not establish reasoning.** Concentration measures where a decision is sensitive to occlusion, not whether the decision is clinically valid.
- **The redistributed benchmark is mutable.** We report the version used and reconstructed provenance from the source repositories for that reason.

---

## 7. Conclusion

Performance on the most widely used brain tumour MRI benchmark is materially overestimated by the evaluation protocol in common use. Reconstructing patient identifiers from the original archives shows that its constituent repositories are not independent, that a majority of recoverable patients span its published train/test boundary, and that 3,813 unique images resolve to 721 leakage-control groups. Correcting the partitioning reduces accuracy from 0.9756 to 0.9405 with no other change, and three complementary designs — one internal, two on a third-party release, one of which uses each image as its own control — place the inflation at between three and seven percentage points.

One of the four classes is separable at 0.9997 balanced accuracy from image regions containing no brain tissue, so its near-perfect per-class figures measure source recognition at least as much as diagnosis. Attaching a conventional classifier to pooled deep features improves accuracy by approximately one point and calibration by rather more, but no downstream classifier dominates and relative ranking is not stable to the evaluation protocol.

We did not identify a verifiably independent public four-class cohort among the candidates screened, and we report that screening in place of an external evaluation rather than presenting a contaminated one. These results support technical feasibility under a leakage-controlled retrospective benchmark; substantial source confounding, the absence of verifiable external validation, and class-dependent attribution quality preclude claims of clinical readiness. Genuine external validation of this task requires institutional data that has not been redistributed.

---

## 8. Mandatory statements

### 8.1 CRediT authorship contribution statement

*To be completed by the authors to reflect the revised study.*

### 8.2 Declaration of competing interest

The authors declare no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### 8.3 Funding

This research did not receive a specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

### 8.4 Ethical approval and informed consent

This study underwent ethical review by the Bournemouth University Ethics Board (Reference 66271). The research uses retrospective, open-source, fully anonymised public databases; a dedicated informed consent statement was not required.

### 8.5 Data and code availability

Source datasets: Figshare [25], SARTAJ [26], BR35H [27]. Candidate external cohorts screened: PMRAM (Mendeley m7w55sw88b) and BDNeuro-MRI v7 (Mendeley zwr4ntf94j), downloaded on *[date]*.

All code is released under an open licence at *[repository URL]*, archived with a persistent identifier at *[DOI]*, including provenance reconstruction, deduplication and grouping, split generation with verification, training and evaluation, attribution analysis, and the external contamination audit. The release includes configuration files, the dependency lock file, split assignments, per-image out-of-fold predictions, and audit manifests sufficient to reproduce every figure in this manuscript from the source repositories. The external screening protocol was committed before the screening was run.

### 8.6 Reporting guidance

A completed CLAIM 2024 checklist is provided as supplementary material. Where an item could not be satisfied — external testing, and acquisition parameters for two of three repositories — this is recorded with the reason rather than left blank. Reporting was additionally informed by STARD-AI where applicable.

---

## References

*[References 1–30 retained from the original submission. New references to be added for: CLAIM 2024; STARD-AI 2025; data leakage in medical imaging deep learning; near-duplicate contamination in benchmark datasets; shortcut learning; selective prediction and risk–coverage analysis; perceptual hashing.]*

---

## Supplementary material

- **S1** Per-source, per-class image counts before and after deduplication
- **S2** Deduplication threshold calibration, including the glioma negative control
- **S3** Grouping threshold sensitivity sweep, 4–12 bits
- **S4** Exact class-wise counts for every partition of every fold, with stored augmented files recorded as zero
- **S5** Hyperparameter grids and values selected in each fold
- **S6** Per-class metrics with 95% group-bootstrap intervals, all models, all configurations
- **S7** Confusion matrices with real counts, all models
- **S8** Fold-level results and between-fold spread
- **S9** External screening: distance distributions, class-wise overlap, SSIM confirmation, stratified results
- **S10** Completed CLAIM 2024 checklist
