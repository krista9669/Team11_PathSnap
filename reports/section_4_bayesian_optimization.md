# Section 4: Bayesian Hyperparameter Optimisation — Resource-Limited Pilot

## Scope and limitation

The requested full-data run could not be completed within the 30-minute deadline. The full 300-tree baseline was stopped before a complete full-data CV result was available; no incomplete-fold score is reported. This is a **pilot**, using six representative trips selected across the existing training feature-file size distribution, three fixed GroupKFold folds, and the same Random Forest preprocessing. It does not replace a full-data final experiment. All 80 training alignments and the ten student feature/alignment sets were prepared using the repository’s existing alignment and candidate/feature code. Student-testing data was not loaded by the tuning objective, and evaluator-testing data was not accessed.

## Assignment questions

1. **Hyperparameters searched:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `criterion`. `class_weight='balanced'` and the median imputer were kept fixed.
2. **Pilot search ranges:** `n_estimators` integer-uniform 50–150, step 50; `max_depth` categorical {None, 4, 8, 12, 16}; `min_samples_split` integer-uniform 2–10; `min_samples_leaf` integer-uniform 1–5; `max_features` categorical {sqrt, log2, 0.5}; `criterion` categorical {gini, entropy}. This reduced search was used to fit the deadline. The original 30-trial target was not completed.
3. **Objective:** maximise mean Macro F1 over the same three trip-grouped folds for every trial and the untuned pilot baseline.
4. **Trials:** 30 planned; 3 completed; 0 failed; 0 pruned. Trial rows were checkpointed to `optimization_trials.csv`.
5. **Best parameters:** trial 0: `{"n_estimators": 100, "max_depth": null, "min_samples_split": 2, "min_samples_leaf": 5, "max_features": "log2", "criterion": "gini", "class_weight": "balanced", "n_jobs": -1, "random_state": 42}`. Its mean grouped CV Macro F1 was 0.842600.
6. **Did tuning improve the model?** The pilot CV objective increased by +0.013054, but student Top-1 Accuracy, Macro F1, and Log Loss were worse. The tuned model was faster at inference. The pilot does not establish that tuning improves the full-data model.

## Official baseline configuration

The official untuned Random Forest remains `n_estimators=300`, `max_depth=None`, `min_samples_split=4`, `min_samples_leaf=2`, `class_weight='balanced'`, `n_jobs=-1`, and `random_state=42`, with median imputation. `max_features='sqrt'` and `criterion='gini'` are the scikit-learn defaults in the existing implementation. The pilot baseline preserves this exact configuration; only the training subset was reduced.

## Untuned versus tuned pilot results

| Model | Grouped CV Macro F1 | Student Top-1 Accuracy | Student Macro F1 | Student Log Loss | Inference Time (s) | Candidate Recall |
|---|---:|---:|---:|---:|---:|---:|
| Untuned Random Forest (300 trees) | 0.829546 | 0.812063 | 0.878550 | 0.068577 | 9.371431 | 0.999478 |
| Tuned pilot Random Forest | 0.842600 | 0.810496 | 0.871397 | 0.083751 | 3.191610 | 0.999478 |

The tuned pilot gained +0.013054 CV Macro F1. Student Top-1 changed by -0.001567; student Macro F1 changed by -0.007153; Log Loss changed by +0.015174 (positive is worse); inference time changed by -6.179822 seconds. Candidate recall is shared because both models use the same candidates and aligned reference: 0.999478.

Top-1 Accuracy follows the existing methodology: select the highest predicted-probability candidate per GPS observation, breaking ties by distance, and compare it with the alignment-derived positive candidate. Macro F1 and Log Loss use the usable aligned student candidate rows. These labels are provisional alignment-derived references, not independently verified ground truth. Median imputation is fit inside each training fold. A preceding global unsupervised feature-selection pass examined training feature files; the model then uses a hard-coded feature set rather than nesting feature selection inside each fold. If that analysis informed the fixed set, held-out pilot feature distributions may have contributed mild transductive information. The imputer is fold-local and no held-out trip labels entered the tuning objective. Fully nested selection requires a new experiment.

## Reproduction and remaining full-data work

Install `requirements.txt`, including Optuna, use Optuna 5.1.0 and set the notebook `RAW` variable to the data root containing `train` and `student_test`, and execute the notebook from the repository root. The notebook supports resuming the saved pilot checkpoints; set `RESUME_PILOT_CHECKPOINTS=False` to rerun the pilot. For the full experiment, rerun the original 300-tree baseline and TPE search using all 80 aligned training trips, keep the fixed group folds shared, and then refit/evaluate both models on the same student partition. The full-data baseline and final evaluation are explicitly incomplete in `optimization_summary.json`.
