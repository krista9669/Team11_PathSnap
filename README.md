# PathSnap
## Machine Learning-Based GPS Map Matching

PathSnap is a machine learning-based map-matching system designed to identify the most likely road segments corresponding to noisy GPS observations. The project combines spatial candidate generation, feature engineering, feature selection and multiple machine learning models to reconstruct vehicle trajectories over a road network.

---

## Team Members

| Sl. No. | University ID |
|:---:|---|
| 1 | PES2UG24CS231 |
| 2 | PES2UG24CS232 |
| 3 | PES2UG24CS233 |
| 4 | PES2UG24CS234 |

---

## 1. Problem Statement

GPS trajectories frequently contain positional inaccuracies caused by signal interference, measurement noise and urban environments. As a result, recorded GPS coordinates may not lie exactly on the roads travelled by a vehicle.

Map matching addresses this problem by associating GPS observations with the most likely road segments in a given road network.

PathSnap investigates machine learning approaches to identify the most probable road segment for each GPS observation and reconstruct the corresponding vehicle trajectory.

## 2. Project Objectives

- Preprocess and validate GPS trajectory and road-network data.
- Generate plausible road-segment candidates for each GPS observation.
- Extract meaningful spatial and movement-related features.
- Perform feature selection to identify informative features and reduce redundancy.
- Train and compare five different machine learning models.
- Evaluate model performance using classification and candidate-ranking metrics.
- Investigate route alignment and continuity for sequence-level map matching.

## 3. Dataset Description

The dataset consists of GPS trajectories and corresponding road-network information.

Each trip contains four files:

| File | Description |
|---|---|
| `.track` | GPS longitude, latitude and timestamp observations |
| `.nodes` | Geographic coordinates of road-network nodes |
| `.arcs` | Connections between road-network nodes |
| `.route` | Reference sequence of road-segment IDs |

### Dataset Partitions

| Partition | Number of Trips | Purpose |
|---|---:|---|
| Training | 80 | Model development and training |
| Student Testing | 10 | Development and comparison |
| Evaluator | Reserved | Not used during development |

The raw dataset is maintained separately from the source repository.

## 4. Methodology

The PathSnap pipeline consists of data preprocessing, candidate generation, feature extraction, feature selection, route alignment, model training and evaluation.

### 4.1 Data Preprocessing

The preprocessing stage prepares raw GPS and road-network data for subsequent analysis.

The operations include:

- Loading and validating trip files.
- Checking missing and invalid values.
- Identifying duplicate GPS observations.
- Validating timestamps and time intervals.
- Analysing temporal gaps in GPS trajectories.
- Checking road-network connectivity.
- Calculating graph statistics.
- Visualizing GPS trajectories and road networks using Folium.

The preprocessing stage helps identify data quality issues before candidate generation and feature extraction.

### 4.2 Candidate Generation

Candidate generation identifies nearby road segments that could potentially correspond to each GPS observation.

The implemented approach uses spatial indexing and geometric distance calculations.

The process involves:

1. Sampling points along road segments at approximately 30-metre intervals.
2. Converting geographic coordinates into unit-sphere XYZ coordinates.
3. Constructing a spatial index using SciPy's `cKDTree`.
4. Performing progressive spatial-radius searches.
5. Identifying unique candidate road-segment IDs.
6. Refining candidate distances using point-to-segment distance calculations.
7. Ranking candidates and retaining the Top-20 road segments.

This reduces the search space before machine learning is applied.

Candidate coverage is an important consideration because a model cannot select the correct road segment if it is absent from the generated candidate set.

### 4.3 Feature Engineering

Feature extraction describes the spatial and movement-related relationship between GPS observations and candidate road segments.

The engineered features include:

| Feature | Description |
|---|---|
| `distance_m` | Distance between GPS observation and candidate road segment |
| `gps_speed_mps` | Estimated GPS movement speed |
| `gps_displacement_m` | Distance between consecutive GPS observations |
| `time_gap_s` | Time difference between consecutive observations |
| `gps_bearing` | Direction of GPS movement |
| `segment_length_m` | Length of candidate road segment |
| `segment_bearing` | Direction of candidate road segment |
| `heading_difference` | Difference between GPS and road-segment directions |
| `projection_distance_m` | Distance associated with projection onto the segment |
| `projection_fraction` | Relative position of the projected point along the segment |

Circular direction features can additionally be represented using sine and cosine transformations.

These features provide information about the spatial proximity, direction and movement characteristics of possible road-segment matches.

### 4.4 Feature Selection

Feature selection identifies informative features while reducing redundancy and multicollinearity.

The analysis includes:

- Variance threshold analysis.
- Correlation analysis.
- Spearman correlation heatmaps.
- High-correlation pair identification.
- Variance Inflation Factor (VIF) analysis.
- Domain-based feature selection.

The selected features are used to support subsequent model development.

### 4.5 Route Alignment and Label Generation

Reference route information is used to derive provisional candidate labels through route alignment.

Monotonic alignment associates GPS observations with positions along the reference route while maintaining non-decreasing route progression.

Candidate labels are generated in memory by comparing candidate arc IDs with aligned route arc IDs.

These labels support supervised model training and evaluation.

The alignment-derived labels are provisional and should not be interpreted as independently verified real-world predictions.

## 5. Machine Learning Models

Five classification models are implemented and evaluated.

| Model | Learning Approach | Main Characteristics |
|---|---|---|
| Logistic Regression | Linear probabilistic classification | Interpretable baseline model |
| Gaussian Naive Bayes | Bayesian probabilistic classification | Uses conditional independence assumptions |
| Random Forest | Bagging-based ensemble learning | Combines multiple decision trees |
| LightGBM | Gradient boosting | Sequentially improves tree-based predictions |
| Multi-Layer Perceptron (MLP) | Artificial neural network | Learns nonlinear relationships through hidden layers |

### 5.1 Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of an observation belonging to a particular class.

It uses a sigmoid function to transform a weighted combination of input features into a probability between zero and one.

### 5.2 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.

It assumes that features are conditionally independent given the class and that continuous features follow Gaussian distributions.

### 5.3 Random Forest

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

It uses bootstrap sampling and random feature selection to build diverse trees and aggregates their predictions through majority voting.

### 5.4 LightGBM

LightGBM is a gradient boosting framework based on decision trees.

It builds trees sequentially, with each successive tree contributing to the correction of errors made by the existing ensemble. It uses histogram-based learning and leaf-wise tree growth for computational efficiency.

### 5.5 Multi-Layer Perceptron

A Multi-Layer Perceptron is a feedforward artificial neural network consisting of input, hidden and output layers.

It learns relationships between input features through forward propagation and updates its weights using backpropagation and an optimization algorithm.

## 6. Model Configurations

The following configurations were used in the current experimental run.

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight='balanced'` |
| Gaussian Naive Bayes | GaussianNB with median imputation and standard scaling |
| Random Forest | `n_estimators=300`, `min_samples_split=4`, `min_samples_leaf=2`, `class_weight='balanced'` |
| LightGBM | `n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `class_weight='balanced'` |
| MLP | `hidden_layer_sizes=(64,32)`, `activation='relu'`, `solver='adam'`, `learning_rate_init=0.001`, `max_iter=300`, `early_stopping=True` |

Median imputation and standard scaling are applied through the model pipelines where configured.

## 7. Experimental Results

The following results were obtained from the current model-training run.

### 7.1 Classification Performance

| Model | Accuracy | Precision | Recall | Binary F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7839 | 0.1776 | 0.9088 | 0.2971 | 0.5847 |
| Gaussian Naive Bayes | 0.9480 | 0.4399 | 0.1258 | 0.1957 | 0.5844 |
| Random Forest | **0.9716** | 0.7290 | 0.6908 | **0.7094** | **0.8472** |
| MLP | 0.9707 | **0.7421** | 0.6403 | 0.6875 | 0.8361 |
| LightGBM | 0.9479 | 0.4899 | **0.8803** | 0.6295 | 0.8008 |

### 7.2 Training and Inference Time

| Model | Training Time (seconds) | Inference Time (seconds) |
|---|---:|---:|
| Logistic Regression | 8.60 | 0.45 |
| Gaussian Naive Bayes | 6.23 | 1.03 |
| Random Forest | 849.59 | 19.25 |
| MLP | 1719.57 | 10.89 |
| LightGBM | 31.67 | 11.19 |

### 7.3 Additional Evaluation Metrics

| Model | Log Loss | Observation Top-1 Accuracy |
|---|---:|---:|
| Logistic Regression | 0.4066 | 0.5164 |
| Gaussian Naive Bayes | 0.1793 | 0.6115 |
| Random Forest | 0.0989 | **0.7303** |
| MLP | **0.0843** | 0.7214 |
| LightGBM | 0.1362 | 0.7261 |

### 7.4 Confusion Matrices

The confusion matrices below use the order:

`[[True Negative, False Positive], [False Negative, True Positive]]`

**Logistic Regression**

```text
[[785406, 225056],
 [  4875,  48586]]
```

**Gaussian Naive Bayes**

```text
[[1001899, 8563],
 [  46735, 6726]]
```

**Random Forest**

```text
[[996734, 13728],
 [ 16529, 36932]]
```

**MLP**

```text
[[998568, 11894],
 [ 19230, 34231]]
```

**LightGBM**

```text
[[961464, 48998],
 [  6397, 47064]]
```

### 7.5 Result Analysis

The experimental results indicate differences in the performance and computational requirements of the five models.

- **Random Forest** achieved the highest Macro F1 score of 0.8472 and observation-level Top-1 accuracy of 73.03%, providing the strongest overall classification performance in the current experiment.
- **MLP** achieved the highest precision of 74.21% and the lowest log loss of 0.0843, but required the longest training time.
- **LightGBM** achieved the highest recall of 88.03% while maintaining a relatively short training time of approximately 31.67 seconds.
- **Logistic Regression** achieved high recall but comparatively low precision, indicating a larger number of false-positive predictions.
- **Gaussian Naive Bayes** achieved high overall accuracy but low recall, demonstrating the limitations of relying on accuracy alone for imbalanced classification.

Random Forest currently provides the strongest overall classification performance, while LightGBM offers a promising balance between computational efficiency and predictive performance.

These results represent the current experimental run and require validation across trips and against independently verified route-level outcomes.

## 8. Evaluation Metrics

The following metrics are used to assess model performance.

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified candidate rows |
| Precision | Measures the proportion of predicted positive candidates that are correct |
| Recall | Measures the proportion of actual positive candidates identified |
| Binary F1-score | Harmonic mean of precision and recall for the positive class |
| Macro F1-score | Computes F1 independently for each class and averages the results |
| Confusion Matrix | Summarizes correct and incorrect class predictions |
| Log Loss | Measures the quality of predicted class probabilities |
| Observation Top-1 Accuracy | Measures whether the highest-ranked candidate matches the derived positive label |
| Training Time | Measures the time required to train a model |
| Inference Time | Measures the time required to generate predictions |

Macro F1 is particularly useful because it gives equal importance to both classes, helping account for class imbalance.

Observation-level Top-1 accuracy evaluates candidate ranking separately for each GPS observation. It is distinct from binary classification accuracy, which evaluates each candidate independently.

## 9. Project Structure

The repository is organized into source code, processed datasets and generated outputs.

```text
Team11_PathSnap/
│
├── data/
│   └── processed/
│       ├── dataset summary CSV files
│       ├── timestamp analysis CSV files
│       ├── graph statistics CSV files
│       ├── trip_*_candidates.csv
│       ├── trip_*_features.csv
│       ├── trip_*_monotonic_alignment.csv
│       └── other processed CSV files
│
├── outputs/
│   │
│   ├── feature_selection/
│   │   └── Feature selection reports and visualizations
│   │
│   ├── model_results/
│   │   └── Model evaluation results and generated outputs
│   │
│   └── trip_00000001.html
│       └── Interactive Folium trajectory visualization
│
├── src/
│   ├── batch_alignment.py
│   ├── batch_feature_extraction.py
│   ├── candidate_generation.py
│   ├── data_processing.py
│   ├── feature_extraction.py
│   ├── feature_selection.py
│   ├── ground_truth_alignment.py
│   ├── inspect_midpoint_ranks.py
│   ├── inspect_missing_candidates.py
│   ├── monotonic_alignment.py
│   ├── train_models.py
│   ├── validate_alignment.py
│   └── validate_candidates.py
│
├── .gitignore
└── README.md
```

### 9.1 Source Code Description

| File | Responsibility |
|---|---|
| `data_processing.py` | Loads, validates and analyses GPS and road-network data |
| `candidate_generation.py` | Generates nearby road-segment candidates using spatial indexing |
| `batch_feature_extraction.py` | Performs candidate generation and feature extraction across multiple trips |
| `feature_extraction.py` | Extracts spatial and movement-related features |
| `feature_selection.py` | Performs statistical feature analysis and selection |
| `ground_truth_alignment.py` | Implements ground-truth route alignment functionality |
| `monotonic_alignment.py` | Performs monotonic GPS-to-route alignment |
| `batch_alignment.py` | Generates alignment outputs across multiple trips |
| `train_models.py` | Trains and evaluates the five machine learning models |
| `validate_candidates.py` | Validates candidate-generation results |
| `validate_alignment.py` | Validates route-alignment results |
| `inspect_midpoint_ranks.py` | Investigates candidate ranking and midpoint-based retrieval |
| `inspect_missing_candidates.py` | Investigates missing candidate road segments |
| `src/hmm/emission_probabilities.py` | Scores road candidates and normalizes per-observation RF emissions |
| `src/hmm/transition_probabilities.py` | Builds road-network transition scores and matrices |
| `src/hmm/viterbi.py` | Reconstructs one road-arc sequence per GPS trip |
| `src/hmm/evaluate_viterbi.py` | Compares Viterbi and classifier-only predictions |
| `src/hmm/run_pipeline.py` | Runs and summarizes a bounded HMM pilot batch |

### 9.2 Directory Description

**`data/processed/`**

Contains intermediate and processed CSV files generated during the preprocessing pipeline, including:

- Dataset and trip summaries.
- Timestamp analysis.
- Road-network statistics.
- Candidate generation results.
- Extracted feature data.
- Route alignment outputs.

**`outputs/feature_selection/`**

Stores feature-selection results, statistical analysis, correlation visualizations and generated reports.

**`outputs/model_results/`**

Stores machine learning model evaluation results, performance comparisons, confusion matrices and other generated model outputs.

**`outputs/`**

Contains generated visualizations, including interactive Folium maps of GPS trajectories.

### 9.3 HMM Route Reconstruction

The HMM pipeline turns Random Forest candidate probabilities into per-observation
emissions, builds road-network transition probabilities, and uses Viterbi
decoding to select one road arc per GPS observation. The evaluation compares
those predictions with Random Forest Top-1 and reports per-trip and pooled
arc-level metrics, selected-transition reachability, GPS-to-road distance,
runtime and output size.

Run the staged pilot from the repository root:

```powershell
python -m src.hmm.run_pipeline
```

By default, this processes only Trips 5, 13 and 16. A different explicit
subset can be supplied with `--trip-ids`, for example
`python -m src.hmm.run_pipeline --trip-ids 5 13 16`. It does not start the
remaining validation trips.
Outputs are written under `outputs/hmm_results/`, with per-trip evaluation
files and a `pilot_batch_summary.csv` under `outputs/hmm_results/evaluation/`.

The designated ten-trip student-test partition can be evaluated separately
with the already-trained model:

```powershell
python -m src.hmm.run_pipeline --partition student_test
```

This prepares that partition's candidate features and route alignments from
`Student folder/student_test/` and writes its aggregate summary to
`outputs/hmm_results/evaluation/student_test_hmm_summary.csv`. It does not
retrain the classifier or access the evaluator-testing partition.

The monotonic-alignment arc labels used for evaluation are provisional,
route-derived references rather than independently verified ground truth.
Results should be described with this limitation; the student-test partition
should be used only as specified by project guidelines, and the
evaluator-testing partition must not be used for parameter selection.

## 10. Installation and Execution

### 10.1 Clone the Repository

```bash
git clone <repository-url>
cd Team11_PathSnap
```

### 10.2 Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

### 10.3 Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses the following primary Python libraries:

```text
numpy
pandas
scikit-learn
scipy
matplotlib
networkx
folium
lightgbm
```

### 10.4 Execute the Pipeline

Ensure that the raw dataset is available at the configured data path.

The main processing workflow can be executed in the following order:

**Step 1: Data preprocessing**

```bash
python src/data_processing.py
```

**Step 2: Batch candidate generation and feature extraction**

```bash
python src/batch_feature_extraction.py
```

**Step 3: Feature selection**

```bash
python src/feature_selection.py
```

**Step 4: Batch route alignment**

```bash
python src/batch_alignment.py
```

**Step 5: Model training and evaluation**

```bash
python src/train_models.py
```

Candidate generation, feature extraction and alignment can also be executed individually when required.

Validation and diagnostic scripts can be run separately to inspect candidate coverage, alignment quality and intermediate processing results.

Run the staged HMM pilot after the Random Forest model and trip-level
alignment/feature files are available:

```powershell
python -m src.hmm.run_pipeline
```

To run the separate ten-trip student-test evaluation, use:

```powershell
python -m src.hmm.run_pipeline --partition student_test
```

## 11. Future Improvements

Potential directions for further development include:

- Improving candidate-generation coverage and spatial-search efficiency.
- Performing trip-level cross-validation.
- Investigating feature importance and hyperparameter optimization.
- Improving HMM transition modelling and sequence-level route continuity.
- Evaluating reconstruction quality on independently held-out trips.
- Exploring cloud-based execution using Modal.
- Comparing observation-level predictions with complete reconstructed routes.

## 12. Conclusion

PathSnap investigates machine learning-based map matching through a structured pipeline involving GPS preprocessing, spatial candidate generation, feature engineering, feature selection and comparative model evaluation.

The current experiments demonstrate promising candidate-classification performance from ensemble learning approaches, particularly Random Forest and LightGBM.

The project establishes a foundation for further investigation into sequence-aware map matching and geographically consistent GPS trajectory reconstruction.

---

**Project:** PathSnap  
**Team:** 11