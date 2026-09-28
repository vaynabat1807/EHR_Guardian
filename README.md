# EHR Guardian – AI Data Safety Radar

EHR Guardian is a prototype system for detecting data-quality problems in Electronic Health Records (EHRs).

The system identifies potential problems in patient records and flags them for further human review without automatically modifying the original records.

## Research Scope

The prototype focuses on four types of EHR data-quality problems:

- Missing information
- Inconsistent or unusual structured values
- Duplicate patient records
- Conflicts between structured medication/allergy information and clinical notes

The clinical notes used in the prototype are synthetic English-language notes.

## Main Detection Components

### 1. Missing Information Detection
Checks required patient fields and identifies records with missing information.

### 2. Unusual Value / Anomaly Detection
Uses validation checks and Isolation Forest to identify unusual numerical observations.

### 3. Duplicate Record Detection
Compares selected patient attributes to identify possible duplicate records.

### 4. Clinical Note Conflict Detection
Uses rule-based NLP to extract medication and allergy information from clinical notes and compares it with structured EHR information.

## Dataset

The prototype uses synthetic EHR data generated with Synthea.

Synthea generated 1,136 synthetic patients, from which 1,000 patients were selected for the research.

The selected dataset contains structured information such as:

- Patient demographics
- Medications
- Allergies
- Observations

Synthetic English clinical notes were also prepared for medication and allergy conflict detection.

## Data Processing

The research dataset was divided into:

- 80% training data
- 10% validation data
- 10% test data

Controlled data-quality problems were introduced into the dataset, and the injected problems were stored as ground truth for evaluation.

## Technologies

- Python 3.13.5
- pandas
- NumPy
- scikit-learn
- Regular Expressions (`re`)
- Synthea
- Java 25.0.2 LTS

## Evaluation Metrics

The prototype is evaluated using:

- Precision
- Recall
- F1-score

The detection components are evaluated separately against the generated ground truth.

## Important Design Principle

EHR Guardian does not automatically modify or delete patient records.

Detected problems are flagged for further review so that the original EHR information remains unchanged.

## Project Structure

```text
EHR_Guardian
│
├── data
│   ├── raw
│   ├── processed
│   └── ground_truth
│
├── src
│
├── .gitignore
├── README.md
└── main.py