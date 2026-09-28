# EHR Guardian – AI Data Safety Radar

EHR Guardian is a prototype system for detecting data-quality problems in Electronic Health Records (EHRs).

## Main Detection Components

- Missing information detection
- Duplicate record detection
- Unusual value / anomaly detection
- Clinical note conflict detection

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Synthea
- Rule-based NLP

## Dataset

The system uses synthetic EHR data generated with Synthea.

A subset of 1,000 synthetic patients was selected for the research.

## Evaluation

The system is evaluated using:

- Precision
- Recall
- F1-score

## Important Note

The prototype does not automatically modify original patient records.

Detected issues are flagged for further review.