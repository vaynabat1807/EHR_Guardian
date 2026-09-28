import pandas as pd

# Load Synthea EHR data
patients = pd.read_csv("data/raw/patients.csv")
medications = pd.read_csv("data/raw/medications.csv")
allergies = pd.read_csv("data/raw/allergies.csv")
observations = pd.read_csv("data/raw/observations.csv")

print("EHR Guardian - Data Loading Test")
print("---------------------------------")

print("Patients:", len(patients))
print("Medications:", len(medications))
print("Allergies:", len(allergies))
print("Observations:", len(observations))

print("\nData loaded successfully!")

print("\nPatients columns:")
print(patients.columns.tolist())

print("\nMedications columns:")
print(medications.columns.tolist())

print("\nAllergies columns:")
print(allergies.columns.tolist())

print("\nObservations columns:")
print(observations.columns.tolist())

from sklearn.model_selection import train_test_split

# Select 1,000 patients from the Synthea dataset
patients_1000 = patients.sample(n=1000, random_state=42).copy()

# Split patients into training, validation, and test sets
train_patients, temp_patients = train_test_split(
    patients_1000,
    test_size=0.20,
    random_state=42
)

validation_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=42
)

print("\nDataset Partitioning")
print("--------------------")
print("Total patients:", len(patients_1000))
print("Training patients:", len(train_patients))
print("Validation patients:", len(validation_patients))
print("Test patients:", len(test_patients))

# Get the selected patient IDs
selected_ids = set(patients_1000["Id"])

# Keep only records belonging to the selected 1,000 patients
medications_1000 = medications[
    medications["PATIENT"].isin(selected_ids)
].copy()

allergies_1000 = allergies[
    allergies["PATIENT"].isin(selected_ids)
].copy()

observations_1000 = observations[
    observations["PATIENT"].isin(selected_ids)
].copy()

print("\nFiltered EHR Data")
print("-----------------")
print("Patients:", len(patients_1000))
print("Medications:", len(medications_1000))
print("Allergies:", len(allergies_1000))
print("Observations:", len(observations_1000))

# Save the selected and filtered data
patients_1000.to_csv(
    "data/processed/patients_1000.csv",
    index=False
)

medications_1000.to_csv(
    "data/processed/medications_1000.csv",
    index=False
)

allergies_1000.to_csv(
    "data/processed/allergies_1000.csv",
    index=False
)

observations_1000.to_csv(
    "data/processed/observations_1000.csv",
    index=False
)

print("\nProcessed data saved successfully!")


print("\nPatient data preview:")
print(patients_1000.head())

print("\nPatient columns:")
print(patients_1000.columns.tolist())



import random

# Create a copy so the original processed data remains unchanged
issue_data = patients_1000.copy()

# Reproducible random generator
random.seed(42)

# Fields selected for the missing-information test
required_fields = ["BIRTHDATE", "GENDER"]

# Store ground truth labels
ground_truth = []

# Select 50 patients for the missing-value test
selected_missing_ids = random.sample(
    list(issue_data["Id"]),
    50
)

# Introduce missing values
for patient_id in selected_missing_ids:
    field = random.choice(required_fields)

    issue_data.loc[
        issue_data["Id"] == patient_id,
        field
    ] = None

    ground_truth.append({
        "patient_id": patient_id,
        "issue_type": "missing_information",
        "field": field
    })

# Save the modified dataset
issue_data.to_csv(
    "data/processed/patients_with_issues.csv",
    index=False
)

# Save ground truth
ground_truth_df = pd.DataFrame(ground_truth)

ground_truth_df.to_csv(
    "data/ground_truth/missing_information.csv",
    index=False
)

print("\nMissing Information Injection")
print("-----------------------------")
print("Patients modified:", len(selected_missing_ids))
print("Ground truth labels:", len(ground_truth_df))
print("Missing information injection completed!")


print("\nObservation columns:")
print(observations_1000.columns.tolist())

print("\nObservation sample:")
print(observations_1000.head())


# Create a copy for unusual-value injection
observation_issues = observations_1000.copy()

# Convert VALUE to numeric where possible
observation_issues["VALUE_NUMERIC"] = pd.to_numeric(
    observation_issues["VALUE"],
    errors="coerce"
)

# Keep only observations with numeric values
numeric_observations = observation_issues[
    observation_issues["VALUE_NUMERIC"].notna()
].copy()

# Select 50 numeric observations
selected_observations = numeric_observations.sample(
    n=50,
    random_state=42
).index

# Store ground truth labels
observation_ground_truth = []

for index in selected_observations:

    observation_ground_truth.append({
        "patient_id": observation_issues.loc[index, "PATIENT"],
        "issue_type": "unusual_value",
        "observation_code": observation_issues.loc[index, "CODE"]
    })

    # Multiply the original numeric value by a large factor
    original_value = numeric_observations.loc[
        index, "VALUE_NUMERIC"
    ]

    observation_issues.loc[
        index, "VALUE_NUMERIC"
    ] = original_value * 10

# Save modified observations
observation_issues.to_csv(
    "data/processed/observations_with_issues.csv",
    index=False
)

# Save ground truth
observation_ground_truth_df = pd.DataFrame(
    observation_ground_truth
)

observation_ground_truth_df.to_csv(
    "data/ground_truth/unusual_values.csv",
    index=False
)

print("\nUnusual Value Injection")
print("-----------------------")
print(
    "Numeric observations available:",
    len(numeric_observations)
)
print(
    "Observations modified:",
    len(selected_observations)
)
print(
    "Ground truth labels:",
    len(observation_ground_truth_df)
)
print("Unusual value injection completed!")


# Create a copy of the clean patient data
duplicate_data = patients_1000.copy()

# Select 25 patients to create duplicate records
duplicate_source = patients_1000.sample(
    n=25,
    random_state=42
).copy()

# Create new IDs for the duplicated records
duplicate_source["Id"] = [
    f"DUP_{i+1:03d}"
    for i in range(len(duplicate_source))
]

# Store ground truth labels
duplicate_ground_truth = []

for original_id, duplicate_id in zip(
    patients_1000.loc[duplicate_source.index, "Id"],
    duplicate_source["Id"]
):
    duplicate_ground_truth.append({
        "original_patient_id": original_id,
        "duplicate_patient_id": duplicate_id,
        "issue_type": "duplicate_record"
    })

# Add duplicate records to the dataset
duplicate_data = pd.concat(
    [duplicate_data, duplicate_source],
    ignore_index=True
)

# Save dataset containing duplicate records
duplicate_data.to_csv(
    "data/processed/patients_with_duplicates.csv",
    index=False
)

# Save ground truth
duplicate_ground_truth_df = pd.DataFrame(
    duplicate_ground_truth
)

duplicate_ground_truth_df.to_csv(
    "data/ground_truth/duplicate_records.csv",
    index=False
)

print("\nDuplicate Record Injection")
print("--------------------------")
print("Original patients:", len(patients_1000))
print("Duplicate records added:", len(duplicate_source))
print("Final records:", len(duplicate_data))
print(
    "Ground truth labels:",
    len(duplicate_ground_truth_df)
)
print("Duplicate record injection completed!")

print("\nMedication columns:")
print(medications_1000.columns.tolist())

print("\nAllergy columns:")
print(allergies_1000.columns.tolist())

print("\nMedication sample:")
print(medications_1000.head())

print("\nAllergy sample:")
print(allergies_1000.head())

# -----------------------------------------
# Synthetic Clinical Notes Generation
# -----------------------------------------

import random

random.seed(42)

# Create one combined medication/allergy record per patient
patient_notes = []

# Get unique patient IDs
patient_ids = patients_1000["Id"].tolist()

for patient_id in patient_ids:

    # Get medication records for this patient
    patient_medications = medications_1000[
        medications_1000["PATIENT"] == patient_id
    ]

    # Get allergy records for this patient
    patient_allergies = allergies_1000[
        allergies_1000["PATIENT"] == patient_id
    ]

    # Select one medication if available
    medication = None

    if not patient_medications.empty:
        medication = patient_medications.iloc[0]["DESCRIPTION"]

    # Select one allergy if available
    allergy = None

    if not patient_allergies.empty:
        allergy = patient_allergies.iloc[0]["DESCRIPTION"]

    # Create a simple English clinical note
    if medication and allergy:
        note = (
            f"The patient is currently taking {medication}. "
            f"The patient has an allergy to {allergy}."
        )

    elif medication:
        note = (
            f"The patient is currently taking {medication}. "
            f"No known allergy is reported."
        )

    elif allergy:
        note = (
            f"The patient has an allergy to {allergy}. "
            f"No current medication is reported."
        )

    else:
        note = (
            "No current medication or known allergy is reported."
        )

    patient_notes.append({
        "patient_id": patient_id,
        "clinical_note": note
    })

# Convert notes to DataFrame
clinical_notes = pd.DataFrame(patient_notes)

# Save synthetic clinical notes
clinical_notes.to_csv(
    "data/processed/clinical_notes.csv",
    index=False
)

print("\nSynthetic Clinical Notes")
print("------------------------")
print("Number of notes:", len(clinical_notes))
print("Clinical notes generated successfully!")

# -----------------------------------------
# Clinical Note Conflict Injection
# -----------------------------------------

conflict_notes = clinical_notes.copy()
conflict_ground_truth = []

# Select 50 patients for controlled conflicts
conflict_patients = patients_1000.sample(
    n=50,
    random_state=42
)

for _, patient in conflict_patients.iterrows():

    patient_id = patient["Id"]

    # Get structured allergy information
    patient_allergies = allergies_1000[
        allergies_1000["PATIENT"] == patient_id
    ]

    # Get structured medication information
    patient_medications = medications_1000[
        medications_1000["PATIENT"] == patient_id
    ]

    # Prefer allergy conflicts when allergy information exists
    if not patient_allergies.empty:

        structured_allergy = patient_allergies.iloc[0]["DESCRIPTION"]

        conflict_text = (
            f"The patient has an allergy to "
            f"Amoxicillin."
        )

        conflict_type = "allergy_conflict"

    elif not patient_medications.empty:

        structured_medication = patient_medications.iloc[0]["DESCRIPTION"]

        conflict_text = (
            f"The patient is currently taking "
            f"Amoxicillin."
        )

        conflict_type = "medication_conflict"

    else:
        continue

    # Replace the patient's clinical note
    conflict_notes.loc[
        conflict_notes["patient_id"] == patient_id,
        "clinical_note"
    ] = conflict_text

    # Store ground truth
    conflict_ground_truth.append({
        "patient_id": patient_id,
        "issue_type": conflict_type
    })

# Save modified clinical notes
conflict_notes.to_csv(
    "data/processed/clinical_notes_with_conflicts.csv",
    index=False
)

# Save ground truth
conflict_ground_truth_df = pd.DataFrame(
    conflict_ground_truth
)

conflict_ground_truth_df.to_csv(
    "data/ground_truth/clinical_note_conflicts.csv",
    index=False
)

print("\nClinical Note Conflict Injection")
print("--------------------------------")
print(
    "Conflict records created:",
    len(conflict_ground_truth_df)
)
print("Clinical note conflicts completed!")

# -----------------------------------------
# Missing Information Detection
# -----------------------------------------

# Load the dataset containing missing values
missing_data = pd.read_csv(
    "data/processed/patients_with_issues.csv"
)

# Define required fields
required_fields = ["BIRTHDATE", "GENDER"]

missing_results = []

# Check every patient
for _, row in missing_data.iterrows():

    for field in required_fields:

        if pd.isna(row[field]) or str(row[field]).strip() == "":
            missing_results.append({
                "patient_id": row["Id"],
                "issue_type": "missing_information",
                "field": field
            })

# Convert results to DataFrame
missing_results_df = pd.DataFrame(
    missing_results
)

# Save detection results
missing_results_df.to_csv(
    "data/processed/missing_detection_results.csv",
    index=False
)

print("\nMissing Information Detection")
print("-----------------------------")
print(
    "Missing issues detected:",
    len(missing_results_df)
)
print("Missing information detection completed!")


# -----------------------------------------
# Unusual Value Detection using Isolation Forest
# -----------------------------------------

from sklearn.ensemble import IsolationForest

# Load observations containing injected unusual values
anomaly_data = pd.read_csv(
    "data/processed/observations_with_issues.csv"
)

# Convert the numeric value column
anomaly_data["VALUE_NUMERIC"] = pd.to_numeric(
    anomaly_data["VALUE_NUMERIC"],
    errors="coerce"
)

# Remove records without numeric values
numeric_data = anomaly_data[
    anomaly_data["VALUE_NUMERIC"].notna()
].copy()

# Prepare data for Isolation Forest
X = numeric_data[["VALUE_NUMERIC"]]

# Train Isolation Forest
isolation_forest = IsolationForest(
    contamination=0.05,
    random_state=42
)

numeric_data["anomaly_prediction"] = isolation_forest.fit_predict(X)

# -1 means anomaly, 1 means normal
anomalies = numeric_data[
    numeric_data["anomaly_prediction"] == -1
].copy()

# Save detected anomalies
anomalies[
    ["PATIENT", "CODE", "VALUE_NUMERIC", "anomaly_prediction"]
].to_csv(
    "data/processed/anomaly_detection_results.csv",
    index=False
)

print("\nUnusual Value Detection")
print("-----------------------")
print(
    "Numeric observations:",
    len(numeric_data)
)
print(
    "Potential anomalies detected:",
    len(anomalies)
)
print("Isolation Forest detection completed!")

# -----------------------------------------
# Duplicate Record Detection
# -----------------------------------------

# Load dataset containing duplicate records
duplicate_data = pd.read_csv(
    "data/processed/patients_with_duplicates.csv"
)

# Normalize text fields for comparison
duplicate_data["NAME_NORMALIZED"] = (
    duplicate_data["FIRST"]
    .fillna("")
    .str.lower()
    .str.strip()
)

duplicate_data["BIRTHDATE_NORMALIZED"] = (
    duplicate_data["BIRTHDATE"]
    .fillna("")
    .str.strip()
)

duplicate_data["GENDER_NORMALIZED"] = (
    duplicate_data["GENDER"]
    .fillna("")
    .str.lower()
    .str.strip()
)

# Find possible duplicate groups
duplicate_groups = duplicate_data[
    duplicate_data.duplicated(
        subset=[
            "NAME_NORMALIZED",
            "BIRTHDATE_NORMALIZED",
            "GENDER_NORMALIZED"
        ],
        keep=False
    )
].copy()

# Keep only records that have possible duplicates
duplicate_results = duplicate_groups[
    [
        "Id",
        "NAME_NORMALIZED",
        "BIRTHDATE_NORMALIZED",
        "GENDER_NORMALIZED"
    ]
].copy()

duplicate_results["issue_type"] = "duplicate_record"

# Save detection results
duplicate_results.to_csv(
    "data/processed/duplicate_detection_results.csv",
    index=False
)

print("\nDuplicate Record Detection")
print("--------------------------")
print(
    "Possible duplicate records detected:",
    len(duplicate_results)
)
print("Duplicate detection completed!")


# -----------------------------------------
# NLP Medication and Allergy Extraction
# -----------------------------------------

import re

# Load clinical notes containing controlled conflicts
notes_data = pd.read_csv(
    "data/processed/clinical_notes_with_conflicts.csv"
)

# Create medication and allergy vocabularies
medication_terms = (
    medications_1000["DESCRIPTION"]
    .dropna()
    .astype(str)
    .str.lower()
    .unique()
    .tolist()
)

allergy_terms = (
    allergies_1000["DESCRIPTION"]
    .dropna()
    .astype(str)
    .str.lower()
    .unique()
    .tolist()
)


def extract_terms(note, terms):
    """
    Extract known medication or allergy terms from a clinical note.
    """
    note = str(note).lower()

    found_terms = []

    for term in terms:
        term = term.strip()

        if not term:
            continue

        pattern = r"\b" + re.escape(term) + r"\b"

        if re.search(pattern, note):
            found_terms.append(term)

    return found_terms


# Extract information from each clinical note
extraction_results = []

for _, row in notes_data.iterrows():

    note = row["clinical_note"]

    medications_found = extract_terms(
        note,
        medication_terms
    )

    allergies_found = extract_terms(
        note,
        allergy_terms
    )

    extraction_results.append({
        "patient_id": row["patient_id"],
        "clinical_note": note,
        "medications_extracted": "; ".join(
            medications_found
        ),
        "allergies_extracted": "; ".join(
            allergies_found
        )
    })


# Convert results to DataFrame
nlp_results = pd.DataFrame(
    extraction_results
)

# Save NLP extraction results
nlp_results.to_csv(
    "data/processed/nlp_extraction_results.csv",
    index=False
)

print("\nNLP Medication and Allergy Extraction")
print("-------------------------------------")
print(
    "Clinical notes processed:",
    len(nlp_results)
)
print("NLP extraction completed!")


# -----------------------------------------
# Medication / Allergy Conflict Detection
# -----------------------------------------

# Load NLP extraction results
nlp_data = pd.read_csv(
    "data/processed/nlp_extraction_results.csv"
)

conflict_results = []

for _, row in nlp_data.iterrows():

    patient_id = row["patient_id"]

    # Get structured medication information
    patient_medications = medications_1000[
        medications_1000["PATIENT"] == patient_id
    ]

    # Get structured allergy information
    patient_allergies = allergies_1000[
        allergies_1000["PATIENT"] == patient_id
    ]

    # Structured medication terms
    structured_medications = set(
        patient_medications["DESCRIPTION"]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Structured allergy terms
    structured_allergies = set(
        patient_allergies["DESCRIPTION"]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # NLP extracted terms
    extracted_medications = set()

    if pd.notna(row["medications_extracted"]):
        extracted_medications = {
            item.strip().lower()
            for item in str(
                row["medications_extracted"]
            ).split(";")
            if item.strip()
        }

    extracted_allergies = set()

    if pd.notna(row["allergies_extracted"]):
        extracted_allergies = {
            item.strip().lower()
            for item in str(
                row["allergies_extracted"]
            ).split(";")
            if item.strip()
        }

    # Check medication conflicts
    medication_conflicts = (
        extracted_medications
        - structured_medications
    )

    # Check allergy conflicts
    allergy_conflicts = (
        extracted_allergies
        - structured_allergies
    )

    # Store conflicts
    if medication_conflicts:
        conflict_results.append({
            "patient_id": patient_id,
            "issue_type": "medication_conflict",
            "conflicting_information": "; ".join(
                medication_conflicts
            )
        })

    if allergy_conflicts:
        conflict_results.append({
            "patient_id": patient_id,
            "issue_type": "allergy_conflict",
            "conflicting_information": "; ".join(
                allergy_conflicts
            )
        })


# Convert to DataFrame
conflict_results_df = pd.DataFrame(
    conflict_results
)

# Save conflict detection results
conflict_results_df.to_csv(
    "data/processed/conflict_detection_results.csv",
    index=False
)

print("\nMedication / Allergy Conflict Detection")
print("---------------------------------------")
print(
    "Potential conflicts detected:",
    len(conflict_results_df)
)
print("Conflict detection completed!")

# -----------------------------------------
# Data Quality Score and Flagging
# -----------------------------------------

# Start with all selected patients
quality_results = patients_1000[["Id"]].copy()

quality_results["missing_issue"] = False
quality_results["anomaly_issue"] = False
quality_results["duplicate_issue"] = False
quality_results["conflict_issue"] = False

# 1. Missing information flags
if not missing_results_df.empty:
    missing_ids = set(
        missing_results_df["patient_id"]
    )

    quality_results["missing_issue"] = (
        quality_results["Id"].isin(missing_ids)
    )

# 2. Anomaly flags
if not anomalies.empty:
    anomaly_ids = set(
        anomalies["PATIENT"]
    )

    quality_results["anomaly_issue"] = (
        quality_results["Id"].isin(anomaly_ids)
    )

# 3. Duplicate flags
if not duplicate_results.empty:
    duplicate_ids = set(
        duplicate_results["Id"]
    )

    quality_results["duplicate_issue"] = (
        quality_results["Id"].isin(duplicate_ids)
    )

# 4. Conflict flags
if not conflict_results_df.empty:
    conflict_ids = set(
        conflict_results_df["patient_id"]
    )

    quality_results["conflict_issue"] = (
        quality_results["Id"].isin(conflict_ids)
    )

# Count detected issues
quality_results["issue_count"] = (
    quality_results[
        [
            "missing_issue",
            "anomaly_issue",
            "duplicate_issue",
            "conflict_issue"
        ]
    ].sum(axis=1)
)

# Calculate data-quality score
quality_results["data_quality_score"] = (
    100 - (quality_results["issue_count"] * 25)
)

# Prevent negative scores
quality_results["data_quality_score"] = (
    quality_results["data_quality_score"].clip(lower=0)
)

# Create review flag
quality_results["requires_review"] = (
    quality_results["issue_count"] > 0
)

# Save final quality results
quality_results.to_csv(
    "data/processed/data_quality_results.csv",
    index=False
)

print("\nData Quality Scoring")
print("--------------------")
print(
    "Total patients evaluated:",
    len(quality_results)
)
print(
    "Patients requiring review:",
    quality_results[
        "requires_review"
    ].sum()
)
print("Data-quality scoring completed!")

# -----------------------------------------
# Evaluation
# -----------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_detection(
    ground_truth_ids,
    detected_ids,
    name
):
    """
    Compare ground-truth patient IDs with
    detected patient IDs.
    """

    all_ids = set(ground_truth_ids) | set(detected_ids)

    y_true = [
        1 if patient_id in ground_truth_ids else 0
        for patient_id in all_ids
    ]

    y_pred = [
        1 if patient_id in detected_ids else 0
        for patient_id in all_ids
    ]

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print(f"\n{name}")
    print("-----------------------------")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    return {
        "component": name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


evaluation_results = []


# 1. Missing information evaluation
missing_ground_truth_ids = set(
    pd.read_csv(
        "data/ground_truth/missing_information.csv"
    )["patient_id"]
)

missing_detected_ids = set(
    missing_results_df["patient_id"]
)

evaluation_results.append(
    evaluate_detection(
        missing_ground_truth_ids,
        missing_detected_ids,
        "Missing Information Detection"
    )
)


# 2. Duplicate detection evaluation
duplicate_ground_truth_data = pd.read_csv(
    "data/ground_truth/duplicate_records.csv"
)

duplicate_ground_truth_ids = set(
    duplicate_ground_truth_data[
        "duplicate_patient_id"
    ]
)

duplicate_detected_ids = set(
    duplicate_results["Id"]
)

evaluation_results.append(
    evaluate_detection(
        duplicate_ground_truth_ids,
        duplicate_detected_ids,
        "Duplicate Record Detection"
    )
)
# Evaluate unusual value / anomaly detection

unusual_ground_truth = pd.read_csv(
    "data/ground_truth/unusual_values.csv"
)

anomaly_results = pd.read_csv(
    "data/processed/anomaly_detection_results.csv"
)

unusual_ground_truth_pairs = set(
    zip(
        unusual_ground_truth["patient_id"],
        unusual_ground_truth["observation_code"]
    )
)

anomaly_detected_pairs = set(
    zip(
        anomaly_results["PATIENT"],
        anomaly_results["CODE"]
    )
)

evaluation_results.append(
    evaluate_detection(
        unusual_ground_truth_pairs,
        anomaly_detected_pairs,
        "Unusual Value / Anomaly Detection"
    )
)


# Evaluate clinical note conflict detection

conflict_ground_truth = pd.read_csv(
    "data/ground_truth/clinical_note_conflicts.csv"
)

conflict_results = pd.read_csv(
    "data/processed/conflict_detection_results.csv"
)

conflict_ground_truth_pairs = set(
    zip(
        conflict_ground_truth["patient_id"],
        conflict_ground_truth["issue_type"]
    )
)

conflict_detected_pairs = set(
    zip(
        conflict_results["patient_id"],
        conflict_results["issue_type"]
    )
)

evaluation_results.append(
    evaluate_detection(
        conflict_ground_truth_pairs,
        conflict_detected_pairs,
        "Clinical Note Conflict Detection"
    )
)


# Save final evaluation results

evaluation_df = pd.DataFrame(
    evaluation_results
)

evaluation_df.to_csv(
    "data/processed/evaluation_results.csv",
    index=False
)

print("\nEvaluation completed!")
print(evaluation_df)