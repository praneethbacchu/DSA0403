import numpy as np
import pandas as pd
from google.colab import files
import io

print("Please upload your student performance CSV file:")
uploaded = files.upload()

# Subject ordering expected
subjects = ['Math', 'Science', 'English', 'History']

try:
    # Read the uploaded file
    filename = list(uploaded.keys())[0]
    df_students = pd.read_csv(io.BytesIO(uploaded[filename]))
    print(f"\nSuccessfully loaded {filename}!")
    
    # Attempt to extract subject columns
    # Match columns case-insensitively or fall back to the first 4 columns
    matched_cols = []
    for s in subjects:
        found = [col for col in df_students.columns if s.lower() in col.lower()]
        if found:
            matched_cols.append(found[0])
            
    if len(matched_cols) == 4:
        student_scores = df_students[matched_cols].values
    else:
        # Fallback to the first 4 numerical columns if names do not match explicitly
        num_cols = df_students.select_dtypes(include=[np.number]).columns[:4]
        student_scores = df_students[num_cols].values
        print(f"Warning: Could not match all subject names. Using the first 4 numeric columns: {list(num_cols)}")
        
except Exception as e:
    print(f"\nCould not process the uploaded file ({e}). Using fallback representative matrix instead.\n")
    # Representative 32x4 array fallback (scores between 50 and 100)
    np.random.seed(42)
    student_scores = np.random.randint(60, 100, size=(32, 4))

print(f"Student Scores Matrix shape: {student_scores.shape}\n")

# 1. Calculate the average score for each subject (mean along columns)
subject_averages = np.mean(student_scores, axis=0)

# 2. Determine the subject with the highest average
best_subject_idx = np.argmax(subject_averages)
best_subject = subjects[best_subject_idx]
best_average = subject_averages[best_subject_idx]

print("--- Subject Average Performance ---")
for i, subj in enumerate(subjects):
    print(f"{subj}: {subject_averages[i]:.2f}")

print(f"\nThe subject with the highest average score is {best_subject} with an average of {best_average:.2f}.")
