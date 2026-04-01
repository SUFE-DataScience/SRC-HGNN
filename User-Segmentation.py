import pandas as pd

data = pd.read_excel('data.xlsx')
data.columns = data.columns.str.strip()

data['Label'] = pd.to_numeric(data['Label'], errors='coerce').fillna(0)
data['Label'] = data['Label'].clip(0, 1)

fraud_rate_per_row = data.groupby('Reviewer_id')['Label'].transform('mean')

data['Label_user'] = (fraud_rate_per_row >= 0.5).astype(int)

data.to_excel('output_user_labels_full.xlsx', index=False)

print("Generated file with original rows and user labels based on 50% threshold: 'output_user_labels_full.xlsx'")
