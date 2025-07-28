# prompt: plot confusion matrix for 1.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the data
try:
  df = pd.read_csv('1.csv')
except FileNotFoundError:
  print("Error: '1.csv' not found. Please make sure the file exists in the current directory.")
  exit()

# Assuming your CSV has columns named 'actual' and 'predicted'
# Replace 'actual' and 'predicted' with the actual column names in your CSV if different.

# Check if 'actual' and 'predicted' columns exist in the DataFrame
if 'Price (Euros)' not in df.columns or 'CPU' not in df.columns:
    print("Error: 'actual' or 'predicted' columns not found in the CSV file. Please check the column names.")
    exit()

# Assign y_true and y_pred if columns exist
y_true = df['CPU']
y_pred = df['Price (Euros)']

# Create the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(df['Price (Euros)'].unique()), 
            yticklabels=sorted(df['CPU'].unique()))
plt.xlabel('Price (Euros)')
plt.ylabel('CPU')
plt.title('Confusion Matrix')
plt.show()