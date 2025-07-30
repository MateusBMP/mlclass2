import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

api_columns = ['Accuracy']
api_X = np.array([[0.5612], [0.5510], [0.5816], [0.5408], [0.5816], [0.5714], [0.5765], [0.5765], [0.5714]])
api_df = pd.DataFrame(api_X, columns=api_columns)

plt.figure(figsize=(10, 5))
plt.plot(api_df['Accuracy'], marker='o', linestyle='-', label='API Accuracy')
for i, value in enumerate(api_df['Accuracy']):
    plt.text(i, value + 0.01, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

test_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
test_X = np.array([
    [0.5816, 0.4118, 0.4667, 0.4375],
    [0.6949, 0.4000, 0.4000, 0.4000],
    [0.6949, 0.3846, 0.3333, 0.3571],
    [0.7288, 0.4762, 0.6667, 0.5556],
    [0.6436, 0.4800, 0.3429, 0.4000],
    [0.6855, 0.4828, 0.5833, 0.5283],
    [0.6918, 0.4906, 0.5417, 0.5149],
    [0.6981, 0.5000, 0.6042, 0.5472],
    [0.7020, 0.5686, 0.5577, 0.5631]
])
test_df = pd.DataFrame(test_X, columns=test_columns)

plt.figure(figsize=(10, 6))
for column in test_columns:
    plt.plot(test_df[column], marker='o', linestyle='-', label=column)

    for i, value in enumerate(test_df[column]):
        plt.text(i, value + 0.01, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
