import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib_venn import venn3

# Load the data
data = pd.read_csv('labeled_data_output.csv')

# 1. Venn Diagram
plt.figure(figsize=(10, 10))
venn3(subsets=(
    np.sum(data['Economic'] & ~data['Environmental'] & ~data['Social']),
    np.sum(~data['Economic'] & data['Environmental'] & ~data['Social']),
    np.sum(data['Economic'] & data['Environmental'] & ~data['Social']),
    np.sum(~data['Economic'] & ~data['Environmental'] & data['Social']),
    np.sum(data['Economic'] & ~data['Environmental'] & data['Social']),
    np.sum(~data['Economic'] & data['Environmental'] & data['Social']),
    np.sum(data['Economic'] & data['Environmental'] & data['Social'])
), set_labels=('Economic', 'Environmental', 'Social'))
plt.title('Distribution of Indicators Across Categories')
plt.savefig('venn_diagram.png')
plt.close()

# 2. Stacked Bar Chart
plt.figure(figsize=(12, 6))
data[['Probability_Economic', 'Probability_Environmental', 'Probability_Social']].mean().plot(kind='bar', stacked=True)
plt.title('Average Probability Distribution Across Categories')
plt.xlabel('Categories')
plt.ylabel('Average Probability')
plt.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('stacked_bar_chart.png')
plt.close()

# 3. Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data[['Probability_Economic', 'Probability_Environmental', 'Probability_Social']].corr(), 
            cmap='YlOrRd', annot=True, fmt='.2f')
plt.title('Correlation Heatmap of Probabilities')
plt.savefig('heatmap.png')
plt.close()

# 4. Scatter Plot Matrix
sns.pairplot(data[['Probability_Economic', 'Probability_Environmental', 'Probability_Social']])
plt.suptitle('Scatter Plot Matrix of Probabilities', y=1.02)
plt.savefig('scatter_plot_matrix.png')
plt.close()

# 5. Box Plot
plt.figure(figsize=(10, 6))
data[['Probability_Economic', 'Probability_Environmental', 'Probability_Social']].boxplot()
plt.title('Distribution of Probabilities')
plt.ylabel('Probability')
plt.savefig('box_plot.png')
plt.close()

# 6. Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=data[['Probability_Economic', 'Probability_Environmental', 'Probability_Social']])
plt.title('Violin Plot of Probability Distributions')
plt.ylabel('Probability')
plt.savefig('violin_plot.png')
plt.close()

# Additional statistical analysis
print("Statistical Summary:")
print(data[['Probability_Economic', 'Probability_Environmental', 'Probability_Social']].describe())

print("\nCorrelation Matrix:")
print(data[['Probability_Economic', 'Probability_Environmental', 'Probability_Social']].corr())

print("\nCross-tabulation of predicted labels:")
print(pd.crosstab(data['Predicted_Labels'], columns='count'))

