import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv(r'C:\Users\LAPTOP\Desktop\Non-uni Courses\Elevvo Internship\Task 4\kaggle_survey.csv', low_memory=False)
print(df.head())

df.drop_duplicates(inplace=True)
df.dropna(how='all', inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

insights = {
    'Total respondents': len(df),
    'Unique countries': df['Q3'].nunique() if 'Q3' in df.columns else 0,
    'Most common gender count': df['Q2'].value_counts().iloc[0] if 'Q2' in df.columns else 0,
    'Average age': df['Q1'].mean() if 'Q1' in df.columns else 0,
    'Python users': (df['Q6'] == 'Python').sum() if 'Q6' in df.columns else 0
}

insights_df = pd.DataFrame(list(insights.items()), columns=['Insight', 'Value'])

# Plotting
plt.barh(insights_df['Insight'], insights_df['Value'])
plt.xscale('log')
plt.title('Top 5 Insights')
plt.xlabel('Value (log scale)')
plt.tight_layout()
plt.show()