# Task 3 - Online Retail Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the CSV
df = pd.read_csv(r"C:\Users\LAPTOP\Desktop\Non-uni Courses\Elevvo Internship\Task 3\onlineRetail.csv")
print("First 5 rows: ")
print(df.head())
print()

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['TotalPrice'] = df['Quantity'] * df['Price']

# Cleaning the data
df = df.dropna(subset=['Customer ID'])
df = df[df['TotalPrice'] > 0]
df = df[~df['Invoice'].astype(str).str.startswith('C')]

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Calculating RFM
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5]).astype(int)

rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

# Segmentation
def segment(row):
    if row['R_score'] >= 4 and row['F_score'] >= 4:
        return 'Champions'
    elif row['R_score'] >= 3 and row['F_score'] >= 3:
        return 'Loyal Customers'
    elif row['R_score'] >= 4:
        return 'Potential Loyalists'
    elif row['R_score'] <= 2 and row['F_score'] >= 4:
        return 'At Risk'
    elif row['R_score'] <= 2:
        return 'Hibernating'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(segment, axis=1)

# Marketing suggestions
suggestions = {
    'Champions': 'Reward with VIP offers, early access, loyalty perks.',
    'Loyal Customers': 'Upsell and cross-sell with personalized bundles.',
    'Potential Loyalists': 'Encourage repeat purchases with small discounts.',
    'At Risk': 'Re-engage with win-back offers and reminders.',
    'Hibernating': 'Reactivate with strong promotions or remove from campaigns.',
    'Others': 'Monitor and gather more data.'
}

print("\nMarketing Suggestions by Segment:")
for seg in rfm['Segment'].unique():
    print(f"{seg}: {suggestions.get(seg, 'General follow-up needed.')}")

# Printing the RFM table (top 20 customers)
print("\nRFM Table:")
print(rfm.head(20))

# Bonus Visualizationss
# 1: Segment count (bar chart)
plt.figure(figsize=(10,5))
sns.countplot(data=rfm, x='Segment', order=rfm['Segment'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Customer Segments")
plt.tight_layout()
plt.show()

# 2: Heatmap of avg Monetary by R and F scores
pivot = rfm.pivot_table(index='R_score', columns='F_score', values='Monetary', aggfunc='mean')
plt.figure(figsize=(8,6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues")
plt.title("Avg Monetary Value by R & F Score")
plt.tight_layout()
plt.show()

# 3: Bar chart of Monetary by Segment
plt.figure(figsize=(10,5))
sns.barplot(data=rfm, x='Segment', y='Monetary', order=rfm['Segment'].value_counts().index, estimator=sum)
plt.xticks(rotation=45)
plt.title("Total Monetary Value by Segment")
plt.tight_layout()
plt.show()
