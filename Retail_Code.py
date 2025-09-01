import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import numpy as np

os.makedirs("output", exist_ok=True)

df = pd.read_excel("data/Online Retail.xlsx")
df = df[df['CustomerID'].notnull()]
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

df['Total'] = df['Quantity'] * df['UnitPrice']

df['Total'] = df['Quantity'] * df['UnitPrice']
ref_date = pd.to_datetime('2011-12-10')
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (ref_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Total': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print("\n📊 ID-wise Recency, Frequency, and Monetary (first 10 rows):")
print(rfm.head(10))

customer_dates = df.groupby('CustomerID')['InvoiceDate'].agg(['min', 'max']).reset_index()
customer_dates.columns = ['CustomerID', 'FirstPurchase', 'LastPurchase']

rfm = pd.merge(rfm, customer_dates, on='CustomerID')

rfm['Tenure'] = (rfm['LastPurchase'] - rfm['FirstPurchase']).dt.days

customer_country = df[['CustomerID', 'Country']].drop_duplicates(subset='CustomerID')
rfm = pd.merge(rfm, customer_country, on='CustomerID', how='left')

rfm['R_rank'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1])
rfm['F_rank'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['M_rank'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])
rfm['RFM_Score'] = rfm['R_rank'].astype(int) + rfm['F_rank'].astype(int) + rfm['M_rank'].astype(int)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

from sklearn.metrics import silhouette_score
score = silhouette_score(rfm_scaled, rfm['Cluster'])
print(f"\n📊 Silhouette Score for KMeans clustering: {score:.2f}")

def label_cluster(row):
    if row['Cluster'] == 0:
        return "Dormant / At-Risk"
    elif row['Cluster'] == 1:
        return "Active Value Seekers"
    elif row['Cluster'] == 2:
        return "Super Loyal High-Spenders"
    elif row['Cluster'] == 3:
        return "Loyal High-Spenders"

rfm['Segment'] = rfm.apply(label_cluster, axis=1)
rfm['Churn_Risk'] = pd.qcut(rfm['Recency'], 3, labels=['Low', 'Medium', 'High'])
rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
rfm['RevenuePerDay'] = rfm['Monetary'] / rfm['Recency']
rfm['CLV'] = rfm['AvgOrderValue'] * rfm['Frequency'] * rfm['Recency']

def assign_customer_type(row):
    if row['RFM_Score'] >= 10 and row['Churn_Risk'] == 'Low' and row['CLV'] > rfm['CLV'].quantile(0.75):
        return 'VIP'
    elif row['RFM_Score'] >= 9 and row['Churn_Risk'] == 'Low':
        return 'Loyal'
    elif row['RFM_Score'] <= 6 and row['Churn_Risk'] == 'High':
        return 'At-Risk'
    elif row['Frequency'] == 1 and row['Monetary'] < rfm['Monetary'].median():
        return 'One-Time Buyer'
    else:
        return 'Regular'

rfm['CustomerType'] = rfm.apply(assign_customer_type, axis=1)

segmentation_preview = rfm[['CustomerID', 'Cluster', 'Recency', 'Frequency', 'Monetary', 'Segment']].head(10)

print("\n📋 Sample Customer Segmentation Table:")
print(segmentation_preview.to_string(index=False))


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], c=rfm['Cluster'], cmap='viridis')
plt.title("Customer Clusters (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(rfm['Recency'], rfm['Monetary'], c=rfm['Cluster'], cmap='rainbow')
plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Customer Clusters")
plt.show()

cols_to_convert = [ 'CustomerID', 'Recency', 'Frequency', 'Monetary', 'CLV', 'AvgOrderValue', 'RevenuePerDay']
rfm[cols_to_convert] = rfm[cols_to_convert].replace([np.inf, -np.inf], np.nan)
rfm[cols_to_convert] = rfm[cols_to_convert].fillna(0)
rfm[cols_to_convert] = rfm[cols_to_convert].round(0).astype(int)

customer_country = df[['CustomerID', 'Country']].drop_duplicates()
rfm = pd.merge(rfm, customer_country, on='CustomerID', how='left')

rfm.to_csv("output/rfm_clusters.csv", index=False)
print("✅ rfm_clusters.csv successfully created in the output/ folder.")

summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CLV': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Num_Customers'})

print("\nCluster Averages (with customer counts):\n", summary)




