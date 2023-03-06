# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and clean data
sales_data = pd.read_csv('sales_data.csv')
sales_data = sales_data.drop_duplicates().dropna()

# Feature engineering
sales_data['total_purchase'] = sales_data['quantity'] * sales_data['price']
features = sales_data.groupby('customer_id').agg({'total_purchase': sum, 'order_date': 'count'}).reset_index()
features = features.rename(columns={'order_date': 'frequency', 'total_purchase': 'monetary'})

# Scaling the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features[['frequency', 'monetary']])

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_features)
features['cluster'] = kmeans.labels_

# Visualizations
plt.scatter(features['frequency'], features['monetary'], c=features['cluster'])
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.show()

# Recommendations
for i in range(3):
    segment = features[features['cluster'] == i]
    print(f"Segment {i}: {len(segment)} customers")
    print(f"Average frequency: {segment['frequency'].mean()}")
    print(f"Average monetary: {segment['monetary'].mean()}")
