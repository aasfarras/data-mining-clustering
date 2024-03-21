import streamlit as st
import pandas as pd
from sklearn_extra.cluster import KMedoids
import plotly.express as px
from sklearn.metrics import  silhouette_samples, silhouette_score, davies_bouldin_score
import numpy as np

st.title('Clustering Lokasi Gempa')

dirty_df = pd.read_excel('.\data\Data_Sulteng_2018_2022.xlsx')

new_df = dirty_df.drop(['No', 'Date', 'Origin Time ', 'Remarks'], axis=1)

new_df = new_df.drop_duplicates()

new_df.reset_index(drop=True)

new_df['Mag'] = new_df['Mag'].replace('4,0', '4.0')

new_df['Lon'] = new_df['Lon'].replace('122, 25', '122.25')

new_df['Lon'] = new_df['Lon'].astype('float')

new_df.Mag = new_df.Mag.astype('float')

dataset = new_df.values

kmedoids = KMedoids(n_clusters=4, random_state=0).fit(dataset)

labels = kmedoids.labels_
label_mapping = {0: 'sangat rendah', 1: 'rendah', 2: 'sedang', 3: 'tinggi'}
new_labels = np.array([label_mapping[label] for label in labels])
print(new_labels)

new_df['cluster'] = new_labels

new_df = new_df.sort_values('Depth', ascending=False)

new_df = new_df.reset_index(drop=True)

new_df['inverse_depth'] = ''

n_max = new_df.index.max()
length = n_max + 1

for i in range(length):
  new_df.iloc[i, new_df.columns.get_loc('inverse_depth')] = new_df.iloc[n_max]['Depth']
  n_max -= 1

new_df['inverse_depth'] = new_df['inverse_depth'].astype('float')

new_df['power'] = new_df['inverse_depth'] + new_df['Mag']

selected_clusters = st.sidebar.multiselect('Select clusters:', new_df['cluster'].unique())
filtered_df = new_df[new_df['cluster'].isin(selected_clusters)]

# px.set_mapbox_access_token(open(".mapbox_token").read())
fig = px.scatter_mapbox(new_df, lat='Lat', lon='Lon', color='cluster', size='power', hover_data=['Mag', 'Depth'], zoom=5)
fig.update_layout(mapbox_style='open-street-map')
st.plotly_chart(fig, use_container_width=True )

new_df['cluster'] = labels
kmedoid_silhouette_score = silhouette_score(new_df, labels)
print(kmedoid_silhouette_score)