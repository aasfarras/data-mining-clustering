import streamlit as st
import pandas as pd
import sklearn
from sklearn_extra.cluster import KMedoids
import plotly.express as px
from sklearn.metrics import silhouette_score
import numpy as np

st.title('Clustering Lokasi Gempa')

dirty_df = pd.read_excel('./Data_Sulteng_2018_2022.xlsx')

dirty_df['Date'] = dirty_df['Date'].replace('12-Aug-29', '2019-08-12 00:00:00')
dirty_df['Date'] = dirty_df['Date'].replace('12-Aug-22', '2022-08-12 00:00:00')
dirty_df['Date'] = dirty_df['Date'].replace('12-Aug-33', '2022-08-12 00:00:00')
dirty_df['Date'] = dirty_df['Date'].replace('12-Aug-34', '2022-08-12 00:00:00')
dirty_df['Year'] = pd.to_datetime(dirty_df['Date'], format='%Y-%m-%d %H:%M:%S').dt.year

new_df = dirty_df.drop(['No', 'Date', 'Origin Time ', 'Remarks'], axis=1)

new_df = new_df.drop_duplicates()

new_df.reset_index(drop=True)

new_df['Mag'] = new_df['Mag'].replace('4,0', '4.0')

new_df['Lon'] = new_df['Lon'].replace('122, 25', '122.25')

new_df['Lon'] = new_df['Lon'].astype('float')

new_df.Mag = new_df.Mag.astype('float')

year_range = st.sidebar.slider('Tahun Kejadian Gempa:', int(new_df['Year'].min()), int(new_df['Year'].max()), (int(new_df['Year'].min()), int(new_df['Year'].max())))

new_df = new_df[(new_df['Year'] >= year_range[0]) & (new_df['Year'] <= year_range[1])]

dataset = new_df.values

# n_medoids = st.sidebar.slider('Select number of medoids:', min_value=2, max_value=10, value=4)


kmedoids = KMedoids(n_clusters=4, random_state=0).fit(dataset)

labels = kmedoids.labels_
label_mapping = {0: 'sangat rendah', 1: 'rendah', 2: 'sedang', 3: 'tinggi'}
new_labels = np.array([label_mapping[label] for label in labels])
# print(new_labels)

new_df['cluster'] = new_labels



new_df = new_df.reset_index(drop=True)

new_df['inverse_depth'] = ''

n_max = new_df.index.max()
length = n_max + 1

for i in range(length):
  new_df.iloc[i, new_df.columns.get_loc('inverse_depth')] = new_df.iloc[n_max]['Depth']
  n_max -= 1

new_df['inverse_depth'] = new_df['inverse_depth'].astype('float')

new_df['power'] = new_df['inverse_depth'] + new_df['Mag']



selected_clusters = st.sidebar.multiselect('Kategori Cluster:', new_df['cluster'].unique())

filtered_df = new_df[new_df['cluster'].isin(selected_clusters)]

filtered_df = filtered_df.sort_values(['cluster', 'power'], ascending=True)

filtered_df['Year'] = filtered_df['Year'].astype(str)


fig = px.scatter_mapbox(filtered_df, lat='Lat', lon='Lon', color='cluster', size='power', hover_data=['Mag', 'Depth'], zoom=5)
fig.update_layout(mapbox_style='open-street-map')
st.plotly_chart(fig, use_container_width=True)

st.write("Selected Data:")
st.write(filtered_df)

kmedoid_silhouette_score = silhouette_score(new_df[['power']], labels)
st.write(f"Silhouette Score: {kmedoid_silhouette_score}")
print(filtered_df.columns)
