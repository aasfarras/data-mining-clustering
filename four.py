import streamlit as st
import pandas as pd
from sklearn_extra.cluster import KMedoids
import plotly.express as px
from sklearn.metrics import silhouette_score
import numpy as np

# Fungsi untuk mengganti tanggal dalam format yang salah
def replace_dates(date):
    date = date.replace('12-Aug-29', '2019-08-12 00:00:00')
    date = date.replace('12-Aug-22', '2022-08-12 00:00:00')
    date = date.replace('12-Aug-33', '2022-08-12 00:00:00')
    date = date.replace('12-Aug-34', '2022-08-12 00:00:00')
    return date

st.title('Clustering Lokasi Gempa')

# Membaca data dari file Excel
dirty_df = pd.read_excel('./data/Data_Sulteng_2018_2022.xlsx')

# Mengganti tanggal dalam format yang salah
dirty_df['Date'] = dirty_df['Date'].apply(replace_dates)

# Mengubah tahun menjadi tipe data numerik
dirty_df['Year'] = pd.to_datetime(dirty_df['Date']).dt.year

# Drop kolom yang tidak diperlukan
new_df = dirty_df.drop(['No', 'Date', 'Origin Time ', 'Remarks'], axis=1)

# Menghilangkan data duplikat
new_df = new_df.drop_duplicates()

# Mengubah format 'Mag' yang menggunakan koma menjadi titik
new_df['Mag'] = new_df['Mag'].str.replace(',', '.').astype(float)

# Mengubah format 'Lon' yang menggunakan koma menjadi titik
new_df['Lon'] = new_df['Lon'].str.replace(',', '.').astype(float)

# Sidebar untuk memilih rentang tahun
year_range = st.sidebar.slider('Tahun Kejadian Gempa:', int(new_df['Year'].min()), int(new_df['Year'].max()), (int(new_df['Year'].min()), int(new_df['Year'].max())))

# Filter data berdasarkan rentang tahun yang dipilih
new_df = new_df[(new_df['Year'] >= year_range[0]) & (new_df['Year'] <= year_range[1])]

# Menggabungkan kedalaman dan magnitudo untuk mendapatkan atribut 'power'
new_df['power'] = new_df['Depth'] + new_df['Mag']

# Mengambil hanya atribut 'power' untuk clustering
dataset = new_df[['power']].values

# Sidebar untuk memilih jumlah klaster
n_clusters = st.sidebar.slider('Jumlah Klaster:', min_value=2, max_value=10, value=4)

# Melakukan clustering dengan algoritma KMedoids
kmedoids = KMedoids(n_clusters=n_clusters, random_state=0).fit(dataset)

# Mendapatkan label klaster untuk setiap data
labels = kmedoids.labels_

# Menambahkan kolom 'cluster' ke DataFrame dengan label klaster
new_df['cluster'] = labels

# Visualisasi
fig = px.scatter_mapbox(new_df, lat='Lat', lon='Lon', color='cluster', size='power', hover_data=['Mag', 'Depth', 'power'], zoom=5)
fig.update_layout(mapbox_style='open-street-map')
st.plotly_chart(fig, use_container_width=True)

# Menampilkan data yang dipilih
st.write("Selected Data:")
st.write(new_df)

# Menghitung dan menampilkan Silhouette Score
silhouette_avg = silhouette_score(dataset, labels)
st.write(f"Silhouette Score: {silhouette_avg}")
