"""Dashboard interaktif: K-Means Pengelompokan Karakteristik Gempa Bumi Global.

Jalankan lokal:
    pip install -r requirements.txt
    streamlit run app.py

Catatan: K-Means adalah analisis eksploratif, bukan prediksi/indikator risiko gempa.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='K-Means | Gempa Global', page_icon='🌍', layout='wide')

FILE = Path(__file__).resolve().parent / 'data_gempa_ringkas.csv'
CANDIDATES = ['mag','depth','nst','dmin','rms','gap']
LABELS = {
    'mag':'Magnitudo (mag)', 'depth':'Kedalaman (km)',
    'nst':'Jumlah stasiun (nst)', 'dmin':'Jarak stasiun (dmin)',
    'rms':'RMS residual (rms)', 'gap':'Azimuthal gap (derajat)'
}
LOG_COLS = ['depth','nst','dmin']
CLIP_COLS = ['depth','nst','dmin','rms','gap']
SEED = 42

st.markdown('''<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2.4rem;}
[data-testid="stMetricValue"] {font-size: 1.7rem;}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    if not FILE.exists():
        raise FileNotFoundError(f'Dataset {FILE.name} tidak ada. Pastikan CSV disertakan di repositori.')
    df = pd.read_csv(FILE,low_memory=False)
    need = set(CANDIDATES + ['id', 'type'])
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f'Kolom dataset yang tidak ditemukan: {sorted(missing)}')
    return df

@st.cache_data(show_spinner='Melakukan preprocessing...')
def prepare(raw):
    # Data pada file induk 10.000, hanya kejadian "earthquake" yang dipakai.
    df = raw.loc[raw.type.eq('earthquake')].drop_duplicates('id').reset_index(drop=True).copy()
    missing_before = {}
    for c in CANDIDATES:
        df[c] = pd.to_numeric(df[c],errors='coerce')
        missing_before[c] = int(df[c].isna().sum())
    df[CANDIDATES] = df[CANDIDATES].fillna(df[CANDIDATES].median(numeric_only=True))
    transformed = df[CANDIDATES].copy()
    for c in LOG_COLS:
        transformed[c] = np.log1p(transformed[c].clip(lower=0))
    bounds = {}
    for c in CLIP_COLS:
        low,high=transformed[c].quantile([0.01,0.99])
        bounds[c]=(float(low),float(high))
        transformed[c]=transformed[c].clip(low,high)
    # Magnitudo TIDAK di-capping agar kejadian berkekuatan besar tetap terwakili.
    X = pd.DataFrame(StandardScaler().fit_transform(transformed),columns=CANDIDATES)
    return df,X,missing_before,bounds

@st.cache_data(show_spinner='Menghitung Elbow Method dan Silhouette Score...')
def evaluate(X):
    values=X.to_numpy()
    sample_indices=np.random.default_rng(SEED).choice(len(X),size=min(1200,len(X)),replace=False)
    results=[]
    for k in range(2,9):
        mdl=KMeans(n_clusters=k,n_init=10,random_state=SEED,max_iter=300)
        labels=mdl.fit_predict(values)
        results.append({'K':k,'Inertia':float(mdl.inertia_),
                        'Silhouette':float(silhouette_score(values[sample_indices],labels[sample_indices]))})
    return pd.DataFrame(results)

@st.cache_data(show_spinner='Menghitung clustering...')
def fit_clusters(X,k):
    m=KMeans(n_clusters=k,n_init=10,random_state=SEED,max_iter=300)
    labels=m.fit_predict(X)
    return np.asarray(labels,dtype=int),float(m.inertia_)

try:
    raw=load_data()
    df,X,missing_before,bounds=prepare(raw)
    evaluations=evaluate(X)
except Exception as err:
    st.error(f'Gagal memuat/memproses data: {err}')
    st.stop()

best_k=int(evaluations.loc[evaluations.Silhouette.idxmax(),'K'])
st.sidebar.header('⚙️ Pengaturan Clustering')
k=st.sidebar.slider('Jumlah klaster (K)',min_value=2,max_value=8,value=best_k,step=1)
opt_x=st.sidebar.selectbox('Fitur sumbu X',CANDIDATES,format_func=lambda c:LABELS[c],index=0)
opt_y=st.sidebar.selectbox('Fitur sumbu Y',CANDIDATES,format_func=lambda c:LABELS[c],index=1)
if opt_x == opt_y:
    st.sidebar.warning('Pilih dua fitur berbeda agar grafik lebih informatif.')
labels,inertia=fit_clusters(X,k)
view=df.copy();view['cluster']=labels.astype(str)
view['cluster_num']=labels
sil=float(evaluations.loc[evaluations.K.eq(k),'Silhouette'].iloc[0])
cluster_info=view.groupby('cluster_num',as_index=False)[CANDIDATES].mean()
counts=view.groupby('cluster_num').size().rename('Jumlah').reset_index()
cluster_info=counts.merge(cluster_info,on='cluster_num')
cluster_info['Persentase (%)']=cluster_info['Jumlah']/len(view)*100

st.title('🌍 Clustering Karakteristik Gempa Bumi Global')
st.caption('Praktikum Test 3 | K-Means • Dataset unggahan gempa global • Data historis 2025–2026')
st.info('Dashboard ini mengelompokkan karakteristik pengukuran kejadian gempa; **tidak** memprediksi gempa, tsunami, atau besarnya dampak. Data memiliki kolom tambahan hasil pengayaan; provenance asli dataset perlu diverifikasi.')

m1,m2,m3,m4,m5=st.columns(5)
m1.metric('Baris dataset',f'{len(raw):,}')
m2.metric('Data gempa',f'{len(view):,}')
m3.metric('Fitur clustering',f'{len(CANDIDATES)}')
m4.metric('K terpilih',str(k))
m5.metric('Silhouette',f'{sil:.3f}')

t1,t2,t3,t4,t5=st.tabs(['📊 Ringkasan & EDA','🧹 Preprocessing','📉 Optimasi K','🧩 Hasil Clustering','⬇️ Unduh & Metodologi'])

with t1:
    st.subheader('Ringkasan dan statistik deskriptif')
    st.dataframe(df[CANDIDATES].describe().T.rename(columns={'count':'Jumlah','mean':'Rerata','std':'Simpangan baku','min':'Min','max':'Maks'}).round(3),use_container_width=True)
    st.markdown(f'**Analisis dataset:** Berkas asli memiliki **{len(raw):,} baris dan {len(raw.columns)} kolom**. Dari semuanya, {len(view):,} data bertipe `earthquake` dianalisis dan {len(raw)-len(view):,} data lain/duplikat ID tidak ikut clustering. Total nilai kosong pada file asli: {int(raw.isna().sum().sum()):,}.')
    st.divider()
    chosen=st.selectbox('Pilih fitur untuk histogram dan boxplot',CANDIDATES,format_func=lambda c:LABELS[c])
    a,b=st.columns(2)
    with a:
        fig=px.histogram(view,x=chosen,nbins=40,labels={chosen:LABELS[chosen]},title=f'Distribusi {LABELS[chosen]}')
        fig.update_layout(bargap=.03)
        st.plotly_chart(fig,use_container_width=True)
    with b:
        fig=px.box(view,y=chosen,points='outliers',labels={chosen:LABELS[chosen]},title=f'Boxplot {LABELS[chosen]}')
        st.plotly_chart(fig,use_container_width=True)
    s=view[chosen]
    st.markdown(f'**Analisis histogram:** Rata-rata **{s.mean():.2f}**, median **{s.median():.2f}**, dan rentang **{s.min():.2f}–{s.max():.2f}**. '+('Rerata di atas median mengindikasikan kecenderungan distribusi miring ke kanan.' if s.mean()>s.median()*1.05 else 'Rerata dan median relatif dekat; bentuk distribusi perlu tetap dilihat dari grafik.'))
    q1,q3=s.quantile([.25,.75]);iqr=q3-q1;n_out=((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()
    st.markdown(f'**Analisis boxplot:** Berdasarkan aturan 1,5×IQR terdapat sekitar **{n_out:,}** titik di luar batas boxplot. Dalam data gempa, nilai ekstrem dapat merupakan kejadian riil; nilai asli tidak dihapus. Beberapa fitur pengamatan dibatasi pada persentil 1–99 hanya untuk pemodelan.')
    st.divider()
    st.subheader('Heatmap korelasi antarf itur'.replace('antarf itur','antarfitur'))
    corr=view[CANDIDATES].corr()
    heat=px.imshow(corr,color_continuous_scale='RdBu_r',zmin=-1,zmax=1,text_auto='.2f',aspect='auto',title='Korelasi Pearson (data asli)')
    st.plotly_chart(heat,use_container_width=True)
    tri=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool)).stack()
    strongest=tri.abs().idxmax(); val=corr.loc[strongest[0],strongest[1]]
    st.markdown(f'**Analisis korelasi:** Pasangan dengan hubungan linear absolut terbesar adalah **{LABELS[strongest[0]]}** dan **{LABELS[strongest[1]]}**, r = **{val:.3f}**. Korelasi menunjukkan keterkaitan statistik, bukan sebab akibat, dan belum menjadi bukti suatu fitur merupakan penentu kausal klaster.')

with t2:
    st.subheader('Tahapan preprocessing data')
    st.markdown('''1. **Seleksi data:** hanya `type == earthquake`, hapus duplikat berdasarkan ID peristiwa.
2. **Penanganan missing:** jika ada nilai kosong pada enam fitur, isi menggunakan median fitur (dalam data yang diunggah tidak ditemukan nilai kosong).
3. **Transformasi:** `log1p` pada `depth`, `nst`, dan `dmin` untuk meredam kemencengan distribusi.
4. **Penanganan nilai ekstrem:** winsorization persentil 1 dan 99 pada `depth`, `nst`, `dmin`, `rms`, `gap` setelah transformasi, tanpa menghapus baris; **magnitudo tidak dipotong**.
5. **Standarisasi:** `StandardScaler`, menghasilkan fitur sekitar rata-rata 0 dan standar deviasi 1.
6. **Clustering:** K-Means `random_state=42`, `n_init=10`, K=2–8.''')
    st.caption('Fitur koordinat, waktu, ID, teks, kelas magnitude, skor kepercayaan, dan label turunan lain tidak digunakan untuk menghindari bias skala atau label yang sudah direkayasa.')
    mc=st.columns(3)
    mc[0].metric('Missing pada 6 fitur',int(sum(missing_before.values())))
    mc[1].metric('Duplikat ID asli',int(raw.id.duplicated().sum()))
    mc[2].metric('Data digunakan',len(view))
    st.subheader('Contoh dataset hasil standarisasi')
    preview=X.add_suffix('_scaled');preview.insert(0,'id',df.id)
    preview['cluster']=labels
    st.dataframe(preview.head(15),use_container_width=True)
    st.markdown('**Analisis hasil preprocessing:** Nilai `_scaled` adalah satuan standar untuk perhitungan jarak Euclidean pada K-Means, bukan nilai magnitudo/kedalaman asli. Rerata klaster ditafsirkan menggunakan nilai sebelum transformasi dan standardisasi.')

with t3:
    st.subheader('Elbow Method dan Silhouette Score')
    c1,c2=st.columns(2)
    with c1:
        f=px.line(evaluations,x='K',y='Inertia',markers=True,title='Elbow Curve (SSE)')
        f.add_vline(x=best_k,line_dash='dash',line_color='#CC6B43')
        st.plotly_chart(f,use_container_width=True)
    with c2:
        f=px.bar(evaluations,x='K',y='Silhouette',title='Silhouette Score menurut K',text_auto='.3f')
        f.update_traces(marker_color=['#CC6B43' if v==best_k else '#237A91' for v in evaluations.K])
        st.plotly_chart(f,use_container_width=True)
    st.markdown('**Analisis elbow:** Inertia menurun ketika K meningkat sehingga tidak boleh dipakai sendirian untuk menyatakan jumlah K terbaik. Cari perubahan laju penurunan (titik siku) dengan mempertimbangkan kesederhanaan dan makna klaster.')
    st.markdown(f'**Analisis silhouette:** Dari K 2–8, nilai tertinggi diperoleh pada **K={best_k} (Silhouette {evaluations.Silhouette.max():.3f})**. Pilihan saat ini adalah **K={k} (Silhouette {sil:.3f})**. Nilai sekitar 0,2 berarti pemisahan klaster masih lemah/beririsan; bukan akurasi klasifikasi.')
    st.dataframe(evaluations.style.format({'Inertia':'{:,.2f}','Silhouette':'{:.4f}'}),hide_index=True,use_container_width=True)
    st.caption('Silhouette dihitung pada sampel tetap 1.200 observasi untuk performa, sedangkan K-Means dan Inertia menggunakan semua 9.998 observasi.')

with t4:
    st.subheader(f'Visualisasi klaster — K={k}')
    st.caption('Grafik menampilkan proyeksi 2 fitur dari clustering 6 dimensi; klaster dapat bertumpang tindih dalam grafik 2D.')
    fig=px.scatter(view,x=opt_x,y=opt_y,color='cluster',labels={opt_x:LABELS[opt_x],opt_y:LABELS[opt_y], 'cluster':'Klaster'},
                   hover_data=['id','place'] if 'place' in view.columns else ['id'],
                   opacity=.6,render_mode='webgl',title=f'{LABELS[opt_x]} vs {LABELS[opt_y]}')
    st.plotly_chart(fig,use_container_width=True)
    st.markdown(f'**Analisis scatter:** Titik dikelompokkan berdasarkan **{len(CANDIDATES)} fitur** yang telah distandardisasi, meskipun grafik hanya menunjukkan **{LABELS[opt_x]}** dan **{LABELS[opt_y]}**. Jika warna beririsan, hal itu konsisten dengan Silhouette **{sil:.3f}** dan perlu dibaca secara hati-hati.')
    with st.expander('Visualisasi PCA 2D (opsional)',expanded=False):
        pca=PCA(n_components=2,random_state=SEED)
        xy=pca.fit_transform(X)
        pca_df=pd.DataFrame({'PC1':xy[:,0],'PC2':xy[:,1],'cluster':view['cluster']})
        st.plotly_chart(px.scatter(pca_df,x='PC1',y='PC2',color='cluster',opacity=.5,render_mode='webgl',title='Proyeksi PCA 2 dimensi'),use_container_width=True)
        st.markdown(f'**Analisis PCA:** Dua komponen menangkap **{sum(pca.explained_variance_ratio_)*100:.1f}%** variasi fitur hasil standarisasi; pemisahan pada PCA bersifat visual, bukan hasil pemodelan ulang.')
    st.subheader('Ringkasan karakteristik setiap klaster')
    st.dataframe(cluster_info.rename(columns={'cluster_num':'Klaster'}).round(3),use_container_width=True,hide_index=True)
    chosen_mean=st.selectbox('Pilih variabel untuk profil klaster',CANDIDATES,format_func=lambda c:LABELS[c],index=0)
    fig=px.bar(cluster_info,x='cluster_num',y=chosen_mean,text_auto='.2f',title=f'Rata-rata {LABELS[chosen_mean]} untuk setiap klaster',labels={'cluster_num':'Klaster',chosen_mean:LABELS[chosen_mean]})
    fig.update_layout(xaxis_type='category')
    st.plotly_chart(fig,use_container_width=True)
    top=int(cluster_info.loc[cluster_info[chosen_mean].idxmax(),'cluster_num'])
    lo=int(cluster_info.loc[cluster_info[chosen_mean].idxmin(),'cluster_num'])
    st.markdown(f'**Analisis profil klaster:** Klaster **{top}** memiliki rerata **{LABELS[chosen_mean]}** tertinggi (**{cluster_info.loc[cluster_info.cluster_num.eq(top),chosen_mean].iloc[0]:.2f}**), sedangkan klaster **{lo}** terendah (**{cluster_info.loc[cluster_info.cluster_num.eq(lo),chosen_mean].iloc[0]:.2f}**). Gunakan tabel rerata seluruh fitur untuk menginterpretasikan kelompok secara menyeluruh.')
    st.markdown('**Interpretasi per klaster:**')
    global_mean=view[CANDIDATES].mean()
    for _,row in cluster_info.iterrows():
        elevated=[LABELS[c] for c in CANDIDATES if row[c] > global_mean[c]*1.05]
        lowered=[LABELS[c] for c in CANDIDATES if row[c] < global_mean[c]*.95]
        st.markdown(f"- **Klaster {int(row.cluster_num)}:** {int(row['Jumlah']):,} kejadian ({row['Persentase (%)']:.1f}%). Lebih tinggi dari rerata seluruh sampel: {', '.join(elevated) if elevated else 'tidak ada fitur menonjol'}; lebih rendah: {', '.join(lowered) if lowered else 'tidak ada fitur menonjol'}.")

with t5:
    st.subheader('Unduh hasil analisis')
    exported=X.add_suffix('_scaled').copy()
    exported.insert(0,'id',df['id'].values)
    exported['cluster']=labels
    st.download_button('Unduh data terstandardisasi + cluster (CSV)',data=exported.to_csv(index=False).encode('utf-8-sig'),file_name='Nama_Test3_ClusteringK-Means.csv',mime='text/csv')
    original=view[['id','time','latitude','longitude','place']+CANDIDATES+['cluster']].copy()
    st.download_button('Unduh hasil klaster + nilai asli (CSV)',data=original.to_csv(index=False).encode('utf-8-sig'),file_name='hasil_clustering_dengan_asli.csv',mime='text/csv')
    st.download_button('Unduh evaluasi K=2..8 (CSV)',data=evaluations.to_csv(index=False).encode('utf-8-sig'),file_name='evaluasi_kmeans.csv',mime='text/csv')
    st.subheader('Batasan dan sumber')
    st.markdown('''- **Sumber kerja:** `global_earthquakes_seismic.csv` yang disediakan untuk praktikum. File ringkas di repositori merupakan ekstraksi kolom dari berkas tersebut.
- **Rujukan definisi kolom:** [USGS – format CSV data gempa](https://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php). **Asal berkas yang diunggah belum dapat dipastikan** hanya dari nama dan strukturnya; beberapa kolom merupakan informasi pengayaan, bukan kolom standar USGS.
- **Makna klaster:** karakteristik pengamatan pada data historis, bukan tingkat bahaya, prediksi, atau klasifikasi formal.
- **Keterbatasan:** sampel terutama magnitudo ≥4,5, jumlah stasiun bisa dipengaruhi kondisi jaringan, dan Silhouette relatif rendah. Evaluasi ini tidak mencerminkan validitas ilmiah seismologi atau kemampuan prediksi gempa.
- **Reproduksibilitas:** `random_state=42`, `n_init=10`, standardisasi global, pencarian K=2..8, sampel silhouette 1.200 dengan seed sama.''')

st.caption('Proyek pembelajaran • Python / Pandas / Scikit-learn / Plotly / Streamlit')
