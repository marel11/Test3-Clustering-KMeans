# Praktikum Clustering K-Means — Gempa Bumi Global

Dashboard interaktif pengelompokan **9.998 kejadian gempa** dari berkas pengguna `global_earthquakes_seismic.csv` (10.000 baris, 25 kolom). Metode: K-Means pada enam fitur numerik: `mag`, `depth`, `nst`, `dmin`, `rms`, `gap`.

> **Penting:** Ini proyek pembelajaran clustering, bukan sistem prediksi gempa, peringatan dini tsunami, atau klasifikasi risiko. Silhouette K=2 pada data unggahan hanya ~0,205.

## Isi proyek
- `app.py`: kode utama untuk menjalankan dashboard di Streamlit Community Cloud.
- `Nama_Test3_ClusteringK-Means.py`: salinan lengkap kode program sesuai nama format tugas.
- `data_gempa_ringkas.csv`: data sumber ringkas yang dikutip dari berkas asli agar deployment ringan.
- `analisis_offline.py`: kode eksplorasi, pemodelan, evaluasi, dan ekspor file/grafik hasil.
- **`Nama_Test3_ClusteringK-Means.ipynb`**: notebook siap dijalankan di VS Code / Jupyter / Google Colab; mencakup tahapan praktikum 1–8, grafik, analisis, dan ekspor CSV. Hasil sel sudah diuji menggunakan data yang diunggah.
- `Nama_Test3_ClusteringK-Means.csv`: fitur terstandardisasi + label `cluster` untuk **K=2** (salinan juga ada di `hasil/`).
- `hasil/hasil_clustering_dengan_asli.csv`: nilai asli dan label cluster.
- `hasil/ringkasan_cluster.csv`: statistik rerata klaster.
- `hasil/metrik_clustering.json`: metrik lengkap pengujian K=2..8.
- `grafik/`: 6 visualisasi untuk laporan.

## Menjalankan notebook `.ipynb`
1. Buka `Nama_Test3_ClusteringK-Means.ipynb` di VS Code (Jupyter extension), JupyterLab, atau Google Colab.
2. Pastikan `data_gempa_ringkas.csv` ada pada folder yang sama; di Google Colab upload file tersebut ke `/content`. Jika file asli `global_earthquakes_seismic.csv` tersedia, notebook otomatis memprioritaskannya.
3. Jalankan **Run All**. Grafik dan penjelasan otomatis ditampilkan, serta file CSV hasil clustering tersimpan pada folder `hasil_notebook/`.
4. **Di VS Code**: bila `ipykernel` belum ada, jalankan `python -m pip install ipykernel` dan pilih kernel yang sama dengan interpreter project.
5. Notebook mencatat grafik hasil eksekusi awal, tetapi untuk hasil terbaru jalankan ulang semua sel. Bila `K_PILIHAN` diubah, run ulang sel model final dan seluruh sel sesudahnya.

## Persiapan pada Mac / Windows / Linux

```bash
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Jika `streamlit` tidak dikenali, jalankan: `python -m streamlit run app.py`.

**Menjalankan ulang analisis offline:**
```bash
python analisis_offline.py
```
Pipeline memprioritaskan `global_earthquakes_seismic.csv` bila Anda menyalin dataset mentah ke folder proyek; jika tidak, analisis tetap dapat berjalan dari `data_gempa_ringkas.csv`. Metadata jumlah kolom mentah pada laporan mengacu pada file lengkap 25 kolom. Aplikasi Streamlit hanya membutuhkan `data_gempa_ringkas.csv`.

## Proses ilmiah
1. Seleksi `type == earthquake`; deduplikasi ID; imputasi median bila ada missing values.
2. `log1p` untuk `depth`, `nst`, `dmin`.
3. Clipping persentil 1–99 setelah transformasi untuk `depth`, `nst`, `dmin`, `rms`, `gap`; magnitudo tidak dipotong.
4. StandardScaler pada enam fitur.
5. K-Means K=2..8, `random_state=42`, `n_init=10`, 10.000 observasi kurang dua non-gempa.
6. SSE/Inertia pada data penuh; Silhouette pada sampel 1.200 baris yang tetap.
7. Visualisasi EDA, korelasi, elbow, Silhouette, scatter 2D, proyeksi PCA, dan tabel profil klaster.

## Mengunggah ke GitHub
1. Buat repositori publik baru di GitHub, misalnya `praktikum-kmeans-gempa`.
2. Unggah `app.py`, `Nama_Test3_ClusteringK-Means.py`, `data_gempa_ringkas.csv`, `requirements.txt`, dan `README.md` (boleh seluruh folder, termasuk laporan/grafik).
3. Masuk ke [Streamlit Community Cloud](https://share.streamlit.io/), klik **Create app**, pilih repositori tersebut dan branch `main`, pilih **Main file path** `app.py`, lalu deploy.
4. Setelah berhasil, salin URL aplikasi dan URL GitHub ke laporan. **URL belum dibuat dan tidak disertakan otomatis** dalam proyek ini.

Alternatif terminal Git (jalankan dalam folder proyek):
```bash
git init
git branch -M main
git add .
git commit -m "Praktikum clustering K-Means gempa global"
git remote add origin https://github.com/USERNAME/NAMA_REPO.git
git push -u origin main
```
Ganti `USERNAME/NAMA_REPO` dengan milik Anda, pastikan repositori sudah dibuat dan autentikasi GitHub berhasil. Perintah `remote add` tidak bisa dijalankan dua kali untuk nama `origin`; gunakan `git remote set-url origin ...` untuk mengubah alamat.

## Sumber dan batasan
Dataset berasal dari **file yang diunggah untuk praktikum**. Kolom seismik tampak menggunakan struktur yang serupa dengan USGS, namun beberapa kolomnya merupakan pengayaan analitis; **asal dataset asli tidak dapat dipastikan dari file saja**. Sumber acuan definisi variabel (bukan bukti provenance file): [USGS CSV Earthquake Format](https://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php). Data peristiwa yang disertakan: 2025-07-30 s.d. 2026-10-13 (UTC).
