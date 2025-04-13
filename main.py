import streamlit as st
import pandas as pd
import os
from langchain_community.llms import Ollama
import nltk
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')


def preprocess_text(text, remove_stopwords=True):
    # Hapus links
    text = re.sub(r"http\S+", "", str(text))
    # Hapus nomor dan karakter khusus
    text = re.sub("[^A-Za-z]+", " ", str(text))
    # Hapus stopwords
    if remove_stopwords:
        tokens = nltk.word_tokenize(text)
        stopwords_ind = set(nltk.corpus.stopwords.words("indonesian"))
        # Tambahan stopwords
        stopwords_ind.update(['com', 'rp', 'm', 'ol', 'triwulan', 'i', 'dan', 'atau', 'perusahaan'])
        tokens = [w for w in tokens if not w.lower() in stopwords_ind]
        text = " ".join(tokens)
    text = text.lower().strip()
    return text


def generate_wordcloud(text, title):
    wordcloud = WordCloud(
        stopwords=nltk.corpus.stopwords.words("indonesian"),
        background_color='white',
        width=1000,
        height=1000
    ).generate(text)

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)
    plt.close()  # Tutup figure untuk menghindari penumpukan di memori


def calculate_tfidf(corpus):
    vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words("indonesian"))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names


def display_top_tfidf_words(tfidf_matrix, feature_names, top_n=10):
    sums = tfidf_matrix.sum(axis=0)
    data = []
    for col, term in enumerate(feature_names):
        data.append((term, sums[0, col]))
    ranking = sorted(data, key=lambda x: x[1], reverse=True)
    top_words = ranking[:top_n]
    st.write(f"Top {top_n} words by TF-IDF:")
    for word, score in top_words:
        st.write(f"{word}: {score:.4f}")


def process_excel_file(uploaded_file, company_df=None, is_second_excel=False):
    combined_df = pd.DataFrame(columns=['Nama Contact', 'Pertanyaan', 'Nilai'])

    sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

    for sheet_name, df in sheets.items():
        st.subheader(f"Data pada Sheet: {sheet_name}")
        with st.expander("Lihat Dataframe Utuh Sebelum Penyaringan"):
            st.dataframe(df)

        try:
            filtered_df = df[['Unnamed: 2', 'Unnamed: 3']]  # Ambil kolom Unnamed: 2 dan Unnamed: 3
            filtered_df.columns = ['Pertanyaan', 'Nilai']

            relevant_columns = [
                'Permintaan Domestik  - Likert Scale',
                'Permintaan Ekspor  - Likert Scale',
                'Kapasitas Utilisasi - Likert Scale',
                'Persediaan - Likert Scale',
                'Investasi - Likert Scale',
                'Biaya Energi - Likert Scale',
                'Biaya Tenaga Kerja (Upah) - Likert Scale',
                'Harga Jual – Likert Scale',
                'Margin Usaha - Likert Scale',
                'Tenaga Kerja - Likert Scale',
                'Perkiraan Penjualan – Likert Scale',
                'Perkiraan Tingkat Upah – Likert Scale',
                'Perkiraan Harga Jual – Likert Scale',
                'Perkiraan Jumlah Tenaga Kerja – Likert Scale',
                'Perkiraan Investasi – Likert Scale'
            ]

            # Ganti en-dash “–” menjadi dash “-” agar seragam
            df['Unnamed: 2'] = df['Unnamed: 2'].astype(str).str.replace('–', '-')

            filtered_df = filtered_df[filtered_df['Pertanyaan'].isin(relevant_columns)]
            # Ubah '–' ke '-' di kolom Pertanyaan
            filtered_df['Pertanyaan'] = filtered_df['Pertanyaan'].str.replace('–', '-')

            nama_contact = df[df['Unnamed: 2'] == 'Nama Contact']['Unnamed: 3'].values[0]
            filtered_df.insert(0, 'Nama Contact', nama_contact)

            if company_df is not None:
                filtered_df = filtered_df.merge(company_df, left_on='Nama Contact', right_on='Nama Perusahaan', how='left')

            combined_df = pd.concat([combined_df, filtered_df])
            combined_df = combined_df.fillna('kosong')
        except KeyError:
            logging.error("Kolom yang dibutuhkan tidak ditemukan dalam file Excel.")
            continue

    avg_values = {}
    if not combined_df.empty:
        st.write("**Dataframe Setelah Penyaringan dari Semua Sheet:**")
        st.dataframe(combined_df)

        with st.expander("Rata-rata Nilai untuk Setiap Kolom Likert Scale"):
            for col in combined_df['Pertanyaan'].unique():
                numeric_values = pd.to_numeric(combined_df[combined_df['Pertanyaan'] == col]['Nilai'], errors='coerce')
                avg_value = numeric_values.mean()
                avg_values[col] = round(avg_value, 2) if not pd.isna(avg_value) else None

            for col, avg_value in avg_values.items():
                col_name = col.replace(' - Likert Scale', ' ')
                st.write(f"{col_name}: {avg_value}")

        if is_second_excel and company_df is not None:
            lu_totals = combined_df[['Nama Contact', 'Lapangan Usaha']].drop_duplicates()['Lapangan Usaha'].value_counts()

            with st.expander("Total LU untuk Setiap Sektor / Jenis Usaha (LU)"):
                for lu, total in lu_totals.items():
                    st.write(f"{lu}: {total}")

            total_lu = lu_totals.sum()
            if total_lu > 0:
                lu_dominance = (lu_totals / total_lu) * 100
                max_dominant_lu = lu_dominance.idxmax()
                st.write("**LU yang Mendominasi:**")
                st.info(f"{max_dominant_lu}: {lu_dominance[max_dominant_lu]:.2f}%")
            else:
                st.write("Tidak ada data LU yang dapat dihitung.")
    else:
        st.write("Dataframe setelah penyaringan kosong.")
    return combined_df, avg_values


def process_domestik_ekspor_df(uploaded_file, company_df):
    combined_df_domestik_ekspor = pd.DataFrame(columns=['Nama Contact', 'Pertanyaan', 'Nilai', 'Lapangan Usaha'])
    sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

    for sheet_name, df in sheets.items():
        try:
            # Ganti en-dash “–” menjadi dash “-” agar seragam
            df['Unnamed: 2'] = df['Unnamed: 2'].astype(str).str.replace('–', '-')

            relevant_columns = ['Permintaan Domestik  - Likert Scale', 'Permintaan Ekspor  - Likert Scale']
            df_filtered = df[['Unnamed: 2', 'Unnamed: 3']]
            df_filtered.columns = ['Pertanyaan', 'Nilai']
            df_filtered = df_filtered[df_filtered['Pertanyaan'].isin(relevant_columns)]

            nama_contact = df[df['Unnamed: 2'] == 'Nama Contact']['Unnamed: 3'].values[0]
            df_filtered.insert(0, 'Nama Contact', nama_contact)

            df_filtered = df_filtered.merge(company_df, left_on='Nama Contact', right_on='Nama Perusahaan', how='left')

            combined_df_domestik_ekspor = pd.concat([combined_df_domestik_ekspor, df_filtered])
            combined_df_domestik_ekspor = combined_df_domestik_ekspor.fillna('kosong')
        except KeyError:
            logging.error("Kolom yang dibutuhkan tidak ditemukan dalam file Excel.")
            continue

    domestic_count = 0
    export_count = 0
    domestic_export_count = 0

    lu_domestic = []
    lu_export = []
    lu_domestic_export = []

    for contact in combined_df_domestik_ekspor['Nama Contact'].unique():
        df_contact = combined_df_domestik_ekspor[combined_df_domestik_ekspor['Nama Contact'] == contact]
        domestic_value = df_contact[df_contact['Pertanyaan'] == 'Permintaan Domestik  - Likert Scale']['Nilai'].values
        export_value = df_contact[df_contact['Pertanyaan'] == 'Permintaan Ekspor  - Likert Scale']['Nilai'].values

        lu_contact = df_contact['Lapangan Usaha'].values[0]

        if len(domestic_value) > 0 and len(export_value) > 0:
            if domestic_value[0] != 'kosong' and export_value[0] == 'kosong':
                domestic_count += 1
                lu_domestic.append(lu_contact)
            elif domestic_value[0] == 'kosong' and export_value[0] != 'kosong':
                export_count += 1
                lu_export.append(lu_contact)
            elif domestic_value[0] != 'kosong' and export_value[0] != 'kosong':
                domestic_export_count += 1
                lu_domestic_export.append(lu_contact)

    total_count = domestic_count + export_count + domestic_export_count

    return (combined_df_domestik_ekspor,
            total_count, domestic_count, export_count, domestic_export_count,
            lu_domestic, lu_export, lu_domestic_export)


def process_alasan_domestik_ekspor_df(uploaded_file):
    alasan_domestik_ekspor_df = pd.DataFrame(columns=['Nama Contact', 'Permintaan Domestik', 'Permintaan Ekspor'])
    sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

    for sheet_name, df in sheets.items():
        try:
            # Ganti en-dash “–” menjadi dash “-” agar seragam
            df['Unnamed: 2'] = df['Unnamed: 2'].astype(str).str.replace('–', '-')

            permintaan_domestik = df[df['Unnamed: 2'] == 'Permintaan/Penjualan  - Permintaan Domestik']['Unnamed: 3'].values[0]
            permintaan_ekspor = df[df['Unnamed: 2'] == 'Permintaan/Penjualan - Permintaan Ekspor']['Unnamed: 3'].values[0]

            alasan_domestik_ekspor_df = pd.concat([
                alasan_domestik_ekspor_df,
                pd.DataFrame({
                    'Nama Contact': [sheet_name],
                    'Permintaan Domestik': [permintaan_domestik],
                    'Permintaan Ekspor': [permintaan_ekspor]
                })
            ], ignore_index=True)

        except (KeyError, IndexError) as e:
            logging.error(f"Data Permintaan Domestik atau Ekspor tidak ditemukan dalam sheet {sheet_name}: {e}")

    return alasan_domestik_ekspor_df


def process_additional_data_df(uploaded_file):
    additional_data_df = pd.DataFrame(columns=[
        'Nama Contact', 'Kapasitas Utilisasi', 'Persediaan', 'Investasi',
        'Biaya-biaya - Bahan Baku Diluar Gaji/Upah', 'Biaya Energi',
        'Biaya Tenaga Kerja (Upah)', 'Harga Jual - Perkembangan Harga Jual',
        'Margin Usaha', 'Perkembangan Jumlah Tenaga Kerja',
        'Perkiraan Penjualan', 'Perkiraan Tingkat Upah', 'Perkiraan Harga Jual',
        'Perkiraan Jumlah Tenaga Kerja', 'Perkiraan Investasi',
        'Pembiayaan dan Suku Bunga'
    ])

    sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

    for sheet_name, df in sheets.items():
        try:
            # Ganti en-dash “–” menjadi dash “-” agar seragam
            df['Unnamed: 2'] = df['Unnamed: 2'].astype(str).str.replace('–', '-')

            data = {
                'Nama Contact': sheet_name,
                'Kapasitas Utilisasi': df[df['Unnamed: 2'] == 'Kapasitas Utilisasi']['Unnamed: 3'].values[0],
                'Persediaan': df[df['Unnamed: 2'] == 'Persediaan']['Unnamed: 3'].values[0],
                'Investasi': df[df['Unnamed: 2'] == 'Investasi']['Unnamed: 3'].values[0],
                'Biaya-biaya - Bahan Baku Diluar Gaji/Upah': df[df['Unnamed: 2'] == 'Biaya-biaya - Bahan Baku Diluar Gaji/Upah']['Unnamed: 3'].values[0],
                'Biaya Energi': df[df['Unnamed: 2'] == 'Biaya Energi']['Unnamed: 3'].values[0],
                'Biaya Tenaga Kerja (Upah)': df[df['Unnamed: 2'] == 'Biaya Tenaga Kerja (Upah)']['Unnamed: 3'].values[0],
                'Harga Jual - Perkembangan Harga Jual': df[df['Unnamed: 2'] == 'Harga Jual - Perkembangan Harga Jual']['Unnamed: 3'].values[0],
                'Margin Usaha': df[df['Unnamed: 2'] == 'Margin Usaha']['Unnamed: 3'].values[0],
                'Perkembangan Jumlah Tenaga Kerja': df[df['Unnamed: 2'] == 'Perkembangan Jumlah Tenaga Kerja']['Unnamed: 3'].values[0],
                'Perkiraan Penjualan': df[df['Unnamed: 2'] == 'Perkiraan Penjualan']['Unnamed: 3'].values[0],
                'Perkiraan Tingkat Upah': df[df['Unnamed: 2'] == 'Perkiraan Tingkat Upah']['Unnamed: 3'].values[0],
                'Perkiraan Harga Jual': df[df['Unnamed: 2'] == 'Perkiraan Harga Jual']['Unnamed: 3'].values[0],
                'Perkiraan Jumlah Tenaga Kerja': df[df['Unnamed: 2'] == 'Perkiraan Jumlah Tenaga Kerja']['Unnamed: 3'].values[0],
                'Perkiraan Investasi': df[df['Unnamed: 2'] == 'Perkiraan Investasi']['Unnamed: 3'].values[0],
                'Pembiayaan dan Suku Bunga': df[df['Unnamed: 2'] == 'Pembiayaan dan Suku Bunga']['Unnamed: 3'].values[0],
            }

            additional_data_df = pd.concat([additional_data_df, pd.DataFrame(data, index=[0])], ignore_index=True)
        except (KeyError, IndexError) as e:
            logging.error(f"Data yang dibutuhkan tidak ditemukan dalam sheet {sheet_name}: {e}")

    return additional_data_df


# Mapping nama kolom "indikator lain" -> key yang sesuai di avg_values
# Agar kita bisa mengambil avg_values_1 dan avg_values_2 untuk indikator-indikator tersebut.
INDICATOR_MAP = {
    "Kapasitas Utilisasi": "Kapasitas Utilisasi - Likert Scale",
    "Persediaan": "Persediaan - Likert Scale",
    "Investasi": "Investasi - Likert Scale",
    "Biaya Energi": "Biaya Energi - Likert Scale",
    "Biaya Tenaga Kerja (Upah)": "Biaya Tenaga Kerja (Upah) - Likert Scale",
    "Harga Jual - Perkembangan Harga Jual": "Harga Jual – Likert Scale",  # Perhatikan en-dash
    "Margin Usaha": "Margin Usaha - Likert Scale",
    "Tenaga Kerja": "Tenaga Kerja - Likert Scale",
    "Perkiraan Penjualan": "Perkiraan Penjualan – Likert Scale",  # Perhatikan en-dash
    "Perkiraan Tingkat Upah": "Perkiraan Tingkat Upah – Likert Scale",
    "Perkiraan Harga Jual": "Perkiraan Harga Jual – Likert Scale",
    "Perkiraan Jumlah Tenaga Kerja": "Perkiraan Jumlah Tenaga Kerja – Likert Scale",
    "Perkiraan Investasi": "Perkiraan Investasi – Likert Scale",
    "Pembiayaan dan Suku Bunga": None  # Misal tidak ada Likert Scale
}


def main():
    st.set_page_config(page_title='ML Summary Liaison', layout="wide")
    st.markdown(
        """
        <style>
        /* Warna background dan teks */
        body {
            background-color: #fafafa;
            color: #333;
        }

        /* Judul utama */
        .css-18e3th9 {
            padding: 2rem;
            background-color: #2b6777;
            border-radius: 0.5rem;
        }
        .css-18e3th9 h1 {
            color: #ffffff !important;
        }

        /* Heading level di dalam tab */
        .stTabs [role="tab"] {
            font-size: 16px;
        }

        /* Kotak sidebar */
        .css-1d391kg {
            background-color: #f0f0f0 !important;
        }

        /* Komponen data frame */
        .stDataFrame {
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('SONIC (liaiSON quIck Capture) \U0001F4CA')

    with st.sidebar:
        st.header("Unggah File Excel")
        st.write("1. File Excel Pertama (triwulan sebelumnya)")
        uploaded_file_1 = st.file_uploader('File 1 (XLSX)', type='xlsx')

        st.write("---")
        st.write("2. File Excel Kedua (triwulan sekarang)")
        uploaded_file_2 = st.file_uploader('File 2 (XLSX)', type='xlsx')

        st.write("---")
        st.write("3. File Excel Database (Nama Perusahaan & Sektor)")
        company_file = st.file_uploader('Database (XLSX)', type='xlsx')

    tabs = st.tabs([
        "Overview",
        "Data Triwulan Sebelumnya",
        "Data Triwulan Sekarang",
        "Perbandingan & Kesimpulan",
        "Analisis Lanjutan (Wordcloud & TF-IDF)",
        "Analisis Kesimpulan LLM"
    ])

    # Variabel global (penampung)
    combined_df_1 = pd.DataFrame()
    avg_values_1 = {}
    combined_df_2 = pd.DataFrame()
    avg_values_2 = {}
    combined_df_domestik_ekspor = pd.DataFrame()
    alasan_domestik_ekspor_df = pd.DataFrame()
    jawaban_domestik = ""
    jawaban_ekspor = ""
    additional_data_df = pd.DataFrame()
    jawaban_additional_data = {}
    total_count = 0
    domestic_count = 0
    export_count = 0
    domestic_export_count = 0
    lu_domestic = set()
    lu_export = set()
    lu_domestic_export = set()

    # TAB Overview
    with tabs[0]:
        st.subheader("Panduan Singkat Penggunaan Aplikasi")
        st.markdown("""
        1. Unggah tiga buah file di **Sidebar**:
           - **File Excel Pertama**: Data triwulan sebelumnya
           - **File Excel Kedua**: Data triwulan yang ingin dibandingkan
           - **File Database**: Berisi Nama Perusahaan dan Sektor
        2. Buka tab **Data Triwulan Sebelumnya** dan **Data Triwulan Sekarang** untuk melihat hasil pemrosesan.
        3. Selanjutnya, buka tab **Perbandingan & Kesimpulan** untuk melihat hasil komparasi kedua data.
        4. Untuk analisis lanjutan (Wordcloud, TF-IDF), buka tab **Analisis Lanjutan**.
        5. Untuk melakukan kesimpulan/model LLM, buka tab **Analisis Kesimpulan LLM**.
        """)

    # TAB Data Triwulan Sebelumnya
    with tabs[1]:
        st.subheader("Data Triwulan Sebelumnya")
        if uploaded_file_1:
            st.success("File Excel Pertama Terunggah!")
            combined_df_1, avg_values_1 = process_excel_file(uploaded_file_1)
        else:
            st.warning("Belum ada file yang diunggah untuk triwulan sebelumnya.")

    # TAB Data Triwulan Sekarang
    with tabs[2]:
        st.subheader("Data Triwulan Sekarang")
        if uploaded_file_2 and company_file:
            company_df = pd.read_excel(company_file)

            st.success("File Excel Kedua & Database Terunggah!")
            combined_df_2, avg_values_2 = process_excel_file(uploaded_file_2, company_df, is_second_excel=True)

            st.markdown("---")
            st.subheader("Dataframe Permintaan Domestik dan Ekspor")
            (combined_df_domestik_ekspor,
             total_count, domestic_count, export_count, domestic_export_count,
             lu_domestic, lu_export, lu_domestic_export) = process_domestik_ekspor_df(uploaded_file_2, company_df)
            st.dataframe(combined_df_domestik_ekspor)

            with st.expander("Jumlah & Persentase Orientasi"):
                st.write(f"Jumlah orientasi Domestik: {domestic_count}")
                st.write(f"Jumlah orientasi Ekspor: {export_count}")
                st.write(f"Jumlah orientasi Domestik dan Ekspor: {domestic_export_count}")

                if total_count > 0:
                    domestic_percentage = (domestic_count / total_count) * 100
                    export_percentage = (export_count / total_count) * 100
                    domestic_export_percentage = (domestic_export_count / total_count) * 100
                else:
                    domestic_percentage = 0
                    export_percentage = 0
                    domestic_export_percentage = 0

                st.write(f"Persen orientasi Domestik: {domestic_percentage:.2f}%")
                st.write(f"Persen orientasi Ekspor: {export_percentage:.2f}%")
                st.write(f"Persen orientasi Domestik dan Ekspor: {domestic_export_percentage:.2f}%")

                st.write(f"LU yang berorientasi Domestik: {', '.join(lu_domestic)}")
                st.write(f"LU yang berorientasi Ekspor: {', '.join(lu_export)}")
                st.write(f"LU yang berorientasi Domestik dan Ekspor: {', '.join(lu_domestic_export)}")

            st.markdown("---")
            st.subheader("Dataframe Alasan Permintaan Domestik & Ekspor")
            alasan_domestik_ekspor_df = process_alasan_domestik_ekspor_df(uploaded_file_2)
            st.dataframe(alasan_domestik_ekspor_df)

            # Kumpulkan jawaban domestik & ekspor
            jawaban_domestik = " ".join([
                f"Pada triwulan laporan, kontak menyatakan bahwa {j}."
                for j in alasan_domestik_ekspor_df['Permintaan Domestik']
            ])
            jawaban_ekspor = " ".join([
                f"Pada triwulan laporan, kontak menyatakan bahwa {j}."
                for j in alasan_domestik_ekspor_df['Permintaan Ekspor']
            ])

            st.markdown("---")
            st.subheader("Dataframe Indikator Lain (Utilisasi dkk.)")
            additional_data_df = process_additional_data_df(uploaded_file_2)
            st.dataframe(additional_data_df)

            # Kumpulkan jawaban "indikator lain" (sebelumnya "data tambahan") dalam dictionary
            for column in additional_data_df.columns[1:]:
                jawaban_column = " ".join([
                    f"Pada triwulan laporan, kontak menyatakan bahwa {ans}."
                    for ans in additional_data_df[column]
                ])
                jawaban_additional_data[column] = jawaban_column

            # Gabungkan semua penjabaran dalam satu expander
            with st.expander("Penjabaran Jawaban dari setiap Indikator"):
                st.subheader("Penjabaran Permintaan Domestik")
                st.write(jawaban_domestik)
                st.subheader("Penjabaran Permintaan Ekspor")
                st.write(jawaban_ekspor)

                st.markdown("---")
                st.subheader("Penjabaran Indikator Lain (Utilisasi, Persediaan, dkk.)")
                for column, jawaban_text in jawaban_additional_data.items():
                    st.markdown(f"**{column}**")
                    st.write(jawaban_text)

        else:
            st.warning("Silakan unggah File Excel Kedua dan Database terlebih dahulu.")

    # TAB Perbandingan & Kesimpulan
    with tabs[3]:
        st.subheader("Perbandingan & Kesimpulan")
        if not combined_df_1.empty and not combined_df_2.empty:
            with st.expander("Di bawah ini adalah hasil perbandingan antara kedua data"):
                changes = {'naik': 0, 'turun': 0}
                turun_indicators = []

                for col in avg_values_1.keys():
                    if col in avg_values_2.keys():
                        value1 = avg_values_1[col] if avg_values_1[col] is not None else 0
                        value2 = avg_values_2[col] if avg_values_2[col] is not None else 0
                        change = value2 - value1
                        change_rounded = round(change, 2)

                        col_label = col.replace(' - Likert Scale', '')
                        if change_rounded > 0:
                            st.write(f"**{col_label}**: Naik sebesar {change_rounded}")
                            changes['naik'] += 1
                        elif change_rounded < 0:
                            st.write(f"**{col_label}**: Turun sebesar {abs(change_rounded)}")
                            changes['turun'] += 1
                            turun_indicators.append(col_label)
                        else:
                            st.write(f"**{col_label}**: Tidak ada perubahan")
            # Kode kesimpulan umum dihilangkan total agar tidak tampak
        else:
            st.warning("Tidak ada data untuk dibandingkan. Pastikan kedua file sudah diunggah dan diproses.")

    # TAB Analisis Lanjutan (Wordcloud & TF-IDF)
    with tabs[4]:
        st.subheader("Analisis Lanjutan (Wordcloud & TF-IDF)")
        if jawaban_domestik or jawaban_ekspor or jawaban_additional_data:
            with st.expander("Visualisasi dan Analisis Kata Kunci (Wordcloud)"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Wordcloud: Permintaan Domestik")
                    if jawaban_domestik:
                        jawaban_domestik_cleaned = preprocess_text(jawaban_domestik)
                        generate_wordcloud(jawaban_domestik_cleaned, 'Wordcloud Permintaan Domestik')
                    else:
                        st.write("Tidak ada jawaban domestik untuk dianalisis.")

                with col2:
                    st.markdown("#### Wordcloud: Permintaan Ekspor")
                    if jawaban_ekspor:
                        jawaban_ekspor_cleaned = preprocess_text(jawaban_ekspor)
                        generate_wordcloud(jawaban_ekspor_cleaned, 'Wordcloud Permintaan Ekspor')
                    else:
                        st.write("Tidak ada jawaban ekspor untuk dianalisis.")

                st.markdown("---")
                st.markdown("### Wordcloud: Indikator Lain (Utilisasi, Persediaan, dll.)")
                if len(jawaban_additional_data) > 0:
                    for column, jawaban_text in jawaban_additional_data.items():
                        if jawaban_text:
                            st.subheader(f"Wordcloud untuk {column}")
                            jawaban_cleaned = preprocess_text(jawaban_text)
                            generate_wordcloud(jawaban_cleaned, f"Wordcloud untuk {column}")
                        else:
                            st.write(f"Tidak ada data untuk {column}")
                else:
                    st.write("Tidak ada Data Indikator Lain untuk divisualisasikan.")

            st.markdown("---")
            with st.expander("Analisis TF-IDF"):
                st.markdown("#### TF-IDF: Permintaan Domestik")
                if jawaban_domestik:
                    corpus_domestik = [preprocess_text(jawaban_domestik)]
                    tfidf_matrix_domestik, feature_names_domestik = calculate_tfidf(corpus_domestik)
                    display_top_tfidf_words(tfidf_matrix_domestik, feature_names_domestik)
                else:
                    st.write("Tidak ada data untuk Permintaan Domestik.")

                st.markdown("#### TF-IDF: Permintaan Ekspor")
                if jawaban_ekspor:
                    corpus_ekspor = [preprocess_text(jawaban_ekspor)]
                    tfidf_matrix_ekspor, feature_names_ekspor = calculate_tfidf(corpus_ekspor)
                    display_top_tfidf_words(tfidf_matrix_ekspor, feature_names_ekspor)
                else:
                    st.write("Tidak ada data untuk Permintaan Ekspor.")

                st.markdown("### TF-IDF: Indikator Lain (Utilisasi, Persediaan, dkk.)")
                if len(jawaban_additional_data) > 0:
                    for column, jawaban_text in jawaban_additional_data.items():
                        st.subheader(f"TF-IDF untuk {column}")
                        jawaban_cleaned = preprocess_text(jawaban_text)
                        corpus = [jawaban_cleaned]
                        tfidf_matrix, feature_names = calculate_tfidf(corpus)
                        display_top_tfidf_words(tfidf_matrix, feature_names)
                else:
                    st.write("Tidak ada data Indikator Lain untuk dianalisis.")
        else:
            st.warning("Belum ada data yang bisa dianalisis di sini.")

    # TAB Topik LLM
    with tabs[5]:
        st.subheader("Topik yang Dibahas (Pemanfaatan Model LLM)")
        st.markdown("Pada bagian ini, kita menampilkan tombol ringkasan per indikator (mirip Domestik & Ekspor).")

        # Ambil rata-rata domestik triwulan sebelumnya & sekarang
        avg_domestik_before = avg_values_1.get('Permintaan Domestik  - Likert Scale', 0)
        avg_domestik_now = avg_values_2.get('Permintaan Domestik  - Likert Scale', 0)

        # Ambil rata-rata ekspor triwulan sebelumnya & sekarang
        avg_ekspor_before = avg_values_1.get('Permintaan Ekspor  - Likert Scale', 0)
        avg_ekspor_now = avg_values_2.get('Permintaan Ekspor  - Likert Scale', 0)

        quarter_now_in = st.text_input("Triwulan Sekarang (contoh: I 2024, II 2023, dsb):", key="quarter_now_in_llm")
        quarter_before_in = st.text_input("Triwulan Sebelumnya (contoh: I 2024, II 2023, dsb):", key="quarter_before_in_llm")

        st.markdown("---")

        st.write("### Generate Summary untuk Permintaan Domestik")
        if st.button("Generate Summary Domestik"):
            if jawaban_domestik.strip() == "":
                st.warning("Tidak ada penjabaran domestik (jawaban_domestik) yang dapat diproses.")
            else:
                llm = Ollama(model="phi4:latest")

                if quarter_now_in and quarter_before_in:
                    prompt_domestik = (
                        f"Berikut adalah data rata-rata nilai antara dua triwulan:\n"
                        f"Permintaan Domestik : {avg_domestik_now} dibandingkan {avg_domestik_before}\n"
                        f"Topik Permintaan Domestik:\n{jawaban_domestik}\n"
                        f"Silakan buat kesimpulan tentang kinerja domestik pada triwulan {quarter_now_in} "
                        f"dibandingkan triwulan {quarter_before_in} berdasarkan data tersebut, "
                        f"tolong menggunakan bahasa indonesia. lalu untuk kata katanya dibuat rapi, sopan, dan netral."
                    )

                    with st.spinner("Sedang memproses kesimpulan domestik..."):
                        kesimpulan_domestik = llm(prompt_domestik)

                    st.subheader("Kesimpulan Domestik:")
                    st.write(kesimpulan_domestik)
                else:
                    st.warning("Mohon isi Triwulan Sekarang dan Triwulan Sebelumnya.")

        st.markdown("---")

        st.write("### Generate Summary untuk Permintaan Ekspor")
        if st.button("Generate Summary Ekspor"):
            if jawaban_ekspor.strip() == "":
                st.warning("Tidak ada penjabaran ekspor (jawaban_ekspor) yang dapat diproses.")
            else:
                llm = Ollama(model="phi4:latest")

                if quarter_now_in and quarter_before_in:
                    prompt_ekspor = (
                        f"Berikut adalah data rata-rata nilai antara dua triwulan:\n"
                        f"Permintaan Ekspor : {avg_ekspor_now} dibandingkan {avg_ekspor_before}\n"
                        f"Topik Permintaan Ekspor:\n{jawaban_ekspor}\n"
                        f"Silakan buat kesimpulan tentang kinerja ekspor pada triwulan {quarter_now_in} "
                        f"dibandingkan triwulan {quarter_before_in} berdasarkan data tersebut, "
                        f"tolong menggunakan bahasa indonesia. lalu untuk kata katanya dibuat rapi, sopan, dan netral."
                    )

                    with st.spinner("Sedang memproses kesimpulan ekspor..."):
                        kesimpulan_ekspor = llm(prompt_ekspor)

                    st.subheader("Kesimpulan Ekspor:")
                    st.write(kesimpulan_ekspor)
                else:
                    st.warning("Mohon isi Triwulan Sekarang dan Triwulan Sebelumnya.")

        st.markdown("---")

        st.write("### Generate Summary untuk Masing-Masing Indikator Lain (Satu Tombol per Indikator)")
        """
          Kita definisikan tombol separate agar prompt-nya sama dengan Domestik/Ekspor:
          1. Memanggil rata-rata (avg) dari avg_values_1 & avg_values_2
          2. Memasukkan text_jawaban
          3. Memasukkan quarter_now_in & quarter_before_in
        """

        for column, text_jawaban in jawaban_additional_data.items():
            # Cari key di avg_values -> "Kapasitas Utilisasi - Likert Scale", dsb.
            col_key = INDICATOR_MAP.get(column, None)
            # Ambil nilai sebelum & sesudah
            avg_add_before = 0
            avg_add_now = 0
            if col_key is not None:
                avg_add_before = avg_values_1.get(col_key, 0)
                avg_add_now = avg_values_2.get(col_key, 0)

            if st.button(f"Generate Summary untuk {column}"):
                if not text_jawaban.strip():
                    st.warning(f"Tidak ada jawaban untuk {column}.")
                else:
                    llm = Ollama(model="phi4:latest")
                    if quarter_now_in and quarter_before_in:
                        # Prompt serupa dengan Permintaan Domestik/Ekspor
                        prompt_indicator = (
                            f"Berikut adalah data rata-rata nilai antara dua triwulan:\n"
                            f"{column} : {avg_add_now} dibandingkan {avg_add_before}\n"
                            f"Topik {column}:\n{text_jawaban}\n"
                            f"Silakan buat kesimpulan tentang kinerja {column} pada triwulan {quarter_now_in} "
                            f"dibandingkan triwulan {quarter_before_in} berdasarkan data tersebut, "
                            f"tolong menggunakan bahasa indonesia. lalu untuk kata katanya dibuat rapi, sopan, dan netral."
                        )

                        with st.spinner(f"Memproses kesimpulan {column}..."):
                            kesimpulan_add = llm(prompt_indicator)

                        st.subheader(f"Kesimpulan {column}:")
                        st.write(kesimpulan_add)
                        st.markdown("---")
                    else:
                        st.warning("Mohon isi Triwulan Sekarang dan Triwulan Sebelumnya.")


if __name__ == "__main__":
    main()
