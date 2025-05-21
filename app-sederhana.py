import streamlit as st
import joblib
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.sparse import hstack
import nltk
import seaborn as sns
import shap
import streamlit.components.v1 as components


# Load model & vectorizers
model = joblib.load("logistic_model.pkl")
vectorizer_text = joblib.load("countvectorizer_text.pkl")
vectorizer_pos = joblib.load("countvectorizer_pos.pkl")

# Custom style
st.set_page_config(page_title="Deteksi Lowongan Kerja Palsu", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Tentang", "Prediksi", "Word Cloud"])

# Prediksi
if page == "Prediksi":
    st.title("🔍 Deteksi Iklan Lowongan Kerja Palsu")
    
    # Penjelasan lebih detail
    st.write("""
    **Deteksi Iklan Lowongan Kerja Palsu** ini menggunakan model pembelajaran mesin untuk menganalisis teks iklan lowongan kerja.
    
    Anda dapat memasukkan teks iklan yang ingin Anda periksa validitasnya. Model akan memprediksi apakah iklan tersebut termasuk **palsu** atau **asli** berdasarkan analisis teks yang diberikan.
    
    - Pastikan teks yang Anda masukkan cukup panjang, minimal terdiri dari 5 kata untuk mendapatkan hasil yang lebih akurat.
    - Jika iklan tersebut **palsu**, model akan memberikan kemungkinan persentase seberapa besar kemungkinan iklan tersebut merupakan iklan palsu.
    - Jika iklan tersebut **asli**, model akan memberikan prediksi dengan probabilitas yang menunjukkan tingkat keasliannya.

    **Langkah-langkah:**
    1. Masukkan teks iklan kerja pada kolom input yang disediakan.
    2. Klik tombol **Prediksi** untuk mendapatkan hasil deteksi.
    
    Pastikan untuk memeriksa hasil prediksi dan analisis yang diberikan untuk menentukan apakah iklan tersebut benar-benar sah atau mencurigakan.
    """)
    # Text area untuk input pengguna
    user_input = st.text_area("📄 Masukkan teks iklan di sini:")

    if st.button("🔎 Prediksi") and user_input.strip():
        if len(user_input.split()) < 5:  # Pastikan input lebih dari 5 kata
            st.warning("⚠️ Input harus terdiri dari setidaknya 5 kata agar model dapat melakukan analisis yang akurat.")
        else:
            # Tokenisasi dan POS tagging
            pos_tags = nltk.pos_tag(nltk.word_tokenize(user_input))
            pos_features = ' '.join(tag[1] for tag in pos_tags)

            # Transformasi teks dan fitur POS menggunakan CountVectorizer
            text_vector = vectorizer_text.transform([user_input])
            pos_vector = vectorizer_pos.transform([pos_features])

            # Gabungkan fitur teks dan POS
            combined_vector = hstack([text_vector, pos_vector])

            # Prediksi dengan model
            prediction = model.predict(combined_vector)[0]
            prob = model.predict_proba(combined_vector)[0][1]

            # Interpretasi fitur penting (SHAP / alternatif manual)
            # Ambil fitur teks penting yang digunakan model
            feature_names = vectorizer_text.get_feature_names_out()
            text_features = text_vector.toarray()[0]

            # Ambil 5 kata dengan nilai tertinggi dari vektor
            top_indices = text_features.argsort()[::-1][:5]
            highlight_words = [feature_names[i] for i in top_indices if text_features[i] > 0]

            # Tampilkan probabilitas sebagai bar chart
            prob_data = [prob, 1 - prob]
            labels = ['Palsu', 'Asli']
            fig, ax = plt.subplots()
            sns.barplot(x=labels, y=prob_data, ax=ax)
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            # Menampilkan hasil prediksi
            if prediction == 1:
                st.error(f"❌ Ini kemungkinan **PALSU** ({prob*100:.2f}%)")
                
                st.subheader("🔎 Kenapa terdeteksi palsu?")
                st.write(
                    f"Iklan ini diklasifikasikan sebagai **palsu** karena mengandung kata-kata seperti:"
                )
                st.markdown(
                    " - " + "\n - ".join([f"`{word}`" for word in highlight_words])
                )
                st.info(
                    "Kata-kata tersebut sering muncul pada iklan palsu berdasarkan pelatihan model, "
                    "seperti janji gaji besar tanpa detail, promosi cepat, atau permintaan transfer uang."
                )
            else:
                st.success(f"✅ Ini kemungkinan **ASLI** ({(1 - prob)*100:.2f}%)")

# Tentang
elif page == "Tentang":
    st.title("ℹ️ Tentang Aplikasi 🧠")

    st.markdown("""
    Aplikasi ini menggunakan model **Logistic Regression** dan teknik **Text Mining** untuk mendeteksi apakah iklan lowongan kerja yang beredar itu **palsu** atau **asli**.

    📊 **Dataset yang Digunakan:**
    - Dataset: *[fake_job_postings.csv](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data)*
    - Himpunan data ini berisi sekitar **18 ribu deskripsi pekerjaan**, di mana sekitar **800 di antaranya adalah iklan palsu**.
    - Data terdiri dari informasi tekstual dan meta-informasi tentang pekerjaan, seperti:
        - **title**: Judul pekerjaan
        - **company_profile**: Profil perusahaan
        - **description**: Deskripsi pekerjaan
        - **requirements**: Persyaratan pekerjaan
        
    🧑‍💻 **Tujuan Dataset:**
    Himpunan data ini digunakan untuk membangun model klasifikasi yang dapat mempelajari deskripsi pekerjaan dan mendeteksi apakah pekerjaan tersebut palsu atau asli. Dengan demikian, model ini dapat membantu menyaring iklan lowongan kerja yang mungkin berisiko atau tidak sah.

    🔗 Dataset ini dapat diakses secara lengkap di **[Kaggle Link](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data)**.
    """)

    # Menampilkan DataFrame asli (fake_job_postings.csv)
    try:
        # Memuat data dari CSV
        df = pd.read_csv('fake_job_postings.csv')

        # Menampilkan DataFrame dengan pencarian dan filter
        st.subheader("Data Iklan Lowongan Kerja Asli")
        
        # Menambahkan opsi pencarian
        search_query = st.text_input("Cari Lowongan (title, description, dll.):")
        if search_query:
            filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)]
            st.dataframe(filtered_df.head(15))
        else:
            st.dataframe(df.head(15))  # Menampilkan beberapa baris pertama

        # Statistik Iklan Asli dan Palsu dengan Tampilan Profesional
        st.subheader("📈 Statistik Iklan Lowongan Kerja")

        # Hitung statistik dasar
        total_ads = len(df)
        true_ads = df['fraudulent'].value_counts().get(0, 0)
        fake_ads = df['fraudulent'].value_counts().get(1, 0)
        true_pct = (true_ads / total_ads) * 100
        fake_pct = (fake_ads / total_ads) * 100

        # Tampilkan statistik dalam 3 kolom
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Total Iklan", f"{total_ads:,}")
        col2.metric("✅ Iklan Asli", f"{true_ads:,}", f"{true_pct:.2f}%")
        col3.metric("❌ Iklan Palsu", f"{fake_ads:,}", f"{fake_pct:.2f}%")

        # Cek tema untuk menyesuaikan warna teks
        theme = st.get_option("theme.base")
        text_color = 'white' if theme == 'dark' else 'black'

        # Visualisasi Distribusi: Pie Chart Profesional
        st.subheader("📊 Distribusi Iklan Asli vs Palsu")

        # Penjelasan distribusi iklan
        st.write("Berikut merupakan distribusi sebaran iklan lowongan pekerjaan palsu dan asli, sesuai dataset tersebut:")

        # Membuat pie chart distribusi
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')

        # Susun data dengan label dan size yang sudah diurutkan
        data = {
            'Asli': true_ads,
            'Palsu': fake_ads
        }
        labels = list(data.keys())
        sizes = list(data.values())
        colors = ['#0D47A1', '#B71C1C']  # Biru dongker & Merah gelap

        def custom_autopct(pct, allvals):
            absolute = int(round(pct/100.*sum(allvals)))
            return f'{pct:.1f}%\n({absolute:,})'

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct=lambda pct: custom_autopct(pct, sizes),
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        )

        # Tambahkan label manual di tengah potongan
        for i, autotext in enumerate(autotexts):
            autotext.set_text(f'{labels[i]}\n{autotext.get_text()}')
            autotext.set_color(text_color)
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
            autotext.set_ha('center')
            autotext.set_va('center')

        ax.axis('equal')

        # Tampilkan chart
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")

# Word Cloud
elif page == "Word Cloud":
    st.title("☁️ Word Cloud: Visualisasi Kata Umum dalam Iklan Lowongan Kerja 🧾")
    
    st.markdown("""
    🔍 **Apa itu Word Cloud?**  
    Word Cloud adalah representasi visual dari kata-kata yang paling sering muncul dalam kumpulan teks. Ukuran kata mencerminkan frekuensi kemunculannya – semakin besar kata, semakin sering ia muncul di dalam data.

    📊 **Mengapa Penting?**  
    Dalam konteks deteksi iklan lowongan kerja **palsu** dan **asli**, Word Cloud dapat digunakan untuk:
    - Mengidentifikasi kata kunci yang **menonjol** di tiap kategori.
    - Memberi wawasan awal sebelum melakukan analisis tekstual lebih lanjut.
    - Membantu mengenali pola-pola umum dari kata-kata yang sering muncul dalam penipuan kerja, seperti "immediate", "payment", atau "training provided".
    
    🧐 **Interpretasi Singkat:**
    - **Iklan Asli ✅** biasanya memiliki kata-kata teknis, posisi pekerjaan, dan istilah profesional seperti *engineer*, *developer*, *team*, dan *experience*.
    - **Iklan Palsu ❌** cenderung menyertakan kata-kata yang terdengar menjanjikan secara finansial namun tidak spesifik, seperti *money*, *income*, *apply now*, atau *no experience*.
    
    👇 Berikut visualisasi Word Cloud dari masing-masing kategori:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟦 Iklan Asli")
        st.image("wc-realjob.png", caption="✅ Word Cloud - Iklan Lowongan Kerja Asli", width=450)
        st.markdown("""
        ### 🔍 Analisis Word Cloud - Iklan Asli
        Berdasarkan word cloud dari **iklan lowongan pekerjaan yang nyata**, terlihat bahwa kata-kata seperti:
        **"experience," "work," "team," "client," "ability," "year," "company," "looking," "time," "service," "project," "provide," "customer,"** dan **"product"** paling sering muncul.

        #### 📌 Insight Utama:
        1. **Fokus pada Pengalaman dan Kemampuan**
           > Kata **"experience"** dominan → menunjukkan kebutuhan atas tenaga kerja berpengalaman. Kata **"ability"** mengarah pada keahlian teknis/soft skill yang spesifik.
        2. **Kerja Tim & Relasi Profesional**
           > Kata **"team"** dan **"client"** sering muncul → banyak pekerjaan asli mengedepankan kolaborasi dan komunikasi antar tim/klien.
        3. **Konteks Bisnis dan Proyek**
           > Munculnya kata seperti **"project," "service," "provide"** → menggambarkan jenis pekerjaan nyata yang berbasis layanan atau target proyek tertentu.
        4. **Citra Perusahaan**
           > Kata **"company"** dan **"product"** → iklan asli cenderung menyampaikan identitas serta nilai perusahaan secara jelas.

        ✅ **Kesimpulan:**  
        Lowongan asli umumnya lebih spesifik, teknis, dan mencerminkan kebutuhan bisnis nyata.
        """)
    with col2:
        st.subheader("🟥 Iklan Palsu")
        st.image("wc-fakejob.png", caption="❌ Word Cloud - Iklan Lowongan Kerja Palsu", width=450)
        st.markdown("""
        ### ⚠️ Analisis Word Cloud - Iklan Palsu
        Dari data iklan lowongan kerja palsu, terlihat bahwa kata-kata seperti:
        **"Customer Service," "data entry," "looking," "High School," "position," "ability," "responsible,"** dan **"communication skill"** sering muncul.

        #### 📌 Pola dan Temuan Utama:
        1. **Pekerjaan Umum & Tidak Spesifik**
           > Kata seperti *"data entry"* atau *"customer service"* sering muncul → menandakan pekerjaan mudah dan kurang jelas peran tugasnya.
        2. **Target Pelamar Minim Pengalaman**
           > Munculnya kata seperti *"High School"* dan *"looking"* → menyasar pencari kerja muda atau fresh graduate tanpa pengalaman.
        3. **Deskripsi Umum dan Umum Lagi**
           > Kata seperti *"ability"*, *"responsible"*, dan *"communication skill"* terlalu generik tanpa penjelasan teknis, berbeda dengan iklan asli.
        4. **Indikasi Penipuan**
           > Penggunaan kata-kata ini sering disertai tawaran gaji tinggi & janji manis. Waspadai permintaan biaya awal, interview online tanpa kontak resmi, atau ketidakjelasan perusahaan.

        ❗ **Kesimpulan:**  
        Word cloud ini menunjukkan bahwa **iklan palsu lebih banyak menggunakan bahasa umum dan membidik mereka yang mudah tertipu oleh tawaran menarik.**
        """)

    st.markdown("""
    💡 **Tips Analisis**  
    - Gunakan Word Cloud ini sebagai titik awal untuk eksplorasi teks.
    - Kombinasikan dengan teknik NLP lainnya seperti **TF-IDF**, **topic modeling**, atau **POS tagging** untuk analisis mendalam.
    - Kata-kata umum yang muncul di Word Cloud bisa digunakan untuk fitur seleksi dalam model prediksi.

    🔁 **Update Dinamis**  
    Word Cloud bisa diperbarui dengan dataset terbaru untuk terus menyesuaikan pola bahasa penipuan kerja yang terus berubah seiring waktu.
    """)
