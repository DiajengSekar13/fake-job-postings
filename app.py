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
from PIL import Image

# Load model & vectorizers
model = joblib.load("logistic_model.pkl")
vectorizer_text = joblib.load("countvectorizer_text.pkl")
vectorizer_pos = joblib.load("countvectorizer_pos.pkl")

# Custom style
st.set_page_config(page_title="Deteksi Lowongan Kerja Palsu", layout="wide")


# Tambahkan logo (jika ada)
logo = Image.open("logo.png")
st.sidebar.image(logo, width=150)

# Header Sidebar
st.sidebar.markdown("## 📊 Job Scam Classifier")
st.sidebar.markdown("### 🚀 Selamat datang!")
st.sidebar.markdown("""
Analisis cerdas untuk membedakan lowongan kerja **asli** dan **palsu**.  
Gunakan fitur prediksi, eksplorasi teks, dan visualisasi word cloud.  
""")

# Navigasi utama
page = st.sidebar.radio(
    "🧭 Navigasi:",
    [
        "📌 Tentang",
        "🔍 Prediksi",
        "☁️ Word Cloud"
    ]
)

# Info tambahan collapsible
with st.sidebar.expander("📂 Info Dataset"):
    st.markdown("""
    - Jumlah data: 17.000+
    - Fitur teks: deskripsi pekerjaan
    - Label: asli atau palsu
    """)

# Kontak / Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
👩‍💻 **Diajeng Sekar**  
📧 [Email](mailto:diajeng.sekar11@gmail.com)  
🔗 [GitHub](https://github.com/DiajengSekar13)  
🔗 [LinkedIn](https://linkedin.com/in/...)
""")

# Prediksi
if page == "🔍 Prediksi":
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
elif page == "📌 Tentang":
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
elif page == "☁️ Word Cloud":
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
    - **Iklan Asli ✅**: kata-kata teknis, posisi pekerjaan, dan istilah profesional seperti *engineer*, *developer*, *team*, dan *experience*.
    - **Iklan Palsu ❌**: kata-kata yang menjanjikan secara finansial namun tidak spesifik, seperti *money*, *income*, *apply now*, atau *no experience*.
    """)

    # PILIHAN WORD CLOUD
    pilihan_wc = st.radio(
        "Pilih kategori iklan untuk ditampilkan Word Cloud-nya:",
        ("Iklan Asli (Real Job)", "Iklan Palsu (Fake Job)")
    )

    if pilihan_wc == "Iklan Asli (Real Job)":
        st.subheader("✅ Word Cloud - Iklan Lowongan Kerja Asli")
        st.image("wc-realjob.png", use_container_width=True)
        st.markdown("""
        ### 🔍 Analisis Word Cloud - Iklan Asli
        Kata-kata dominan:
        **"experience", "work", "team", "client", "ability", "year", "company", "looking", "time", "service", "project", "provide", "customer", "product"**

        #### 📌 Insight:
        - **Pengalaman & Kemampuan:** Kata "experience" dan "ability" menandakan keahlian teknis atau soft skill.
        - **Tim & Klien:** "team", "client" → kolaborasi dan komunikasi penting.
        - **Bisnis & Proyek:** "project", "service", "provide" → fokus pada hasil nyata.
        - **Identitas Perusahaan:** "company", "product" → profesional & transparan.

        ✅ **Kesimpulan:**  
        Lowongan asli lebih spesifik, teknis, dan menggambarkan kebutuhan bisnis yang nyata.
        """)

    elif pilihan_wc == "Iklan Palsu (Fake Job)":
        st.subheader("❌ Word Cloud - Iklan Lowongan Kerja Palsu")
        st.image("wc-fakejob.png", use_container_width=True)
        st.markdown("""
        ### ⚠️ Analisis Word Cloud - Iklan Palsu
        Kata-kata dominan:
        **"Customer Service", "data entry", "looking", "High School", "position", "ability", "responsible", "communication skill"**

        #### 📌 Temuan:
        - **Umum & Tidak Spesifik:** pekerjaan mudah tanpa peran jelas.
        - **Sasar Pelamar Minim Pengalaman:** "High School", "looking".
        - **Deskripsi Generik:** tanpa teknikal detail.
        - **Waspada Penipuan:** kata menarik tapi jebakan.

        ❗ **Kesimpulan:**  
        Iklan palsu menggunakan bahasa umum dan menjanjikan hal-hal bombastis, tapi seringkali tanpa kejelasan.
        """)

    st.markdown("""
    💡 **Tips Analisis**  
    - Jadikan Word Cloud sebagai awal eksplorasi.
    - Lanjutkan dengan teknik NLP seperti **TF-IDF**, **topic modeling**, dan lainnya.
    - Kata-kata di sini dapat dijadikan fitur dalam pemodelan prediktif.

    🔁 **Update Dinamis**  
    Word Cloud bisa diperbarui sesuai tren bahasa iklan kerja terbaru.
    """)