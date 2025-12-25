# 📊 CustomerVoice Şikayet App

[English](#english) | [Türkçe](#türkçe)

---

## English

CustomerVoice is an end-to-end, AI-powered customer complaint analysis and churn risk management platform. Designed specifically for e-commerce platforms, it provides data-driven insights to improve customer satisfaction and prevent churn.

### 🚀 Application Versions

To provide the best experience across different environments, we offer two versions:

1.  **🏠 Local / Full Version (`streamlit_app_v3.py`):**
    *   **Features:** Includes all modules: Dashboard, Complaint Analysis, Time Series Forecasting, and **Summarization & Sentiment Analysis**.
    *   **Models:** Uses high-performance local models (**mT5** for summarization, **XLM-RoBERTa** for sentiment).
    *   **Environment:** Best for local machines with GPU support.

2.  **☁️ Cloud / Light Version (`streamlit_app_cloud.py`):**
    *   **Features:** Optimized for Streamlit Cloud and GitHub deployment.
    *   **Modules:** Includes Dashboard, Complaint Analysis, and Time Series Forecasting. **Summarization & Sentiment modules are removed** to ensure fast loading and avoid large model download issues.
    *   **Environment:** Perfect for quick demos and public hosting on Streamlit Cloud.

### 🚀 Key Features

*   **🔍 Real-time Complaint Analysis:** 
    *   **Text Classification:** Automatically routes complaints to 10 different operational units using BERT.
    *   **Churn Scoring:** Calculates churn risk score (0-100) using a multi-factor algorithm.
    *   **Semantic Similarity:** Finds the top 10 most similar past complaints using Cosine Similarity.
*   **📊 Dynamic Dashboard:** 
    *   Interactive KPI cards and distributions (Churn Band, Unit, Category).
    *   Advanced filtering by category, unit, and risk level.
*   **📈 Time Series Forecasting:** 
    *   Predicts future complaint volume using Facebook Prophet.
    *   Detects statistical anomalies and provides daily, weekly, and monthly analysis.
*   **📝 Summarization & Sentiment (Local Only):** 
    *   **Summarization:** Condenses long complaints into meaningful summaries using mT5.
    *   **Sentiment Analysis:** Detects dominant emotions (Anger, Frustration, etc.) using XLM-RoBERTa.
*   **🌐 Localization:** Full support for both Turkish and English (TR-EN) with a dynamic language toggle.
*   **🎨 Modern UI:** Sleek, modern dark mode interface built with Streamlit.

### 🤖 Tech Stack & Models

*   **Models:** BERT (Classification), mT5 (Summarization - Local), XLM-RoBERTa (Sentiment - Local), Cosine Similarity (Vector Space).
*   **Libraries:** Python, Streamlit, PyTorch, Transformers (Hugging Face), Pandas, NumPy, Plotly, Scikit-learn, Prophet.

### 🧠 Technical Deep Dive

#### Data Preprocessing
*   **Text Cleaning:** Turkish-specific character normalization, removal of stop-words, and regex-based cleaning of HTML/URL tags.
*   **Tokenization:** Leveraging Hugging Face's `AutoTokenizer` with SentencePiece for efficient multilingual tokenization.

#### Machine Learning Models
*   **Classification (BERT):** Fine-tuned on a labeled dataset of 10k+ complaints to achieve high accuracy in routing complaints to operational units.
*   **Summarization (mT5 / BERT2BERT):** Powerful abstractive summarization optimized for Turkish context.
*   **Sentiment (XLM-RoBERTa / mDeBERTa-v3):** High-performance models to detect emotional nuances.
*   **Forecasting (Prophet):** Handles seasonality (daily, weekly, monthly) and detects anomalies in complaint volume.

#### Churn Scoring Logic
The app uses a custom-weighted algorithm that considers:
1.  **Category Risk:** Some sub-categories have inherently higher churn potential.
2.  **Signal Strength:** Detection of 8 specific "high-danger" phrases (e.g., "legal action", "never again").
3.  **Text Length:** Longer, more detailed complaints often correlate with higher grievance levels.

---

## Türkçe

CustomerVoice, yapay zeka destekli uçtan uca müşteri şikayet analizi ve churn (müşteri kaybı) risk yönetimi platformudur. E-ticaret platformları için özel olarak tasarlanan bu sistem, müşteri memnuniyetini artırmak ve kaybı önlemek için veriye dayalı içgörüler sunar.

### 🚀 Uygulama Versiyonları

Farklı ortamlar için en iyi deneyimi sunmak adına iki farklı versiyon sunuyoruz:

1.  **🏠 Yerel / Tam Versiyon (`streamlit_app_v3.py`):**
    *   **Özellikler:** Tüm modülleri içerir: Dashboard, Şikayet Analizi, Zaman Serisi Tahmini ve **Özetleme & Duygu Analizi**.
    *   **Modeller:** Yüksek performanslı yerel modelleri (**mT5** özetleme, **XLM-RoBERTa** duygu analizi) kullanır.
    *   **Ortam:** GPU desteği olan yerel bilgisayarlar için en iyisidir.

2.  **☁️ Bulut / Hafif Versiyon (`streamlit_app_cloud.py`):**
    *   **Özellikler:** Streamlit Cloud ve GitHub yayını için optimize edilmiştir.
    *   **Modüller:** Dashboard, Şikayet Analizi ve Zaman Serisi Tahmini modüllerini içerir. Hızlı yükleme ve büyük model indirme sorunlarını önlemek için **Özetleme ve Duygu Analizi modülleri kaldırılmıştır**.
    *   **Ortam:** Streamlit Cloud üzerinde hızlı sunum ve genel paylaşım için mükemmeldir.

### 🚀 Temel Özellikler

*   **🔍 Gerçek Zamanlı Şikayet Analizi:** 
    *   **Metin Sınıflandırma:** BERT kullanarak şikayetleri otomatik olarak 10 farklı operasyonel birime yönlendirir.
    *   **Churn Skorlama:** Çok faktörlü algoritma ile churn risk skorunu (0-100) hesaplar.
    *   **Anlamsal Benzerlik:** Cosine Similarity kullanarak geçmişteki en benzer 10 şikayeti bulur.
*   **📊 Dinamik Dashboard:** 
    *   İnteraktif KPI kartları ve dağılımlar (Churn Bandı, Birim, Kategori).
    *   Kategori, birim ve risk seviyesine göre gelişmiş filtreleme.
*   **📈 Zaman Serisi Analizi:** 
    *   Facebook Prophet kullanarak gelecek şikayet yoğunluğunu tahmin eder.
    *   İstatistiksel anomalileri tespit eder; günlük, haftalık ve aylık analiz sunar.
*   **📝 Özetleme & Duygu Analizi (Sadece Yerel):** 
    *   **Özetleme:** mT5 kullanarak uzun şikayet metinlerini anlamlı özetlere dönüştürür.
    *   **Duygu Analizi:** XLM-RoBERTa ile baskın duyguları (Öfke, Hayal Kırıklığı vb.) tespit eder.
*   **🌐 Dil Desteği:** Dinamik dil değiştirme özelliği ile tam Türkçe ve İngilizce (TR-EN) desteği.
*   **🎨 Modern Arayüz:** Streamlit ile geliştirilmiş modern ve şık karanlık tema.

### 🤖 Teknoloji Yığını ve Modeller

*   **Modeller:** BERT (Sınıflandırma), mT5 (Özetleme - Yerel), XLM-RoBERTa (Duygu Analizi - Yerel), Cosine Similarity (Vektör Uzayı).
*   **Kütüphaneler:** Python, Streamlit, PyTorch, Transformers (Hugging Face), Pandas, NumPy, Plotly, Scikit-learn, Prophet.

### 🧠 Teknik Detaylar

#### Veri Önişleme
*   **Metin Temizleme:** Türkçe karakter normalizasyonu, stop-word'lerin temizlenmesi ve regex ile HTML/URL etiketlerinden arındırma.
*   **Tokenization:** Hugging Face `AutoTokenizer` ve SentencePiece ile çok dilli, verimli metin parçalama.

#### Yapay Zeka Modelleri
*   **Sınıflandırma (BERT):** 10.000+ etiketli şikayet verisiyle fine-tune edilerek yüksek doğrulukla birim yönlendirmesi yapar.
*   **Özetleme (mT5 / BERT2BERT):** Türkçe bağlamı için optimize edilmiş güçlü üretken özetleme modelleri.
*   **Duygu Analizi (XLM-RoBERTa / mDeBERTa-v3):** Duygusal tonları tespit eden yüksek performanslı modeller.
*   **Zaman Serisi (Prophet):** Mevsimsellik analizi yapar ve şikayet yoğunluğundaki anomalileri tespit eder.

#### Churn Skorlama Mantığı
Uygulama, şu faktörleri göz önüne alan özel bir ağırlıklandırma algoritması kullanır:
1.  **Kategori Riski:** Bazı alt kategoriler (örn: iade sorunları) doğası gereği daha yüksek kayıp potansiyeline sahiptir.
2.  **Sinyal Gücü:** "Yasal işlem", "bir daha asla" gibi 8 kritik "yüksek tehlike" ifadesinin tespiti.
3.  **Metin Uzunluğu:** Daha uzun ve detaylı şikayetler genellikle daha yüksek mağduriyet seviyesiyle koreledir.

---

## 👥 Developers (High Five Team)

*   **EMRE AÇAR**
*   **ECEM UZMAN**
*   **ELİF CELEP**
*   **İBRAHİM AKDAŞ**
*   **OĞUZHAN EREZ**

---

## 📦 Installation & Usage / Kurulum ve Kullanım

1. **Clone the repository / Depoyu klonlayın:**
   ```bash
   git clone <repository-url>
   cd customer_complaint
   ```

2. **Install dependencies / Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app / Uygulamayı çalıştırın:**

   *   **For Local/Full Version:**
       ```bash
       streamlit run streamlit_app_v3.py
       ```
   *   **For Cloud/Light Version:**
       ```bash
       streamlit run streamlit_app_cloud.py
       ```

---

## 📊 Dataset / Veri Seti

The dataset consists of 10,000+ anonymized real customer complaints, enriched with churn signals and semantic vectors. Personal data has been removed for privacy.

Veri seti, churn sinyalleri ve anlamsal vektörlerle zenginleştirilmiş 10.000'den fazla anonimleştirilmiş gerçek müşteri şikayetinden oluşmaktadır. Gizlilik nedeniyle kişisel veriler arındırılmıştır.
