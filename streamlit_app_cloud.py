import re
import pickle
import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# DİL DESTEĞİ (TRANSLATIONS)
# =========================================================
LANG_DICT = {
    "TR": {
        "app_title": "Şikayet Analiz Sistemi",
        "info_box": "Şikayet Metinleri İçin Analiz, Churn Skor Hesaplama ve Time Series Tahminleme Platformu",
        "contact_title": "📧 Geliştirici ile İletişim",
        "contact_linkedin": "🔗 Linkedin Profili",
        "contact_others": "Diğer geliştiriciler için <b>Hakkında</b> bölümüne tıklayınız.",
        "tabs": ["📊 Dashboard", "🔍 Şikayet Analizi", "📈 Zaman Serisi", "ℹ️ Hakkında"],
        "loading_data": "Veriler hazırlanıyor...",
        "loading_models": "Model ve veriler yükleniyor...",
        "dashboard_title": "📊 Dashboard",
        "filters": "🎛️ Filtreler",
        "ana_kategori": "Ana Kategori",
        "alt_kategori": "Alt Kategori",
        "churn_band": "Churn Band",
        "all": "Tümü",
        "kpi_total": "Toplam",
        "kpi_avg_score": "Ort. Skor",
        "kpi_high_risk": "Yüksek Risk",
        "kpi_critical": "Kritik Riskli (MOR)",
        "kpi_high_band": "Yüksek Riskli (KIRMIZI)",
        "kpi_complaint": "Şikayet",
        "dist_band_birim": "🎨 Churn Band & Birim Dağılımı",
        "dist_alt_kategori": "📊 Alt Kategori Şikayet Dağılımı (Churn Band Renkli)",
        "signal_analysis": "🔥 Churn Sinyal Analizi",
        "dist_signal": "📊 Churn Sinyal Dağılımı",
        "birim_avg_score": "📊 Birim × Ortalama Churn Skoru",
        "dist_score": "📈 Churn Skoru Dağılımı",
        "complaint_analysis_title": "## 🔍 Şikayet Analizi",
        "baslik_label": "### 📝 Şikayet Başlığı (Opsiyonel)",
        "baslik_placeholder": "Şikayet başlığı (opsiyonel)",
        "metin_label": "### 📄 Şikayet Metni",
        "metin_placeholder": "Şikayet metni",
        "analyze_btn": "🔍 Analiz et",
        "results_title": "### 📊 Tahmin Sonuçları",
        "responsible_unit": "📋 Sorumlu Birim",
        "confidence": "Güven Oranı",
        "churn_score": "Churn Skoru",
        "churn_signals": "Churn Sinyalleri",
        "similar_complaints": "Benzer Şikayetler",
        "details_btn": "📄 Detayları Gör",
        "ts_title": "## 📈 Zaman Serisi Analizi",
        "ts_filters": "🔍 Veri Filtreleme",
        "ts_cats_label": "Analiz Edilecek Kategori",
        "ts_sigma": "Anomali Hassasiyeti (Sigma)",
        "ts_forecasts": "🚀 Gelecek Tahminleri",
        "ts_analyze_all": "📊 Tüm Zaman Kırılımlarını Analiz Et (Günlük, Haftalık, Aylık)",
        "ts_daily": "📅 Günlük Tahmin Analizi",
        "ts_weekly": "📅 Haftalık Tahmin Analizi",
        "ts_monthly": "📅 Aylık Tahmin Analizi",
        "summary_title": "## 📝 Şikayet Özeti & Duygu Analizi",
        "summary_label": "### 📝 Şikayet Özeti",
        "sentiment_label": "### 📊 Duygu Analizi",
        "start_analyze_btn": "🔍 Analizi Başlat",
        "about_title": "## ℹ️ Proje Hakkında",
        "about_purpose": "### 🎯 Projenin Amacı ve Kapsamı",
        "about_usage": "### 🛠️ Nasıl Kullanılır?",
        "about_models": "### 🤖 Kullanılan Yapay Zeka Modelleri",
        "about_dataset": "### 📊 Kullanılan Veri Seti",
        "about_tech": "### 💻 Teknoloji Yığını (Tech Stack)",
        "about_devs": "### 👥 Geliştiriciler",
        "about_team": "High Five Team",
        "about_members": ""
    },
    "EN": {
        "app_title": "Complaint Analysis System",
        "info_box": "Analysis, Churn Score Calculation and Time Series Forecasting Platform for Complaint Texts",
        "contact_title": "📧 Contact Developer",
        "contact_linkedin": "🔗 Linkedin Profile",
        "contact_others": "Click <b>About</b> section for other developers.",
        "tabs": ["📊 Dashboard", "🔍 Complaint Analysis", "📈 Time Series", "ℹ️ About"],
        "loading_data": "Preparing data...",
        "loading_models": "Loading models and data...",
        "dashboard_title": "📊 Dashboard",
        "filters": "🎛️ Filters",
        "ana_kategori": "Main Category",
        "alt_kategori": "Sub Category",
        "churn_band": "Churn Band",
        "all": "All",
        "kpi_total": "Total",
        "kpi_avg_score": "Avg. Score",
        "kpi_high_risk": "High Risk",
        "kpi_critical": "Critical Risk (PURPLE)",
        "kpi_high_band": "High Risk (RED)",
        "kpi_complaint": "Complaint",
        "dist_band_birim": "🎨 Churn Band & Unit Distribution",
        "dist_alt_kategori": "📊 Sub Category Complaint Distribution (Churn Band Colored)",
        "signal_analysis": "🔥 Churn Signal Analysis",
        "dist_signal": "📊 Churn Signal Distribution",
        "birim_avg_score": "📊 Unit × Average Churn Score",
        "dist_score": "📈 Churn Score Distribution",
        "complaint_analysis_title": "## 🔍 Complaint Analysis",
        "baslik_label": "### 📝 Complaint Title (Optional)",
        "baslik_placeholder": "Complaint title (optional)",
        "metin_label": "### 📄 Complaint Text",
        "metin_placeholder": "Complaint text",
        "analyze_btn": "🔍 Analyze",
        "results_title": "### 📊 Prediction Results",
        "responsible_unit": "📋 Responsible Unit",
        "confidence": "Confidence Score",
        "churn_score": "Churn Score",
        "churn_signals": "Churn Signals",
        "similar_complaints": "Similar Complaints",
        "details_btn": "📄 View Details",
        "ts_title": "## 📈 Time Series Analysis",
        "ts_filters": "🔍 Data Filtering",
        "ts_cats_label": "Category to Analyze",
        "ts_sigma": "Anomaly Sensitivity (Sigma)",
        "ts_forecasts": "🚀 Future Forecasts",
        "ts_analyze_all": "📊 Analyze All Time Intervals (Daily, Weekly, Monthly)",
        "ts_daily": "📅 Daily Forecast Analysis",
        "ts_weekly": "📅 Weekly Forecast Analysis",
        "ts_monthly": "📅 Monthly Forecast Analysis",
        "summary_title": "## 📝 Complaint Summary & Sentiment Analysis",
        "summary_label": "### 📝 Complaint Summary",
        "sentiment_label": "### 📊 Sentiment Analysis",
        "start_analyze_btn": "🔍 Start Analysis",
        "about_title": "## ℹ️ About Project",
        "about_purpose": "### 🎯 Project Purpose and Scope",
        "about_usage": "### 🛠️ How to Use?",
        "about_models": "### 🤖 AI Models Used",
        "about_dataset": "### 📊 Dataset Used",
        "about_tech": "### 💻 Tech Stack",
        "about_devs": "### 👥 Developers",
        "about_team": "High Five Team",
        "about_members": "",
        "units": {
            "Ürün & Kalite Sorunları": "Product & Quality Issues",
            "Finans & İade İşlemleri": "Finance & Refund Transactions",
            "Lojistik & Teslimat": "Logistics & Delivery",
            "Sistem & Sipariş Yönetimi": "System & Order Management",
            "Müşteri Hizmetleri": "Customer Services",
            "Kampanya & Promosyon": "Campaign & Promotion",
            "Üyelik & Hesap": "Membership & Account",
            "Ödeme & Fatura": "Payment & Invoice",
            "Satıcı Performansı": "Seller Performance",
            "Teknik Sorunlar": "Technical Issues",
            "Genel": "General"
        },
        "churn_labels": {
            "MOR": "Critical Risk",
            "KIRMIZI": "High Risk",
            "SARI": "Medium Risk",
            "YEŞİL": "Low Risk",
            "Critical Risk (PURPLE)": "Critical Risk",
            "High Risk (RED)": "High Risk",
            "Medium Risk (YELLOW)": "Medium Risk",
            "Low Risk (GREEN)": "Low Risk"
        },
        "signals": {
            "1. Kesin Kopuş": "1. Absolute Churn",
            "2. Duygusal Kopuş": "2. Emotional Churn",
            "3. Çözümsüzlük & Güven Kaybı": "3. No Solution & Trust Loss",
            "4. Mağduriyet": "4. Grievance",
            "5. Sabır Tükenişi": "5. Patience Exhaustion",
            "6. Tekrarlayan Problem": "6. Recurring Problem",
            "7. Yasal Tehdit": "7. Legal Threat",
            "8. İlk Kez Sorun": "8. First Time Issue"
        },
        "alt_categories": {
            "teslim edilmeyen paket": "Undelivered package",
            "müşteriye teslim edilmeyen paket": "Undelivered package",
            "yanlış veya eksik ürün gönderimi": "Wrong or missing product",
            "kargo teslimat": "Cargo delivery",
            "kargo teslimat sorunu": "Cargo delivery",
            "satıcı sipariş iptali": "Seller order cancellation",
            "iade süreci tamamlanmamış": "Incomplete return process",
            "iade süreci": "Return process",
            "ürün ile ilgili sorunlar": "Product related issues",
            "uygulama": "Application",
            "iade reddi": "Return rejection",
            "garanti sorunu": "Warranty issue",
            "fiyat farkı talebi": "Price difference request"
        }
    }
}

def t(key):
    """Translation helper function"""
    lang = st.session_state.get("lang", "TR")
    return LANG_DICT[lang].get(key, key)

def tc(value, category_type="units"):
    """Helper to translate categorical values like units, signals, etc."""
    lang = st.session_state.get("lang", "TR")
    if not value:
        return value
    
    if lang == "TR":
        return value
    
    # EN translations
    trans_dict = LANG_DICT["EN"].get(category_type, {})
    
    # 1. Doğrudan eşleşme dene
    if value in trans_dict:
        return trans_dict[value]
    
    # 2. Ön ekleri temizle (I., II., 1. vb.)
    # Daha agresif bir temizlik: başta olabilecek boşlukları ve sayı/nokta kombinasyonlarını temizle
    val_str = str(value).strip()
    clean_value = re.sub(r'^[A-Z0-9]+\.\s*', '', val_str).strip()
    
    # Eğer temizlenmiş hali sözlükte varsa çeviriyi döndür
    if clean_value in trans_dict:
        return trans_dict[clean_value]
    
    # Eğer hala bulunamadıysa ama temizleme yapıldıysa temiz halini döndür
    # En azından "I." kısmı gitmiş olur
    return clean_value

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CustomerVoice Şikayet App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# MODERN DARK MODE CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

    /* Ana tema - Modern Dark Slate */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%);
        color: #f1f5f9;
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }
    
    # Logo Alanı Düzenlemesi
    [data-testid="stImage"] {
        display: block;
        margin-left: 0;
        margin-right: auto;
        max-width: 108px;
        padding-top: 0;
        margin-top: 0;
    }
    
    /* Başlıklar */
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif !important;
    }
    h1 { color: #ffffff; font-size: 2.8rem; font-weight: 800; text-align: center; margin-bottom: 2rem; }
    h2 { color: #f8fafc !important; font-size: 1.8rem; font-weight: 700; margin-top: 1.5rem; }
    h3 { color: #f8fafc !important; font-size: 1.4rem; font-weight: 600; }
    
    /* Input alanları */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Butonlar - Modern Indigo Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 15px rgba(99, 102, 241, 0.4); }
    
    /* Metrikler */
    [data-testid="stMetricValue"] {
        color: #f8fafc; 
        font-size: 2.5rem; 
        font-weight: 700;
        font-family: 'Poppins', sans-serif !important;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8; font-size: 1rem; font-weight: 500; }
    
    /* Sekmeler - Modern Minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        border-bottom: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #94a3b8 !important;
        padding: 0.8rem 1.5rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #6366f1 !important;
        border-bottom: 2px solid #6366f1 !important;
    }
    
    /* Churn Band Renkleri - Modern Pastel */
    .churn-mor { background: #8b5cf6; color: white; padding: 1rem; border-radius: 12px; text-align: center; font-weight: 700; margin: 1rem 0; font-family: 'Poppins', sans-serif; }
    .churn-kirmizi { background: #ef4444; color: white; padding: 1rem; border-radius: 12px; text-align: center; font-weight: 700; margin: 1rem 0; font-family: 'Poppins', sans-serif; }
    .churn-sari { background: #f59e0b; color: white; padding: 1rem; border-radius: 12px; text-align: center; font-weight: 700; margin: 1rem 0; font-family: 'Poppins', sans-serif; }
    .churn-yesil { background: #10b981; color: white; padding: 1rem; border-radius: 12px; text-align: center; font-weight: 700; margin: 1rem 0; font-family: 'Poppins', sans-serif; }
    
    /* Kartlar */
    .main-container {
        background: #1e293b;
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #334155;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* Diğer */
    .stAlert { background-color: #1e293b; border: 1px solid #334155; color: #f1f5f9; }
    [data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL YÜKLEME (CACHE)
# =========================================================
@st.cache_resource
def load_models():
    """Model ve tokenizer'ı yükle"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "bert_based_classification_models")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).to(device).eval()
    
    emb_model = AutoModel.from_pretrained(
        model_path, local_files_only=True
    ).to(device).eval()
    
    return tokenizer, clf_model, emb_model, device

@st.cache_data
def load_data():
    """Veri setini yükle"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(base_path, "df_weigthed_final.pkl")
    
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    
    return df

# =========================================================
# SABİTLER
# =========================================================
LABEL_NAMES = [
    "fiyat farkı talebi",
    "garanti sorunu",
    "iade reddi",
    "iade süreci tamamlanmamış",
    "kargo teslimat",
    "satıcı sipariş iptali",
    "teslim edilmeyen paket",
    "uygulama",
    "yanlış veya eksik ürün gönderimi",
    "ürün ile ilgili sorunlar"
]

# CHURN SİNYAL RİSKLERİ (CATEGORY_WEIGHTS)
CATEGORY_WEIGHTS = {
    "1. Kesin Kopuş": 1.00,
    "7. Yasal Tehdit": 0.95,
    "3. Çözümsüzlük & Güven Kaybı": 0.85,
    "2. Duygusal Kopuş": 0.75,
    "5. Sabır Tükenişi": 0.70,
    "6. Tekrarlayan Problem": 0.65,
    "4. Mağduriyet": 0.60,
    "8. İlk Kez Sorun": 0.30
}

# Eski isimle uyumluluk için
CHURN_SIGNAL_RISK = CATEGORY_WEIGHTS

KEYWORDS = [
    # 1️⃣ KESİN KOPUŞ
    ("1. Kesin Kopuş", "bir daha asla"),
    ("1. Kesin Kopuş", "bir daha alışveriş yapmayacağım"),
    ("1. Kesin Kopuş", "bir daha alışveriş"),
    ("1. Kesin Kopuş", "alışveriş yapmayacağım"),
    ("1. Kesin Kopuş", "alışveriş yapmayı düşünmüyorum"),
    ("1. Kesin Kopuş", "güvenerek alışveriş yaptım"),
    ("1. Kesin Kopuş", "bir daha alışveriş yapmayacağım"),
    ("1. Kesin Kopuş", "bir daha asla alışveriş"),
    ("1. Kesin Kopuş", "bir daha alışveriş yapmayı"),
    ("1. Kesin Kopuş", "bir daha alışveriş yapmayı düşünmüyorum"),
    ("1. Kesin Kopuş", "bir daha asla alışveriş yapmayacağım"),
    ("1. Kesin Kopuş", "güvenerek alışveriş yaptım"),
    ("1. Kesin Kopuş", "alışveriş yapmayacağım"),
    ("1. Kesin Kopuş", "bir daha"),
    ("1. Kesin Kopuş", "bir daha alışveriş"),
    ("1. Kesin Kopuş", "bir daha hepsiburada"),
    
    # 2️⃣ DUYGUSAL KOPUŞ
    ("2. Duygusal Kopuş", "hayal"),
    ("2. Duygusal Kopuş", "hayal kırıklığı"),
    ("2. Duygusal Kopuş", "hayal kırıklığına"),
    ("2. Duygusal Kopuş", "bir hayal kırıklığı"),
    ("2. Duygusal Kopuş", "pişman oldum"),
    ("2. Duygusal Kopuş", "güvenerek alışveriş yaptım"),
    ("2. Duygusal Kopuş", "büyük bir hayal"),
    ("2. Duygusal Kopuş", "hayal kırıklığı hepsiburada"),
    ("2. Duygusal Kopuş", "hayal kırıklığı yarattı"),
    ("2. Duygusal Kopuş", "beni hayal kırıklığına"),
    ("2. Duygusal Kopuş", "beni hayal kırıklığına uğrattı"),
    ("2. Duygusal Kopuş", "büyük bir hayal kırıklığı yaşadım"),
    ("2. Duygusal Kopuş", "dalga geçer gibi"),
    
    # 3️⃣ ÇÖZÜMSÜZLÜK & GÜVEN KAYBI
    ("3. Çözümsüzlük & Güven Kaybı", "geri dönüş"),
    ("3. Çözümsüzlük & Güven Kaybı", "bir çözüm"),
    ("3. Çözümsüzlük & Güven Kaybı", "çözüm sunulmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "herhangi bir çözüm"),
    ("3. Çözümsüzlük & Güven Kaybı", "bir çözüm sunulmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "geri dönüş yapılmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "çözüm bekliyorum"),
    ("3. Çözümsüzlük & Güven Kaybı", "geri dönüş yapılmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "ulaşamıyorum"),
    ("3. Çözümsüzlük & Güven Kaybı", "bilgi verilmedi"),
    ("3. Çözümsüzlük & Güven Kaybı", "herhangi bir çözüm sunulmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "henüz bir çözüm sunulmadı"),
    ("3. Çözümsüzlük & Güven Kaybı", "sonuç alamadım"),
    ("3. Çözümsüzlük & Güven Kaybı", "çözüm sunulmuyor"),
    ("3. Çözümsüzlük & Güven Kaybı", "çözüm yok"),
    ("3. Çözümsüzlük & Güven Kaybı", "geri dönüş alamadım"),
    
    # 4️⃣ MAĞDURİYET
    ("4. Mağduriyet", "mağdur"),
    ("4. Mağduriyet", "mağduriyet"),
    ("4. Mağduriyet", "mağduriyetim"),
    ("4. Mağduriyet", "mağduriyetimin"),
    ("4. Mağduriyet", "mağdur oldum"),
    ("4. Mağduriyet", "mağduriyet yaşıyorum"),
    ("4. Mağduriyet", "yaşadığım mağduriyet"),
    ("4. Mağduriyet", "mağduriyetimin giderilmesini"),
    ("4. Mağduriyet", "mağduriyetimin giderilmesini"),
    ("4. Mağduriyet", "yaşadığım mağduriyet"),
    ("4. Mağduriyet", "mağduriyetim devam ediyor"),
    ("4. Mağduriyet", "ve mağduriyetimin giderilmesini talep ediyorum"),
    ("4. Mağduriyet", "mağdur edildim"),
    ("4. Mağduriyet", "mağduriyet yaşıyorum"),
    
    # 5️⃣ SABIR TÜKENİŞİ
    ("5. Sabır Tükenişi", "defalarca"),
    ("5. Sabır Tükenişi", "pişman"),
    ("5. Sabır Tükenişi", "defalarca aramama rağmen"),
    ("5. Sabır Tükenişi", "sürekli"),
    ("5. Sabır Tükenişi", "en kısa sürede"),
    ("5. Sabır Tükenişi", "hala"),
    ("5. Sabır Tükenişi", "halen"),
    ("5. Sabır Tükenişi", "aynı sorun"),
    ("5. Sabır Tükenişi", "sorun devam ediyor"),
    ("5. Sabır Tükenişi", "artık"),
    ("5. Sabır Tükenişi", "acilen"),
    ("5. Sabır Tükenişi", "en kısa sürede giderilmesini bekliyorum"),
    
    # 6️⃣ TEKRARLAYAN PROBLEM
    ("6. Tekrarlayan Problem", "benzer sorunlar"),
    ("6. Tekrarlayan Problem", "benzer bir sorun"),
    ("6. Tekrarlayan Problem", "benzer sorunlar yaşadım"),
    ("6. Tekrarlayan Problem", "benzer bir sorun yaşadım"),
    ("6. Tekrarlayan Problem", "benzer bir sorun yaşamıştım"),
    ("6. Tekrarlayan Problem", "benzer sorunların tekrar"),
    ("6. Tekrarlayan Problem", "benzer sorunların tekrar yaşanmaması"),
    ("6. Tekrarlayan Problem", "benzer durumların tekrar yaşanmaması"),
    ("6. Tekrarlayan Problem", "daha önce de benzer"),
    ("6. Tekrarlayan Problem", "önce de benzer bir sorun"),
    
    # 7️⃣ YASAL TEHDİT
    ("7. Yasal Tehdit", "tüketici hakem"),
    ("7. Yasal Tehdit", "tüketici hakem heyeti"),
    ("7. Yasal Tehdit", "hakem heyeti"),
    ("7. Yasal Tehdit", "hukuki"),
    ("7. Yasal Tehdit", "cimer"),
    ("7. Yasal Tehdit", "yasal haklarımı"),
    ("7. Yasal Tehdit", "tüketici hakları"),
    
    # 8️⃣ İLK KEZ SORUN
    ("8. İlk Kez Sorun", "ilk kez böyle bir sorun"),
    ("8. İlk Kez Sorun", "ilk kez başıma geliyor"),
    ("8. İlk Kez Sorun", "ilk kez böyle bir durum"),
    ("8. İlk Kez Sorun", "ilk kez böyle bir sorunla"),
    ("8. İlk Kez Sorun", "ilk kez böyle bir durumla"),
    ("8. İlk Kez Sorun", "ilk kez"),
]

ALT_KATEGORI_WEIGHTS = {
    "teslim edilmeyen paket": 1.00,
    "yanlış veya eksik ürün gönderimi": 0.90,
    "kargo teslimat": 0.80,
    "satıcı sipariş iptali": 0.75,
    "iade süreci tamamlanmamış": 0.65,
    "ürün ile ilgili sorunlar": 0.60,
    "uygulama": 0.55,
    "iade reddi": 0.40,
    "garanti sorunu": 0.40,
    "fiyat farkı talebi": 0.20
}

# Eski isimle uyumluluk için
ALT_KATEGORI_RISK = ALT_KATEGORI_WEIGHTS

# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def clean_reviews_tr(text):
    """Türkçe metin temizleme"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = text.replace("İ", "i").replace("I", "ı").lower()
    
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b(?:https?|www)\S+\b", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-zçğıöşü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def length_score(tokens):
    """Uzunluk skoru hesapla"""
    if tokens < 20:
        return 0
    elif tokens < 50:
        return 8
    elif tokens < 100:
        return 15
    elif tokens < 200:
        return 20
    else:
        return 25

def churn_signal_score_and_report(temiz_metin):
    """Churn sinyal skoru ve raporu hesapla"""
    active = {}
    
    # Hangi churn sinyalleri var?
    for cat, phrase in KEYWORDS:
        if re.search(rf"\b{re.escape(phrase)}\b", temiz_metin):
            active[cat] = CATEGORY_WEIGHTS[cat]
    
    if not active:
        return 0, []
    
    # En güçlü 2 sinyal (skor için)
    sorted_active = sorted(active.items(), key=lambda x: x[1], reverse=True)
    
    churn_signal_score = sorted_active[0][1] * 30
    if len(sorted_active) > 1:
        churn_signal_score += sorted_active[1][1] * 15
    
    # Tüm aktif sinyallerin listesi - CATEGORY_WEIGHTS'a göre sıralı (yüksekten düşüğe)
    all_signals = [cat for cat, _ in sorted(active.items(), key=lambda x: CATEGORY_WEIGHTS.get(x[0], 0), reverse=True)]
    
    return churn_signal_score, all_signals

def alt_kategori_score(alt_kategori):
    """Alt kategori skoru hesapla"""
    if not alt_kategori:
        return 0
    return ALT_KATEGORI_WEIGHTS.get(alt_kategori, 0) * 20

def churn_band(score):
    """Churn band belirle"""
    if score >= 70:
        return "MOR"
    elif score >= 50:
        return "KIRMIZI"
    elif score >= 35:
        return "SARI"
    else:
        return "YEŞİL"

def get_churn_color(band):
    """Churn band rengi - MODERN PASTEL"""
    colors = {
        "MOR": "#8b5cf6",  # Modern Purple
        "KIRMIZI": "#ef4444",  # Modern Red
        "SARI": "#f59e0b",  # Modern Amber
        "YEŞİL": "#10b981"  # Modern Emerald
    }
    return colors.get(band, "#6366f1")

def get_churn_label(band):
    """Churn band görsel label"""
    if st.session_state.get("lang") == "EN":
        return LANG_DICT["EN"]["churn_labels"].get(band, band)
    labels = {
        "MOR": "Kritik Riskli (MOR)",
        "KIRMIZI": "Yüksek Riskli (KIRMIZI)",
        "SARI": "Orta Riskli (SARI)",
        "YEŞİL": "Düşük Riskli (YEŞİL)"
    }
    return labels.get(band, band)

def remove_category_number(category):
    """Kategori isminden sayıyı kaldır (örn: '5. Sabır Tükenişi' -> 'Sabır Tükenişi')"""
    # Sayı ve nokta ile başlayan kısmı kaldır
    import re
    return re.sub(r'^\d+\.\s*', '', category).strip()

def get_category_icon(category_name):
    """Kategori için uygun ikon döndür"""
    icons = {
        "Kesin Kopuş": "🚫",
        "Duygusal Kopuş": "💔",
        "Çözümsüzlük & Güven Kaybı": "❓",
        "Mağduriyet": "😔",
        "Sabır Tükenişi": "😤",
        "Tekrarlayan Problem": "🔄",
        "Yasal Tehdit": "⚖️",
        "İlk Kez Sorun": "🆕"
    }
    return icons.get(category_name, "📌")

def get_responsible_unit(alt_kategori):
    """Alt kategori için sorumlu birim döndür"""
    unit_mapping = {
        "ürün ile ilgili sorunlar": "Ürün & Kalite Sorunları",
        "iade süreci tamamlanmamış": "Finans & İade İşlemleri",
        "iade reddi": "Finans & İade İşlemleri",
        "kargo teslimat": "Lojistik & Teslimat",
        "teslim edilmeyen paket": "Lojistik & Teslimat",
        "fiyat farkı talebi": "Finans & İade İşlemleri",
        "yanlış veya eksik ürün gönderimi": "Ürün & Kalite Sorunları",
        "satıcı sipariş iptali": "Sistem & Sipariş Yönetimi",
        "uygulama": "Sistem & Sipariş Yönetimi",
        "garanti sorunu": "Ürün & Kalite Sorunları"
    }
    unit = unit_mapping.get(alt_kategori, "Genel")
    if st.session_state.get("lang") == "EN":
        return LANG_DICT["EN"]["units"].get(unit, unit)
    return unit

# =========================================================
# ANA TAHMİN FONKSİYONU
# =========================================================
def predict_complaint(baslik, sikayet_metni, df, tokenizer, clf_model, emb_model, device, top_k_similar=5):
    """Şikayet analizi yap"""
    # Başlık boşsa sadece metin kullan
    if baslik and baslik.strip():
        full_text = f"{baslik} {sikayet_metni}"
    else:
        full_text = sikayet_metni
    
    # 1. ALT KATEGORİ (BERT)
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        logits = clf_model(**inputs).logits
    
    probs = F.softmax(logits, dim=1)[0]
    top_idx = torch.argmax(probs).item()
    
    alt_kategori = LABEL_NAMES[top_idx]
    olasilik = round(probs[top_idx].item() * 100, 2)
    
    # Tüm kategorilerin olasılıklarını al
    all_probs = {LABEL_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(len(LABEL_NAMES))}
    
    # 2. CHURN SCORE
    temiz_metin = clean_reviews_tr(full_text)
    token_len = len(temiz_metin.split())
    
    # Alt kategori skoru
    alt_score = ALT_KATEGORI_WEIGHTS.get(alt_kategori, 0) * 20
    
    # Churn sinyal skoru
    active = {}
    for cat, phrase in KEYWORDS:
        if re.search(rf"\b{re.escape(phrase)}\b", temiz_metin):
            active[cat] = CATEGORY_WEIGHTS[cat]
    
    if active:
        sorted_active = sorted(active.items(), key=lambda x: x[1], reverse=True)
        churn_signal_score = sorted_active[0][1] * 30
        if len(sorted_active) > 1:
            churn_signal_score += sorted_active[1][1] * 15
    else:
        churn_signal_score = 0
    
    # Length skoru
    length_score_value = length_score(token_len)
    
    # Toplam churn score
    churn_score = churn_signal_score + alt_score + length_score_value
    
    # Aktif sinyaller listesi - CATEGORY_WEIGHTS'a göre sıralı (yüksekten düşüğe)
    if active:
        triggered = [cat for cat, _ in sorted(active.items(), key=lambda x: CATEGORY_WEIGHTS.get(x[0], 0), reverse=True)]
    else:
        triggered = []
    
    churn_band_value = churn_band(churn_score)
    
    # 3. EN BENZER 5 ŞİKAYET
    def get_embedding(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        with torch.no_grad():
            return emb_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
    
    query_emb = get_embedding(full_text)
    corpus_emb = np.vstack(df["embedding"].values)
    
    sims = cosine_similarity(query_emb, corpus_emb)[0]
    top_idx = np.argsort(sims)[::-1][:top_k_similar]
    
    similarity_df = df.iloc[top_idx][
        ["tarih_saat", "kullanici", "baslik", "sikayet_metni"]
    ].copy()
    similarity_df["benzerlik_skoru"] = [round(sims[i], 4) for i in top_idx]
    similarity_df = similarity_df.reset_index(drop=True)
    
    return {
        "alt_kategori": alt_kategori,
        "olasilik": olasilik,
        "all_probs": all_probs,
        "churn_score": round(churn_score, 2),
        "churn_band": churn_band_value,
        "churn_signal_score": round(churn_signal_score, 2),
        "length_score": length_score_value,
        "alt_kategori_score": round(alt_score, 2),
        "triggered_categories": triggered,
        "similar_complaints": similarity_df,
        "token_len": token_len
    }

# =========================================================
# DASHBOARD FONKSİYONU
# =========================================================
def show_dashboard(df):
    """Dashboard - KPI Kartları, Kategori Dağılımları, Grafikler"""
    
    # Dark mode CSS - Filtreler dahil
    st.markdown("""
    <style>
    .stSelectbox label { color: #fff !important; font-weight: 600 !important; font-size: 1rem !important; }
    div[data-baseweb="select"] > div { background-color: #1e293b !important; color: #fff !important; border: 1px solid #334155 !important; }
    div[data-baseweb="select"] span { color: #fff !important; font-weight: 500 !important; }
    div[data-baseweb="select"] svg { fill: #fff !important; }
    [data-baseweb="popover"] { background-color: #1e293b !important; }
    [data-baseweb="popover"] li { color: #fff !important; background-color: #1e293b !important; }
    [data-baseweb="popover"] li:hover { background-color: #334155 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"## {t('dashboard_title')}")
    
    # Filtreler - Sıra: Ana Kategori, Alt Kategori, Churn Band
    st.markdown(f"### {t('filters')}")
    f1, f2, f3 = st.columns(3)
    
    df_copy = df.copy()
    
    # 1. Ana Kategori
    with f1:
        raw_ana_kats = sorted(df_copy['Ana_Kategori'].dropna().unique().tolist()) if 'Ana_Kategori' in df_copy.columns else []
        ana_kats_display = {}
        for k in raw_ana_kats:
            # "I. ", "II. ", "IV. " gibi Romen rakamı veya sayı ve nokta ile başlayan kısımları temizle
            val_str = str(k).strip()
            clean_k = re.sub(r'^[A-Z0-9]+\.\s*', '', val_str).strip()
            # Temizlenmiş haliyle çeviri yap, orijinali (k) değer olarak sakla
            display_name = tc(clean_k, "units")
            ana_kats_display[display_name] = k
        
        ana_kats_options = [t('all')] + sorted(list(ana_kats_display.keys()))
        sel_ana_display = st.selectbox(t('ana_kategori'), ana_kats_options)
        sel_ana = ana_kats_display.get(sel_ana_display, t('all'))
    
    # 2. Alt Kategori
    with f2:
        if sel_ana != t('all') and 'Ana_Kategori' in df_copy.columns and 'Alt_Kategori' in df_copy.columns:
            filtered_df_for_cats = df_copy[df_copy['Ana_Kategori'] == sel_ana]
            raw_alt_kats = sorted(filtered_df_for_cats['Alt_Kategori'].dropna().unique().tolist())
        else:
            raw_alt_kats = sorted(df_copy['Alt_Kategori'].dropna().unique().tolist()) if 'Alt_Kategori' in df_copy.columns else []
        
        alt_kats_display = {tc(k, "alt_categories"): k for k in raw_alt_kats}
        alt_kats_options = [t('all')] + sorted(list(alt_kats_display.keys()))
        sel_alt_display = st.selectbox(t('alt_kategori'), alt_kats_options)
        sel_alt = alt_kats_display.get(sel_alt_display, t('all'))
    
    # 3. Churn Band
    with f3:
        raw_bands = df_copy['churn_band'].dropna().unique().tolist() if 'churn_band' in df_copy.columns else []
        bands_display = {tc(k, "churn_labels"): k for k in raw_bands}
        bands_options = [t('all')] + sorted(list(bands_display.keys()))
        sel_band_display = st.selectbox(t('churn_band'), bands_options)
        sel_band = bands_display.get(sel_band_display, t('all'))
    
    # Filtreleme
    fdf = df_copy.copy()
    if sel_ana != t('all'):
        fdf = fdf[fdf['Ana_Kategori'] == sel_ana]
    if sel_alt != t('all'):
        fdf = fdf[fdf['Alt_Kategori'] == sel_alt]
    if sel_band != t('all'):
        fdf = fdf[fdf['churn_band'] == sel_band]
    
    n = len(fdf)
    
    # MODERN PASTEL RENKLER
    colors = {'MOR': '#8b5cf6', 'KIRMIZI': '#ef4444', 'SARI': '#f59e0b', 'YEŞİL': '#10b981'}
    
    # Churn Band hesapla
    mor = (fdf['churn_band'] == 'MOR').sum() if 'churn_band' in fdf.columns else 0
    kirmizi = (fdf['churn_band'] == 'KIRMIZI').sum() if 'churn_band' in fdf.columns else 0
    sari = (fdf['churn_band'] == 'SARI').sum() if 'churn_band' in fdf.columns else 0
    yesil = (fdf['churn_band'] == 'YEŞİL').sum() if 'churn_band' in fdf.columns else 0
    avg_score = fdf['churn_score'].mean() if 'churn_score' in fdf.columns and n > 0 else 0
    
    # ═══════════════════════════════════════════════════════════════
    # KPI KARTLARI - FİLTRELERİN ALTINDA (ÇERÇEVE İLE, EŞİT BOYUT)
    # ═══════════════════════════════════════════════════════════════
    high_risk = mor + kirmizi
    high_pct = (high_risk/n*100) if n > 0 else 0
    
    kpi_style = """
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1rem; text-align: center; min-height: 120px; display: flex; flex-direction: column; justify-content: center; overflow: hidden; word-wrap: break-word;">
        <p style="color: #94a3b8; margin: 0; font-size: 0.85rem; white-space: nowrap;">{icon} {label}</p>
        <h2 style="color: {color}; margin: 0.2rem 0; font-size: 1.8rem; font-weight: 700; line-height: 1.2; overflow: hidden; text-overflow: ellipsis;">{value}</h2>
        <p style="color: #64748b; margin: 0; font-size: 0.8rem; white-space: nowrap;">{sub}</p>
    </div>
    """
    
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.markdown(kpi_style.format(border='#334155', icon='📊', label=t('kpi_total'), color='#fff', value=f'{n:,}', sub=t('kpi_complaint')), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_style.format(border='#334155', icon='📉', label=t('kpi_avg_score'), color='#f59e0b', value=f'{avg_score:.1f}', sub='Churn'), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_style.format(border='#334155', icon='🚨', label=t('kpi_high_risk'), color='#ef4444', value=f'{high_risk:,}', sub=f'%{high_pct:.1f}'), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_style.format(border=colors['MOR'], icon='🟣', label=t('kpi_critical'), color=colors['MOR'], value=f'{mor:,}', sub=f'%{(mor/n*100) if n > 0 else 0:.1f}'), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_style.format(border=colors['KIRMIZI'], icon='🔴', label=t('kpi_high_band'), color=colors['KIRMIZI'], value=f'{kirmizi:,}', sub=f'%{(kirmizi/n*100) if n > 0 else 0:.1f}'), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # CHURN BAND + BİRİM TEK KPI KARTI
    # ═══════════════════════════════════════════════════════════════
    st.markdown(f"### {t('dist_band_birim')}")
    
    col_band, col_birim = st.columns(2)
    
    # Churn Band Kartı
    with col_band:
        st.markdown(f"#### 🎯 {t('churn_band')}")
        labels_pie = [tc('MOR', 'churn_labels'), tc('KIRMIZI', 'churn_labels'), tc('SARI', 'churn_labels'), tc('YEŞİL', 'churn_labels')]
        fig_band = go.Figure(data=[go.Pie(
            labels=labels_pie,
            values=[mor, kirmizi, sari, yesil],
            hole=0.6,
            marker=dict(colors=[colors['MOR'], colors['KIRMIZI'], colors['SARI'], colors['YEŞİL']]),
            textinfo='label+value+percent',
            textfont=dict(size=12, color='#fff'),
            textposition='outside',
            pull=[0.05, 0.02, 0, 0]
        )])
        fig_band.add_annotation(
            text=f"<b>{n:,}</b><br>{t('kpi_total')}",
            x=0.5, y=0.5, font=dict(size=16, color='#fff'), showarrow=False
        )
        fig_band.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#fff',
            height=350,
            margin=dict(l=80, r=80, t=50, b=80),
            showlegend=False
        )
        st.plotly_chart(fig_band, use_container_width=True)
    
    # Birim Kartı (Sadece Ana Kategoriler) - Indigo Shades
    with col_birim:
        st.markdown(f"#### 📁 {'Birim Dağılımı' if st.session_state.lang == 'TR' else 'Unit Distribution'}")
        if 'Ana_Kategori' in fdf.columns and n > 0:
            birim_counts = fdf['Ana_Kategori'].value_counts().sort_values(ascending=True)
            birim_labels = [tc(idx, 'units') for idx in birim_counts.index]
            
            # Indigo tonları (açıktan koyuya)
            indigo_tonlar = ['#a5b4fc', '#818cf8', '#6366f1', '#4f46e5', '#4338ca', '#3730a3']
            bar_colors = (indigo_tonlar * 2)[:len(birim_counts)]
            
            fig_birim = go.Figure()
            fig_birim.add_trace(go.Bar(
                x=birim_counts.values,
                y=birim_labels,
                orientation='h',
                marker=dict(color=bar_colors),
                text=[f"{v:,}" for v in birim_counts.values],
                textposition='inside',
                textfont=dict(color='#fff', size=14),
                hovertemplate="<b>%{y}</b><br>" + (f"{t('kpi_total')}: %{{x:,}}" if st.session_state.lang == 'TR' else "Total: %{x:,}") + "<extra></extra>",
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.9)', font_size=14, font_family='Arial', font_color='#fff')
            ))
            fig_birim.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#fff',
                height=350,
                margin=dict(l=180, r=20, t=20, b=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=12, color='#fff'))
            )
            st.plotly_chart(fig_birim, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # ALT KATEGORİ DAĞILIMI - STACKED BAR (MOR, KIRMIZI, SARI, YEŞİL)
    # ═══════════════════════════════════════════════════════════════
    st.markdown(f"### {t('dist_alt_kategori')}")
    
    if 'Alt_Kategori' in fdf.columns and 'churn_band' in fdf.columns and n > 0:
        # Her alt kategori için churn band sayıları
        alt_cats = fdf['Alt_Kategori'].value_counts().head(10).index.tolist()
        
        # Her band için sayıları hesapla ve birimleri ekle
        mor_vals = []
        kirmizi_vals = []
        sari_vals = []
        yesil_vals = []
        alt_cats_with_unit = []
        
        for alt_cat in alt_cats:
            cat_df = fdf[fdf['Alt_Kategori'] == alt_cat]
            mor_vals.append((cat_df['churn_band'] == 'MOR').sum())
            kirmizi_vals.append((cat_df['churn_band'] == 'KIRMIZI').sum())
            sari_vals.append((cat_df['churn_band'] == 'SARI').sum())
            yesil_vals.append((cat_df['churn_band'] == 'YEŞİL').sum())
            # Birim ekle (kısa format)
            birim = get_responsible_unit(alt_cat)
            # Birim ismini kısalt ve çevir
            if st.session_state.lang == "EN":
                birim_short = tc(birim, 'units').replace("Product & Quality Issues", "Product").replace("Finance & Refund Transactions", "Finance").replace("Logistics & Delivery", "Logistics").replace("System & Order Management", "System")
                alt_cat_translated = tc(alt_cat, 'alt_categories')
                alt_cats_with_unit.append(f"{birim_short} | {alt_cat_translated}")
            else:
                birim_short = birim.replace("Ürün & Kalite Sorunları", "Ürün").replace("Finans & İade İşlemleri", "Finans").replace("Lojistik & Teslimat", "Lojistik").replace("Sistem & Sipariş Yönetimi", "Sistem")
                alt_cats_with_unit.append(f"{birim_short} | {alt_cat}")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=tc('YEŞİL', 'churn_labels'), 
            y=alt_cats_with_unit, x=yesil_vals, orientation='h', marker_color=colors['YEŞİL'],
            hovertemplate="<b>%{y}</b><br>" + (f"{tc('YEŞİL', 'churn_labels')}: %{{x:,}}" if st.session_state.lang == 'TR' else f"{tc('YEŞİL', 'churn_labels')}: %{{x:,}}") + "<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            name=tc('SARI', 'churn_labels'), 
            y=alt_cats_with_unit, x=sari_vals, orientation='h', marker_color=colors['SARI'],
            hovertemplate="<b>%{y}</b><br>" + (f"{tc('SARI', 'churn_labels')}: %{{x:,}}" if st.session_state.lang == 'TR' else f"{tc('SARI', 'churn_labels')}: %{{x:,}}") + "<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            name=tc('KIRMIZI', 'churn_labels'), 
            y=alt_cats_with_unit, x=kirmizi_vals, orientation='h', marker_color=colors['KIRMIZI'],
            hovertemplate="<b>%{y}</b><br>" + (f"{tc('KIRMIZI', 'churn_labels')}: %{{x:,}}" if st.session_state.lang == 'TR' else f"{tc('KIRMIZI', 'churn_labels')}: %{{x:,}}") + "<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            name=tc('MOR', 'churn_labels'), 
            y=alt_cats_with_unit, x=mor_vals, orientation='h', marker_color=colors['MOR'],
            hovertemplate="<b>%{y}</b><br>" + (f"{tc('MOR', 'churn_labels')}: %{{x:,}}" if st.session_state.lang == 'TR' else f"{tc('MOR', 'churn_labels')}: %{{x:,}}") + "<extra></extra>"
        ))
        
        fig.update_layout(
            barmode='stack',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#fff',
            height=450,
            margin=dict(l=200, r=30, t=20, b=10),
            xaxis=dict(showgrid=False, zeroline=False, title=dict(text=t('kpi_complaint'), font=dict(size=14, color='#fff')), tickfont=dict(size=12, color='#fff')),
            yaxis=dict(
                showgrid=False, 
                tickfont=dict(size=11, color='#fff'),
                autorange='reversed'
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=12, color='#ffffff'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # CHURN SİNYAL ANALİZİ - İKİ GRAFİK
    # ═══════════════════════════════════════════════════════════════
    st.markdown(f"### {t('signal_analysis')}")
    
    sig_col1, sig_col2 = st.columns(2)
    
    with sig_col1:
        st.markdown(f"#### {t('dist_signal')}")
        if 'top_churn_signal_1' in fdf.columns and n > 0:
            signal1_counts = fdf['top_churn_signal_1'].value_counts()
            signal2_counts = fdf['top_churn_signal_2'].value_counts() if 'top_churn_signal_2' in fdf.columns else pd.Series()
            
            all_signals = signal1_counts.add(signal2_counts, fill_value=0)
            all_signals_dict = all_signals.to_dict()
            all_signals_sorted_list = sorted(
                all_signals_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:8]
            all_signals_sorted = pd.Series(dict(all_signals_sorted_list))
            
            signal_names = [remove_category_number(tc(s, 'signals')) for s in all_signals_sorted.index]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=all_signals_sorted.values,
                y=signal_names,
                orientation='h',
                marker=dict(color='#6366f1'),
                text=[f"{int(v):,}" for v in all_signals_sorted.values],
                textposition='inside',
                textfont=dict(color='#fff', size=14),
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.9)', font_size=14, font_family='Arial', font_color='#fff')
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#fff',
                height=400,
                margin=dict(l=150, r=60, t=10, b=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(
                    showgrid=False, 
                    tickfont=dict(size=11, color='#fff'), 
                    autorange='reversed'
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with sig_col2:
        st.markdown(f"#### {t('birim_avg_score')}")
        if 'Ana_Kategori' in fdf.columns and 'churn_score' in fdf.columns and n > 0:
            birim_churn = fdf.groupby('Ana_Kategori').agg(
                avg_churn=('churn_score', 'mean'),
                count=('churn_score', 'count')
            ).reset_index().sort_values('avg_churn', ascending=True)
            
            birim_labels = [tc(b, 'units') for b in birim_churn['Ana_Kategori']]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=birim_churn['avg_churn'],
                y=birim_labels,
                orientation='h',
                marker=dict(color='#818cf8'),
                text=[f"{v:.1f}" for v in birim_churn['avg_churn']],
                textposition='inside',
                textfont=dict(color='#fff', size=12),
                hovertemplate="<b>%{y}</b><br>" + (f"{t('kpi_avg_score')}: %{{x:.1f}}" if st.session_state.lang == 'TR' else "Avg Score: %{x:.1f}") + "<extra></extra>",
                hoverlabel=dict(bgcolor='rgba(0,0,0,0.9)', font_size=14, font_family='Arial', font_color='#fff')
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#fff',
                height=400,
                margin=dict(l=180, r=20, t=10, b=10),
                xaxis=dict(title=dict(text=t('kpi_avg_score'), font=dict(size=14, color='#fff')), showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=12, color='#fff')),
                yaxis=dict(showgrid=False, tickfont=dict(size=12, color='#fff'))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════
    # CHURN SKORU DAĞILIMI
    # ═══════════════════════════════════════════════════════════════
    st.markdown(f"### {t('dist_score')}")
    
    if 'churn_score' in fdf.columns and n > 0:
        # Renkleri ve etiketleri hazırla
        plot_df = fdf.copy()
        plot_colors = colors.copy()
        
        if st.session_state.lang == "EN":
            plot_df['churn_band'] = plot_df['churn_band'].map(lambda x: tc(x, 'churn_labels'))
            plot_colors = {tc(k, 'churn_labels'): v for k, v in colors.items()}

        fig = px.histogram(
            plot_df, x='churn_score', nbins=30,
            color='churn_band',
            color_discrete_map=plot_colors
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#fff',
            height=350,
            margin=dict(l=40, r=40, t=30, b=40),
            xaxis=dict(title=dict(text=t('churn_score'), font=dict(size=14, color='#fff')), tickfont=dict(size=12, color='#fff')),
            yaxis=dict(title=dict(text=t('kpi_complaint'), font=dict(size=14, color='#fff')), tickfont=dict(size=12, color='#fff')),
            legend=dict(font=dict(size=12, color='#ffffff'))
        )
        st.plotly_chart(fig, use_container_width=True)
        
        median_score = fdf['churn_score'].median()
        high_count = len(fdf[fdf['churn_score'] >= 60])
        st.markdown(f"""
        <p style="color: #94a3b8; font-size: 0.9rem; text-align: center;">
        📈 {t('kpi_avg_score') if st.session_state.lang == 'TR' else 'Average'}: <b>{avg_score:.1f}</b> | {'Medyan' if st.session_state.lang == 'TR' else 'Median'}: <b>{median_score:.1f}</b> | {t('kpi_high_risk')}: <b>{high_count}</b> {t('kpi_complaint')}
        </p>
        """, unsafe_allow_html=True)

# =========================================================
# ŞİKAYET ANALİZİ FONKSİYONU (MEVCUT EKRAN)
# =========================================================
def show_complaint_analysis(tokenizer, clf_model, emb_model, device, df):
    """Şikayet Analizi sekmesi - Mevcut ekran"""
    # Başlık (Logo olduğu için h2 yapıldı)
    st.markdown(f"{t('complaint_analysis_title')}")
    st.markdown("---")
    
    # Örnek metinler
    ornek_baslik = "Sipariş Görüntüleme Sorunu" if st.session_state.lang == "TR" else "Order Visibility Issue"
    ornek_metin = """Hepsiburada'dan sipariş verdim ancak siparişim 'Siparişlerim' kısmında görünmüyor. Sipariş veremez olduk. Artık lütfen yardımcı olur musunuz?""" if st.session_state.lang == "TR" else """I placed an order on Hepsiburada but it doesn't show up in 'My Orders'. We can't even place orders anymore. Please help me."""
    
    # Session state ile ilk yükleme kontrolü
    if 'initial_analysis_done' not in st.session_state:
        st.session_state.initial_analysis_done = False
    
    # ANA LAYOUT - SOL: INPUTLAR, SAĞ: SONUÇLAR
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown(f"{t('baslik_label')}")
        baslik = st.text_input(
            "Şikayet Başlığı",
            value=ornek_baslik,
            placeholder=t('baslik_placeholder'),
            label_visibility="collapsed"
        )
        
        st.markdown(f"{t('metin_label')}")
        sikayet_metni = st.text_area(
            "Şikayet Metni",
            value=ornek_metin,
            height=400,
            placeholder=t('metin_placeholder'),
            label_visibility="collapsed"
        )
        
        # ANALİZ BUTONU
        analiz_butonu = st.button(
            t('analyze_btn'),
            type="primary",
            use_container_width=True
        )
        
        # Benzer şikayet sayısı sabit 10
        top_k = 10
    
    with col_right:
        st.markdown(f"{t('results_title')}")
        
        # İlk yüklemede veya butona basıldığında analiz yap
        should_analyze = analiz_butonu or (not st.session_state.initial_analysis_done and sikayet_metni and sikayet_metni.strip())
        
        # SONUÇLAR SAĞ KOLONDA - SADECE METRİKLER VE CHURN
        if should_analyze:
            if not sikayet_metni or not sikayet_metni.strip():
                st.warning("⚠️ " + ("Lütfen şikayet metnini doldurun!" if st.session_state.lang == "TR" else "Please fill in the complaint text!"))
            else:
                with st.spinner("🔄 " + ("Analiz yapılıyor..." if st.session_state.lang == "TR" else "Analyzing...")):
                    try:
                        results = predict_complaint(
                            baslik, sikayet_metni, df, 
                            tokenizer, clf_model, emb_model, device, top_k
                        )
                        
                        # Sonuçları session state'e kaydet
                        st.session_state.analysis_results = results
                        st.session_state.last_metin = sikayet_metni
                        
                        st.success("✅ " + ("Analiz tamamlandı!" if st.session_state.lang == "TR" else "Analysis completed!"))
                        st.markdown("---")
                        
                        # İlk analiz tamamlandı olarak işaretle
                        st.session_state.initial_analysis_done = True
                        
                        # SORUMLU BİRİM VE ALT KATEGORİ - EN ÜSTTE
                        responsible_unit = get_responsible_unit(results["alt_kategori"])
                        alt_kategori_display = tc(results["alt_kategori"], 'alt_categories').title()
                        
                        st.markdown(f'<p style="font-size: 1.6rem; font-weight: 600; color: #6366f1; margin-bottom: 0.5rem;">{t("responsible_unit")}: <strong>{responsible_unit}</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size: 1.5rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.5rem;">{t("alt_kategori")}: <strong>{alt_kategori_display}</strong></p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-size: 1.4rem; margin-top: 0.5rem; margin-bottom: 1rem;">{t("confidence")}: <strong>%{results["olasilik"]}</strong></p>', unsafe_allow_html=True)
                        
                        # CHURN ANALİZİ - KOMPAKT VE YANYANA
                        churn_score = results["churn_score"]
                        churn_band_value = results["churn_band"]
                        churn_band_label = get_churn_label(churn_band_value)
                        color = get_churn_color(churn_band_value)
                        
                        # Churn Skoru ve Band yanyana
                        st.markdown(f"""
                        <div style="background: #1e293b; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border: 1px solid {color};">
                            <p style="font-size: 1.4rem; font-weight: 700; margin: 0; color: #fff;">
                                {t('churn_score')}: <span style="color: {color};">{churn_score}</span> 
                                <span style="background: {color}; color: #fff; padding: 0.2rem 0.8rem; border-radius: 8px; margin-left: 0.5rem; font-size: 1.2rem;">{churn_band_label}</span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Gauge grafik küçültüldü
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=churn_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            number={'font': {'size': 28, 'color': '#ffffff'}},
                            gauge={
                                'axis': {'range': [None, 100], 'tickcolor': '#ffffff', 'tickfont': {'size': 12}},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.2)"},
                                    {'range': [35, 50], 'color': "rgba(245, 158, 11, 0.2)"},
                                    {'range': [50, 70], 'color': "rgba(239, 68, 68, 0.2)"},
                                    {'range': [70, 100], 'color': "rgba(139, 92, 246, 0.2)"}
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 2},
                                    'thickness': 0.75,
                                    'value': churn_score
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#ffffff",
                            height=180,
                            margin=dict(t=30, b=30, l=15, r=15)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ {'Hata oluştu' if st.session_state.lang == 'TR' else 'Error occurred'}: {str(e)}")
                        st.exception(e)
        else:
            st.info("👈 " + ("Sol taraftaki formu doldurup 'Analiz et' butonuna tıklayın." if st.session_state.lang == "TR" else "Fill in the form on the left and click 'Analyze'."))
    
    # TETİKLENEN KATEGORİLER VE BENZER ŞİKAYETLER - ALTA TAM GENİŞLİKTE
    if should_analyze and sikayet_metni and sikayet_metni.strip():
        try:
            # Analiz sonuçlarını session state'ten al (zaten yapılmışsa)
            if 'analysis_results' in st.session_state and st.session_state.get('last_metin') == sikayet_metni:
                results = st.session_state.analysis_results
            else:
                if 'analysis_results' in st.session_state:
                    results = st.session_state.analysis_results
                else:
                    with st.spinner("🔄 Analiz yapılıyor..."):
                        results = predict_complaint(
                            baslik, sikayet_metni, df, 
                            tokenizer, clf_model, emb_model, device, top_k
                        )
                        st.session_state.analysis_results = results
                        st.session_state.last_metin = sikayet_metni
            
            # CHURN SİNYALLERİ - TETİKLENEN KATEGORİLER
            st.markdown(f'<div class="main-container"><p style="font-size: 1.5rem; font-weight: 700; text-align: center; margin-bottom: 1.5rem;">{t("churn_signals")}</p>', unsafe_allow_html=True)
            
            if results["triggered_categories"]:
                sorted_categories = sorted(
                    results["triggered_categories"],
                    key=lambda x: CATEGORY_WEIGHTS.get(x, 0),
                    reverse=True
                )
                
                num_categories = len(sorted_categories)
                cols = st.columns(4)
                
                if num_categories <= 4:
                    start_col = (4 - num_categories) // 2
                    
                    for idx, cat_raw in enumerate(sorted_categories):
                        col_idx = start_col + idx
                        # Çeviri ve numara temizleme
                        display_name = remove_category_number(tc(cat_raw, 'signals'))
                        icon = get_category_icon(remove_category_number(cat_raw)) # İkon için TR ismi kullan
                        
                        with cols[col_idx]:
                            category_html = f"""
                            <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 1.2rem; text-align: center;">
                                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                                <div style="font-size: 0.95rem; font-weight: 600; color: #ffffff;">{display_name}</div>
                            </div>
                            """
                            st.markdown(category_html, unsafe_allow_html=True)
                else:
                    for row_start in range(0, num_categories, 4):
                        row_categories = sorted_categories[row_start:row_start+4]
                        cols = st.columns(4)
                        
                        for idx, cat_raw in enumerate(row_categories):
                            # Çeviri ve numara temizleme
                            display_name = remove_category_number(tc(cat_raw, 'signals'))
                            icon = get_category_icon(remove_category_number(cat_raw))
                            
                            with cols[idx]:
                                category_html = f"""
                                <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 1.2rem; text-align: center;">
                                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                                    <div style="font-size: 0.95rem; font-weight: 600; color: #ffffff;">{display_name}</div>
                                </div>
                                """
                                st.markdown(category_html, unsafe_allow_html=True)
            else:
                msg = "Tetiklenen churn sinyali bulunamadı." if st.session_state.lang == "TR" else "No churn signals triggered."
                st.markdown(f'<p style="text-align: center; color: #94a3b8; font-size: 1rem;">{msg}</p>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # BENZER ŞİKAYETLER
            st.markdown(f'<div class="main-container"><p style="font-size: 1.5rem; font-weight: 700; text-align: center; margin-bottom: 1.5rem;">{t("similar_complaints")}</p>', unsafe_allow_html=True)
            
            if not results["similar_complaints"].empty:
                for idx, row in results["similar_complaints"].iterrows():
                    similarity_pct = row['benzerlik_skoru'] * 100
                    col_main, col_score = st.columns([4, 1])
                    
                    with col_main:
                        col_user, col_date = st.columns(2)
                        with col_user:
                            if 'kullanici' in row and pd.notna(row['kullanici']):
                                st.markdown(f'<p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">👤 {row["kullanici"]}</p>', unsafe_allow_html=True)
                        with col_date:
                            if 'tarih_saat' in row and pd.notna(row['tarih_saat']):
                                st.markdown(f'<p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">📅 {row["tarih_saat"]}</p>', unsafe_allow_html=True)
                        
                        st.markdown(f'<p style="font-size: 1.1rem; margin-top: 0.5rem; margin-bottom: 0.5rem; color: #f1f5f9;"><strong>{row["baslik"]}</strong></p>', unsafe_allow_html=True)
                        
                        with st.expander(t("details_btn")):
                            st.markdown(f'<p style="color: #cbd5e1; font-size: 0.95rem;">{row["sikayet_metni"]}</p>', unsafe_allow_html=True)
                    
                    with col_score:
                        st.markdown(f'<div style="text-align: right;"><p style="font-size: 1.4rem; font-weight: 700; color: #6366f1; margin: 0;">%{similarity_pct:.1f}</p></div>', unsafe_allow_html=True)
                    
                    st.markdown('<hr style="border-color: #334155; margin: 1rem 0;">', unsafe_allow_html=True)
            else:
                msg = "Benzer şikayet bulunamadı." if st.session_state.lang == "TR" else "No similar complaints found."
                st.markdown(f'<p style="text-align: center; color: #94a3b8; font-size: 1rem;">{msg}</p>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Hata oluştu: {str(e)}")
            st.exception(e)

# =========================================================
# ZAMAN SERİSİ ANALİZİ SAYFASI
# =========================================================
def dataset_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Kategorileri 10 kategoriye düşür"""
    if "sorun" not in df.columns:
        if "Alt_Kategori" in df.columns:
            df = df.rename(columns={"Alt_Kategori": "sorun"})
        else:
            st.error("Dosyada 'sorun' veya 'Alt_Kategori' kolonu bulunamadı.")
            return pd.DataFrame()
    
    cols_to_keep = ["tarih_saat", "sorun"]
    if "text" in df.columns:
        cols_to_keep.append("text")
    
    df = df[cols_to_keep].copy()

    etiket_eslestirme = {
        'ürün ile ilgili sorunlar': 'ürün ile ilgili sorunlar',
        'teslim edilmeyen paket': 'müşteriye teslim edilmeyen paket',
        'kargoya teslim edilmeyen paket': 'müşteriye teslim edilmeyen paket',
        'kargoya geç teslim': 'müşteriye teslim edilmeyen paket',
        'geç teslimat': 'kargo teslimat sorunu',
        'hasarlı paket': 'kargo teslimat sorunu',
        'iade süreci tamamlanmamış': 'iade süreci',
        'eksik ücret iadesi': 'iade süreci',
        'iade reddi': 'iade reddi',
        'uygulama sorunu': 'uygulama',
        'kupon sorunu': 'uygulama',
        'ödeme sorunu': 'uygulama',
        'siparişi iptal edememe': 'uygulama',
        'satıcı sipariş iptali': 'satıcı sipariş iptali',
        'yanlış veya eksik ürün gönderimi': 'yanlış veya eksik ürün gönderimi',
        'kullanılmış ürün gönderimi': 'yanlış veya eksik ürün gönderimi',
        'garanti sorunu': 'garanti sorunu',
        'fiyat farkı talebi': 'fiyat farkı talebi',
    }

    df["sorun"] = df["sorun"].astype(str).str.strip()
    df["kategoriler"] = df["sorun"].map(etiket_eslestirme)
    df = df[df["kategoriler"].notna()].copy()
    return df

def find_strong_active_start(ts: pd.DataFrame, window: int = 7, min_avg: float = 5.0):
    if ts.empty:
        return None
    roll = ts["y"].rolling(window=window, min_periods=window).mean()
    valid = roll[roll >= min_avg]
    if valid.empty:
        return None
    return valid.index[0]

def slice_to_strong_active(ts: pd.DataFrame, window: int = 7, min_avg: float = 5.0):
    start = find_strong_active_start(ts, window=window, min_avg=min_avg)
    if start is None:
        return ts.copy(), None
    return ts.loc[start:].copy(), start

def show_time_series_analysis():
    """Zaman Serisi Tahmin ve Anomali Tespiti sekmesi"""
    st.markdown(f"{t('ts_title')}")
    st.markdown("---")

    # SESSION STATE KONTROLÜ (Grafiklerin kaybolmaması için)
    if 'ts_daily_fig' not in st.session_state: st.session_state.ts_daily_fig = None
    if 'ts_weekly_fig' not in st.session_state: st.session_state.ts_weekly_fig = None
    if 'ts_monthly_fig' not in st.session_state: st.session_state.ts_monthly_fig = None

    @st.cache_data
    def read_excel_file(file):
        return pd.read_excel(file)

    @st.cache_data
    def prepare_df(df: pd.DataFrame):
        d = dataset_preprocessing(df)
        d["tarih_saat"] = pd.to_datetime(d["tarih_saat"], errors="coerce")
        d = d.dropna(subset=["tarih_saat"])
        return d

    @st.cache_data
    def resample_counts(df: pd.DataFrame, freq: str, min_count: int = 1):
        ts = (
            df.set_index("tarih_saat")
            .resample(freq)
            .size()
            .reset_index(name="y")
        )
        ts.columns = ["ds", "y"]
        if min_count > 0:
            ts = ts[ts["y"] >= min_count].copy()
        return ts

    @st.cache_data
    def run_prophet(ts: pd.DataFrame, periods: int, freq: str):
        from prophet import Prophet
        model = Prophet(yearly_seasonality=True, weekly_seasonality=(freq == "D"), daily_seasonality=False)
        model.fit(ts)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        return forecast

    @st.cache_data
    def detect_anomalies(ts: pd.DataFrame, forecast: pd.DataFrame, sigma: float):
        merged = ts.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left")
        merged["anomaly"] = 0
        merged.loc[merged["y"] > merged["yhat_upper"] + sigma * (merged["yhat_upper"] - merged["yhat"]), "anomaly"] = 1
        merged.loc[merged["y"] < merged["yhat_lower"] - sigma * (merged["yhat"] - merged["yhat_lower"]), "anomaly"] = -1
        return merged

    def plot_forecast(actual: pd.DataFrame, forecast: pd.DataFrame, title: str, color="#6366f1", anomalies=None):
        fig = go.Figure()
        
        # Güven aralığı
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
            y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Güven Aralığı" if st.session_state.lang == "TR" else "Confidence Interval"
        ))
        
        # Gerçek veri
        fig.add_trace(go.Scatter(
                x=actual["ds"],
                y=actual["y"],
            mode="lines+markers", 
            name="Gerçek Veri" if st.session_state.lang == "TR" else "Actual Data", 
            line=dict(color="#94a3b8", width=2),
            marker=dict(size=4)
        ))
        
        # Tahmin
        fig.add_trace(go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat"],
                mode="lines",
                name="Tahmin" if st.session_state.lang == "TR" else "Forecast", 
            line=dict(color=color, dash="dash", width=2)
        ))
        
        # Anomaliler
        if anomalies is not None:
            outliers = anomalies[anomalies["anomaly"] != 0]
            if not outliers.empty:
                fig.add_trace(go.Scatter(
                    x=outliers["ds"],
                    y=outliers["y"],
                    mode="markers",
                    name="Anomali" if st.session_state.lang == "TR" else "Anomaly",
                    marker=dict(color="#ef4444", size=10, symbol="x", line=dict(width=2, color="#fff"))
                ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#fff',
            title=dict(text=title, font=dict(size=18)),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Tarih" if st.session_state.lang == "TR" else "Date"),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title=t('kpi_complaint')),
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    # Veri yükleme logic
    base_path = os.path.dirname(os.path.abspath(__file__))
    default_pkl_path = os.path.join(base_path, "df_weigthed_final.pkl")
    df_raw = None
    if os.path.exists(default_pkl_path):
        with open(default_pkl_path, "rb") as f:
            df_raw = pickle.load(f)

    if df_raw is None:
        st.error("❌ " + ("Veri yüklenemedi." if st.session_state.lang == "TR" else "Data could not be loaded."))
        return

    df_clean = prepare_df(df_raw)

    # Filtreleme Alanı
    st.markdown(f"### {t('ts_filters')}")
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        all_label = t('all')
        raw_cats = sorted(df_clean["kategoriler"].dropna().astype(str).unique().tolist())
        # Kategori isimlerini çevir (Alt Kategorileri kullanacak şekilde düzelttim)
        cats_display = {}
        for k in raw_cats:
            # "I. ", "II. ", "IV. " gibi Romen rakamı veya sayı ve nokta ile başlayan kısımları temizle
            val_str = str(k).strip()
            clean_k = re.sub(r'^[A-Z0-9]+\.\s*', '', val_str).strip()
            # Çeviri sözlüğünde 'alt_categories' kısmına bakması gerektiğini belirttim
            display_name = tc(clean_k, "alt_categories")
            cats_display[display_name] = k
            
        cats_options = [all_label] + sorted(list(cats_display.keys()))
        
        category_filter_display = st.selectbox(t('ts_cats_label'), options=cats_options)
        category_filter = cats_display.get(category_filter_display, all_label)
        
        # Kategori değişirse grafikleri sıfırla
        if st.session_state.get('last_category_filter') != category_filter:
            st.session_state.ts_daily_fig = None
            st.session_state.ts_weekly_fig = None
            st.session_state.ts_monthly_fig = None
            st.session_state.last_category_filter = category_filter
    
    with col_f2:
        sigma_val = st.slider(t('ts_sigma'), 0.5, 3.0, 1.5, 0.1)

    def filter_df(df: pd.DataFrame):
        d = df.copy()
        all_label = "Tümü" if st.session_state.lang == "TR" else "All"
        if category_filter and category_filter != all_label:
            d = d[d["kategoriler"] == category_filter]
        return d

    # TAHMİN BÖLÜMÜ
    st.markdown(f"### {t('ts_forecasts')}")
    
    # Tümünü Analiz Et Butonu - En Üstte Tam Genişlik
    if st.button(t('ts_analyze_all'), use_container_width=True, type="primary"):
        with st.spinner(("Tüm zaman serileri analiz ediliyor, lütfen bekleyin..." if st.session_state.lang == "TR" else "Analyzing all time series, please wait...")):
            # 1. Günlük
            ts_d = resample_counts(filter_df(df_clean), freq="D", min_count=5)
            if len(ts_d) >= 14:
                fc_d = run_prophet(ts_d, periods=30, freq="D")
                anom_d = detect_anomalies(ts_d, fc_d, sigma_val)
                st.session_state.ts_daily_fig = plot_forecast(ts_d, fc_d, ("30 Günlük Tahmin ve Anomali Analizi" if st.session_state.lang == "TR" else "30-Day Forecast and Anomaly Analysis"), anomalies=anom_d)
            
            # 2. Haftalık
            ts_w = resample_counts(filter_df(df_clean), freq="W", min_count=10)
            if len(ts_w) >= 8:
                fc_w = run_prophet(ts_w, periods=12, freq="W")
                anom_w = detect_anomalies(ts_w, fc_w, sigma_val)
                st.session_state.ts_weekly_fig = plot_forecast(ts_w, fc_w, ("12 Haftalık Tahmin ve Anomali Analizi" if st.session_state.lang == "TR" else "12-Week Forecast and Anomaly Analysis"), color="#10b981", anomalies=anom_w)
            
            # 3. Aylık
            ts_m = resample_counts(filter_df(df_clean), freq="ME", min_count=20)
            if len(ts_m) >= 6:
                fc_m = run_prophet(ts_m, periods=6, freq="ME")
                anom_m = detect_anomalies(ts_m, fc_m, sigma_val)
                st.session_state.ts_monthly_fig = plot_forecast(ts_m, fc_m, ("6 Aylık Tahmin ve Anomali Analizi" if st.session_state.lang == "TR" else "6-Month Forecast and Anomaly Analysis"), color="#f59e0b", anomalies=anom_m)
            
            st.success("✅ " + ("Tüm analizler başarıyla tamamlandı!" if st.session_state.lang == "TR" else "All analyses completed successfully!"))

    st.markdown("<br>", unsafe_allow_html=True)
    col_t1, col_t2, col_t3 = st.columns(3)
    
    with col_t1:
        if st.button(t('ts_daily'), use_container_width=True):
            with st.spinner(("Günlük analiz yapılıyor..." if st.session_state.lang == "TR" else "Performing daily analysis...")):
                ts = resample_counts(filter_df(df_clean), freq="D", min_count=5)
                if len(ts) >= 14:
                    fc = run_prophet(ts, periods=30, freq="D")
                    anomalies = detect_anomalies(ts, fc, sigma_val)
                    st.session_state.ts_daily_fig = plot_forecast(ts, fc, ("30 Günlük Tahmin ve Anomali Analizi" if st.session_state.lang == "TR" else "30-Day Forecast and Anomaly Analysis"), anomalies=anomalies)
                else:
                    st.warning("⚠️ " + ("Günlük analiz için yeterli veri bulunamadı (En az 14 gün gerekli)." if st.session_state.lang == "TR" else "Not enough data for daily analysis (at least 14 days required)."))
                    
    with col_t2:
        if st.button(t('ts_weekly'), use_container_width=True):
            with st.spinner(("Haftalık analiz yapılıyor..." if st.session_state.lang == "TR" else "Performing weekly analysis...")):
                ts = resample_counts(filter_df(df_clean), freq="W", min_count=10)
                if len(ts) >= 8:
                    fc = run_prophet(ts, periods=12, freq="W")
                    anomalies = detect_anomalies(ts, fc, sigma_val)
                    st.session_state.ts_weekly_fig = plot_forecast(ts, fc, ("12 Haftalık Tahmin ve Anomali Analizi" if st.session_state.lang == "TR" else "12-Week Forecast and Anomaly Analysis"), color="#10b981", anomalies=anomalies)
                else:
                    st.warning("⚠️ " + ("Haftalık analiz için yeterli veri bulunamadı (En az 8 hafta gerekli)." if st.session_state.lang == "TR" else "Not enough data for weekly analysis (at least 8 weeks required)."))

    with col_t3:
        if st.button(t('ts_monthly'), use_container_width=True):
            with st.spinner(("Aylık analiz yapılıyor..." if st.session_state.lang == "TR" else "Performing monthly analysis...")):
                ts = resample_counts(filter_df(df_clean), freq="ME", min_count=20)
                if len(ts) >= 6:
                    fc = run_prophet(ts, periods=6, freq="ME")
                    anomalies = detect_anomalies(ts, fc, sigma_val)
                    st.session_state.ts_monthly_fig = plot_forecast(ts, fc, ("6 Aylık Tahmin ve Anomali Analizi" if st.session_state.lang == "TR" else "6-Month Forecast and Anomaly Analysis"), color="#f59e0b", anomalies=anomalies)
                else:
                    st.warning("⚠️ " + ("Aylık analiz için yeterli veri bulunamadı (En az 6 ay gerekli)." if st.session_state.lang == "TR" else "Not enough data for monthly analysis (at least 6 months required)."))

    # HAZIRLANMIŞ GRAFİKLERİ GÖSTER
    if st.session_state.ts_daily_fig:
        st.plotly_chart(st.session_state.ts_daily_fig, use_container_width=True)
    if st.session_state.ts_weekly_fig:
        st.plotly_chart(st.session_state.ts_weekly_fig, use_container_width=True)
    if st.session_state.ts_monthly_fig:
        st.plotly_chart(st.session_state.ts_monthly_fig, use_container_width=True)

    # BİLGİ KARTLARI
    all_label = "Tümü" if st.session_state.lang == "TR" else "All"
    if category_filter != all_label:
        fdf = filter_df(df_clean)
        msg = f"💡 **{category_filter}** kategorisi için toplam **{len(fdf)}** kayıt üzerinden analiz yapılmaktadır." if st.session_state.lang == "TR" else f"💡 Analysis is performed over **{len(fdf)}** records for the **{category_filter}** category."
        st.info(msg)
    else:
        msg = f"💡 Tüm kategoriler üzerinden toplam **{len(df_clean)}** kayıt analiz edilmektedir." if st.session_state.lang == "TR" else f"💡 Total of **{len(df_clean)}** records are analyzed across all categories."
        st.info(msg)

# =========================================================
# ÖZET ANALİZİ FONKSİYONU
# =========================================================
# =========================================================
# HAKKINDA SAYFASI
# =========================================================
# =========================================================
# HAKKINDA SAYFASI
# =========================================================
def show_about_section():
    """Proje ve Geliştirici Bilgileri"""
    st.markdown(f"{t('about_title')}")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # PROJENİN AMACI
        st.markdown(f"{t('about_purpose')}")
        if st.session_state.lang == "TR":
            st.markdown("""
            Bu proje, e-ticaret platformlarındaki müşteri şikayetlerini yapay zeka ve veri bilimi teknikleri kullanarak uçtan uca analiz etmek için tasarlanmıştır. Temel odak noktamız, **müşteri kaybını (churn) önlemek** ve müşteri memnuniyetini artırmak için işletmelere hızlı, veriye dayalı ve aksiyon alınabilir içgörüler sunmaktır.
            
            Sistem; metin sınıflandırma, duygu analizi, anomali tespiti ve gelecek tahmini gibi karmaşık süreçleri tek bir çatı altında toplayarak, şikayet yönetimini operasyonel bir yük olmaktan çıkarıp stratejik bir avantaja dönüştürür.
            """)
        else:
            st.markdown("""
            This project is designed to analyze customer complaints on e-commerce platforms end-to-end using AI and data science techniques. Our main focus is to provide businesses with fast, data-driven, and actionable insights to **prevent customer churn** and increase customer satisfaction.
            
            The system collects complex processes such as text classification, sentiment analysis, anomaly detection, and future forecasting under a single roof, transforming complaint management from an operational burden into a strategic advantage.
            """)

        # NASIL KULLANILIR
        st.markdown(f"{t('about_usage')}")
        if st.session_state.lang == "TR":
            st.markdown("""
            Sistem 5 ana modülden oluşmaktadır:
            1.  **📊 Dashboard:** Platformdaki tüm şikayetlerin genel panoramasını sunar. Birim bazlı dağılımlar, yüksek riskli şikayet oranları ve churn skorları burada görselleştirilir.
            2.  **🔍 Şikayet Analizi:** Yeni gelen veya tekil bir şikayeti anlık olarak analiz eder. Şikayeti ilgili birime yönlendirir, churn riskini hesaplar ve geçmişteki en benzer 10 şikayeti bularak çözüm önerisi sunar.
            3.  **📈 Zaman Serisi:** Gelecekteki şikayet yoğunluğunu tahmin eder. Geçmiş verilerdeki "anomali" noktalarını tespit ederek, operasyonel kapasite planlamasına yardımcı olur.
            4.  **📝 Özet & Duygu:** Çok uzun ve karmaşık şikayet metinlerini tek bir paragrafa indirger. Aynı zamanda müşterinin baskın duygusunu (Öfke, Tehdit vb.) tespit ederek önceliklendirme yapmanızı sağlar.
            5.  **ℹ️ Hakkında:** Projenin teknik altyapısı ve geliştirici ekip hakkında bilgiler içerir.
            """)
        else:
            st.markdown("""
            The system consists of 5 main modules:
            1.  **📊 Dashboard:** Provides a general panorama of all complaints on the platform. Unit-based distributions, high-risk complaint rates, and churn scores are visualized here.
            2.  **🔍 Complaint Analysis:** Analyzes a new or individual complaint instantly. Directs the complaint to the relevant unit, calculates churn risk, and finds the 10 most similar past complaints to offer solution suggestions.
            3.  **📈 Time Series:** Predicts future complaint density. Helps with operational capacity planning by detecting "anomaly" points in historical data.
            4.  **📝 Summary & Sentiment:** Reduces very long and complex complaint texts to a single paragraph. Also detects the dominant emotion of the customer (Anger, Threat, etc.) to allow for prioritization.
            5.  **ℹ️ About:** Contains information about the technical infrastructure of the project and the developer team.
            """)

        # MODELLER
        st.markdown(f"{t('about_models')}")
        if st.session_state.lang == "TR":
            st.markdown("""
            Projemizde her bir görev için özelleşmiş, sektör standardı modeller kullanılmıştır:
            *   **BERT (Bidirectional Encoder Representations from Transformers):** Şikayetlerin 10 farklı alt kategoriye sınıflandırılması ve birim yönlendirmesi için kullanıldı.
            *   **mT5 (Multilingual T5):** Çok dilli özetleme yeteneği sayesinde uzun şikayet metinlerini anlamlı özetlere dönüştürür.
            *   **Facebook Prophet:** Zaman serisi verilerinde mevsimsellik ve trend analizi yaparak gelecek tahmini ve anomali tespiti gerçekleştirir.
            *   **XLM-RoBERTa (Zero-Shot):** Şikayet metinlerindeki ince duygusal tonları (duygu analizi) önceden tanımlanmış etiketlere göre tespit eder.
            *   **Cosine Similarity (Vektör Uzayı):** Şikayetlerin anlamsal benzerliklerini hesaplayarak benzer vakaları eşleştirir.
            """)
        else:
            st.markdown("""
            Specialized, industry-standard models were used for each task in our project:
            *   **BERT (Bidirectional Encoder Representations from Transformers):** Used for classifying complaints into 10 different subcategories and for unit redirection.
            *   **mT5 (Multilingual T5):** Transforms long complaint texts into meaningful summaries thanks to its multilingual summarization capability.
            *   **Facebook Prophet:** Performs trend analysis and seasonality detection in time series data for future forecasting and anomaly detection.
            *   **XLM-RoBERTa (Zero-Shot):** Detects subtle emotional tones (sentiment analysis) in complaint texts according to predefined labels.
            *   **Cosine Similarity (Vector Space):** Matches similar cases by calculating the semantic similarities of complaints.
            """)

        # VERİ SETİ BİLGİSİ
        st.markdown(f"{t('about_dataset')}")
        if st.session_state.lang == "TR":
            st.markdown("""
            Analizlerimizde kullanılan veri seti, Türkiye'nin önde gelen e-ticaret platformlarına yönelik gerçek müşteri geri bildirimlerinden derlenmiştir:
            *   **Boyut:** Yaklaşık **10.000+ satır** ve **35 sütundan** oluşan zengin bir veri yapısı.
            *   **Veri Kaynağı:** Çeşitli dijital müşteri deneyimi platformlarından toplanan halka açık şikayet kayıtları.
            *   **Sütun İçerikleri:**
                *   *Metinsel:* Şikayet başlığı, içeriği ve firma bilgileri.
                *   *Kategorik:* Otomatik atanan ana ve alt operasyonel birimler.
                *   *Analitik:* Churn sinyalleri, risk skorları ve anlamsal vektörler (embeddings).
                *   *Zaman:* Şikayetin oluşturulduğu tarih ve saat bilgileri.
            *   **Zenginleştirme:** Ham şikayet metinleri; churn skorları, anlamsal vektörler ve zaman damgaları ile işlenerek analize hazır hale getirilmiştir.
            *   **Gizlilik:** Kişisel veriler arındırılarak anonimleştirilmiş bir yapı sunulmaktadır.
            """)
        else:
            st.markdown("""
            The dataset used in our analysis was compiled from real customer feedback for Turkey's leading e-commerce platforms:
            *   **Size:** A rich data structure consisting of approximately **10,000+ rows** and **35 columns**.
            *   **Data Source:** Publicly available complaint records collected from various digital customer experience platforms.
            *   **Column Details:**
                *   *Textual:* Complaint title, content, and company info.
                *   *Categorical:* Automatically assigned main and sub operational units.
                *   *Analytical:* Churn signals, risk scores, and semantic vectors (embeddings).
                *   *Time:* Date and time info of the complaint.
            *   **Enrichment:** Raw complaint texts have been processed with churn scores, semantic vectors, and timestamps to be ready for analysis.
            *   **Privacy:** A structure is provided where personal data is anonymized.
            """)

    with col2:
        # TECH STACK
        st.markdown(f"{t('about_tech')}")
        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.05); border-left: 3px solid #6366f1; border-radius: 8px; padding: 1rem; margin-bottom: 2rem;">
            <ul style="list-style-type: none; padding: 0; font-size: 0.95rem;">
                <li>🔹 <b>Python:</b> {"Ana programlama dili" if st.session_state.lang == "TR" else "Core programming language"}</li>
                <li>🔹 <b>Streamlit:</b> {"Web arayüzü ve UI yönetimi" if st.session_state.lang == "TR" else "Web interface and UI management"}</li>
                <li>🔹 <b>PyTorch:</b> {"Derin öğrenme altyapısı" if st.session_state.lang == "TR" else "Deep learning infrastructure"}</li>
                <li>🔹 <b>Transformers (Hugging Face):</b> {"NLP modellerinin yönetimi" if st.session_state.lang == "TR" else "NLP model management"}</li>
                <li>🔹 <b>Pandas & NumPy:</b> {"Veri manipülasyonu" if st.session_state.lang == "TR" else "Data manipulation"}</li>
                <li>🔹 <b>Plotly:</b> {"İnteraktif görselleştirme" if st.session_state.lang == "TR" else "Interactive visualization"}</li>
                <li>🔹 <b>Scikit-learn:</b> {"Metrikler ve benzerlik analizleri" if st.session_state.lang == "TR" else "Metrics and similarity analysis"}</li>
                <li>🔹 <b>SentencePiece:</b> {"Tokenizasyon süreçleri" if st.session_state.lang == "TR" else "Tokenization processes"}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # TAKIM
        st.markdown(f"{t('about_devs')}")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%); border-radius: 12px; padding: 1.2rem; border: 1px solid rgba(99, 102, 241, 0.2);">
            <p style="color: #6366f1; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.8rem; text-align: left;">{t('about_team')}</p>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 0.5rem 0;">
            <ul style="list-style-type: none; padding: 0; font-size: 1.1rem; color: #fff; margin: 0;">
                <li style="margin: 0.4rem 0;">👤 EMRE AÇAR</li>
                <li style="margin: 0.4rem 0;">👤 ECEM UZMAN</li>
                <li style="margin: 0.4rem 0;">👤 ELİF CELEP</li>
                <li style="margin: 0.4rem 0;">👤 İBRAHİM AKDAŞ</li>
                <li style="margin: 0.4rem 0;">👤 OĞUZHAN EREZ</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# ANA FONKSİYON
# =========================================================
def main():
    # DİL SEÇİMİ (Language Initialization)
    if 'lang' not in st.session_state:
        st.session_state.lang = "TR"

    # Üst Alan - Logo ve Sekmeler Yan Yana (Sol Üst Hizalama)
    col_logo, col_tabs = st.columns([1, 5], vertical_alignment="top")
    
    with col_logo:
        base_path = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(base_path, "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path)
        else:
            st.markdown(f"<h3>📊 {t('app_title')}</h3>", unsafe_allow_html=True)
        
        # LOGO ALTI BİLGİ KUTUSU
        st.markdown(f"""
        <div id="info-box" style="
            background: rgba(99, 102, 241, 0.05);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 8px;
            padding: 0.8rem;
            margin-top: 0.5rem;
            font-size: 0.85rem;
            color: #cbd5e1;
            line-height: 1.4;
            text-align: center;
            width: 100%;
        ">
            {t('info_box')}
        </div>
        """, unsafe_allow_html=True)
                        
        # DİL SEÇİMİ (SEKME GÖRÜNÜMLÜ BUTONLAR)
        st.markdown("""
        <style>
            /* "Language" yazısını tamamen gizle */
            div[data-testid="stWidgetLabel"] {
                display: none !important;
            }
            
            /* Segmented Control Görünümü */
            .stRadio > div[role="radiogroup"] {
                background-color: rgba(255, 255, 255, 0.03) !important;
                border-radius: 8px !important;
                padding: 2px !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                display: flex !important;
                flex-direction: row !important;
                justify-content: center !important;
                gap: 2px !important;
                width: 100% !important;
                margin-top: -8px !important;
            }
            
            /* Radio dairesini gizle */
            .stRadio div[role="radiogroup"] label div[data-testid="stWidgetLabel"] {
                display: none !important;
            }
            .stRadio div[role="radiogroup"] label > div:first-child {
                display: none !important;
            }
            
            .stRadio label {
                flex: 1 !important;
                padding: 4px 0 !important;
                border-radius: 6px !important;
                transition: all 0.3s !important;
                margin: 0 !important;
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                cursor: pointer !important;
                min-width: 50px !important;
            }
            
            .stRadio label div[data-testid="stMarkdownContainer"] p {
                font-weight: 800 !important;
                font-size: 0.85rem !important;
                margin: 0 !important;
                white-space: nowrap !important; /* E N harflerinin alt alta gelmesini engeller */
                display: block !important;
            }
            
            /* Seçili olmayan dil */
            .stRadio label[data-baseweb="radio"] {
                background-color: transparent !important;
                color: #94a3b8 !important;
            }
            
            /* Seçili olan dil */
            .stRadio label[data-baseweb="radio"]:has(input:checked) {
                background-color: #6366f1 !important;
                color: white !important;
                box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4) !important;
            }
        </style>
        """, unsafe_allow_html=True)

        lang_options = ["TR", "EN"]
        current_idx = lang_options.index(st.session_state.lang)
        
        # Sekme görünümlü radio butonu
        selected_lang = st.radio(
            "",
            options=lang_options,
            index=current_idx,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if selected_lang != st.session_state.lang:
            st.session_state.lang = selected_lang
            st.rerun()

        # İLETİŞİM KUTUSU (SOL ALTTA SABİT - BİLGİ KUTUSU GENİŞLİĞİNDE)
        st.markdown(f"""
        <style>
            /* Bilgi kutusunun genişliğini ve konumunu yakalamak için Streamlit'in kolon yapısını kullanıyoruz */
            [data-testid="column"]:first-child {{
                position: relative;
            }}
            .fixed-contact-box {{
                position: fixed;
                bottom: 20px;
                /* Sol kolonun (col_logo) genişliğini hedefliyoruz */
                width: calc(100% / 6 - 2rem); 
                min-width: 180px;
                max-width: 250px;
                background: rgba(16, 185, 129, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(16, 185, 129, 0.2);
                border-radius: 12px;
                padding: 1rem;
                z-index: 999;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
        </style>
        <div class="fixed-contact-box">
            <strong style="color: #10b981; display: block; margin-bottom: 0.5rem; font-family: 'Poppins', sans-serif; font-size: 0.85rem; text-align: center;">{t('contact_title')}</strong>
            <b style="font-size: 1.1rem; color: #fff; display: block; margin-bottom: 0.3rem; text-align: center;">EMRE AÇAR</b>
            <a href="https://www.linkedin.com/in/emreacarc/" target="_blank" style="color: #6366f1; text-decoration: none; font-size: 0.85rem; display: block; margin-bottom: 0.3rem; text-align: center;">{t('contact_linkedin')}</a>
            <span style="font-size: 0.8rem; color: #cbd5e1; display: block; text-align: center;">ar.emreacar@gmail.com</span>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 0.8rem 0;">
            <span style="font-size: 0.7rem; color: #94a3b8; line-height: 1.2; display: block; text-align: center;">{t('contact_others')}</span>
            </div>
            """, unsafe_allow_html=True)

    with st.spinner(t('loading_models')):
        tokenizer, clf_model, emb_model, device = load_models()
        df = load_data()
    
    with col_tabs:
        tab1, tab2, tab3, tab4 = st.tabs(t('tabs'))
    
    with tab1: show_dashboard(df)
    with tab2: show_complaint_analysis(tokenizer, clf_model, emb_model, device, df)
    with tab3: show_time_series_analysis()
    with tab4: show_about_section()

if __name__ == "__main__":
    main()
