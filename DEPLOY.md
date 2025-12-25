# 🚀 Deployment Guide / Yayına Alma Rehberi

[English](#english) | [Türkçe](#türkçe)

---

## English

### Step 1: GitHub Repository Setup
1. Create a new public repository on GitHub.
2. Ensure you have Git LFS installed for large files (`.pkl`, `.safetensors`):
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git lfs track "*.safetensors"
   ```
3. Push your code:
   ```bash
   git add .
   git commit -m "Initial commit: CustomerVoice App"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

### Step 2: Streamlit Cloud Deployment
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Connect your GitHub account.
3. Click **"New app"**.
4. Select your repository, branch (`main`).
5. **CRITICAL:** For cloud deployment, select **`streamlit_app_cloud.py`** as the main file path. This version is optimized for the cloud environment and does not require heavy model downloads.
6. Click **"Deploy!"**.

---

## Türkçe

### Adım 1: GitHub Hazırlığı
1. GitHub'da yeni bir public repository oluşturun.
2. Büyük dosyalar için (.pkl, .safetensors) Git LFS'in kurulu olduğundan emin olun:
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git lfs track "*.safetensors"
   ```
3. Kodları gönderin:
   ```bash
   git add .
   git commit -m "Initial commit: CustomerVoice App"
   git branch -M main
   git remote add origin <repo-url-adresiniz>
   git push -u origin main
   ```

### Adım 2: Streamlit Cloud'a Bağlama
1. [share.streamlit.io](https://share.streamlit.io) adresine gidin.
2. GitHub hesabınızı bağlayın.
3. **"New app"** butonuna tıklayın.
4. Repository'nizi, branch'inizi (`main`) seçin.
5. **KRİTİK:** Bulut kurulumu için ana dosya yolu olarak **`streamlit_app_cloud.py`** dosyasını seçin. Bu versiyon bulut ortamı için optimize edilmiştir ve ağır model indirmeleri gerektirmez.
6. **"Deploy!"** butonuna tıklayın.
