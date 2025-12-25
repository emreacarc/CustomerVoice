@echo off
echo GitHub Deposu Tamir Ediliyor ve Yukleniyor...
echo.

G:
cd /d "G:\My Drive\DataScienceProjects\customer_complaint"

:: Git ayarlarini temizle ve yeniden baslat
echo 1. Git deposu yeniden iliskilendiriliyor...
git init

:: Remote adresini tekrar ekle (hata verirse onemli degil, devam eder)
git remote add origin https://github.com/emreacarc/CustomerVoice.git 2>nul
git remote set-url origin https://github.com/emreacarc/CustomerVoice.git

echo 2. Dosyalar taraniyor...
git add README.md DEPLOY.md streamlit_app_cloud.py streamlit_app_v3.py requirements.txt .gitignore logo.png

echo 3. Kayit yapiliyor (Commit)...
git commit -m "Uygulama Yerel ve Bulut olarak ikiye ayrildi, dokumantasyon guncellendi"

echo 4. GitHub'a yukleniyor (Zorlamali Push)...
:: Not: Bu islem mevcut dosyalarinizi GitHub'dakiyle senkronize eder.
git push -u origin main -f

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Main denemesi basarisiz, Master deneniyor...
    git push -u origin master -f
)

echo.
echo Islem tamamlandi! Eger 'Permission Denied' alirsaniz klasoru C: surucusune kopyalayip denemeniz gerekecektir.
pause
