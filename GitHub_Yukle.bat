@echo off
echo GitHub Deposu Tamir Ediliyor ve Yukleniyor...
echo.

:: Bulunulan dizine git
cd /d "%~dp0"

:: Git ayarlarini temizle ve yeniden baslat
echo 1. Git deposu yeniden iliskilendiriliyor...
git init

:: Remote adresini tekrar ekle
git remote add origin https://github.com/emreacarc/CustomerVoice.git 2>nul
git remote set-url origin https://github.com/emreacarc/CustomerVoice.git

:: LFS kurulumunu kontrol et ve aktif et
echo 2. Git LFS ayarları yapılıyor...
git lfs install
git lfs track "*.safetensors"
git lfs track "*.pkl"
git lfs track "bert_based_classification_models/*"

echo 3. Dosyalar ekleniyor...
:: Tum dosyalari ekle (.gitignore'dakiler haric)
git add .

echo 4. Kayit yapiliyor (Commit)...
git commit -m "Cloud deployment fix: including models and data files"

echo 5. GitHub'a yukleniyor (Zorlamali Push)...
:: Not: Bu islem mevcut dosyalarinizi GitHub'dakiyle senkronize eder.
git push -u origin main -f

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Main denemesi basarisiz, Master deneniyor...
    git push -u origin master -f
)

echo.
echo Islem tamamlandi!
pause
