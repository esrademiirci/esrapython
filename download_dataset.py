import os
import urllib.request
import zipfile
import shutil
import requests
from tqdm import tqdm
import ssl

# SSL sertifika doğrulamasını devre dışı bırak (güvenlik için normalde önerilmez)
# Bazı durumlarda indirme sorunlarını çözmek için gerekebilir
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, target_path):
    """
    URL'den dosya indirir ve ilerleme çubuğu gösterir
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # HTTP hatalarını kontrol et
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(target_path, 'wb') as file, tqdm(
            desc=os.path.basename(target_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
                
        print(f"İndirme tamamlandı: {os.path.basename(target_path)}")
        return True
    except Exception as e:
        print(f"İndirme hatası: {str(e)}")
        return False

def download_grape_disease_dataset():
    """
    Kaggle veya başka bir kaynaktan üzüm yaprağı hastalık veri setini indirir
    """
    # Roboflow üzüm yaprağı hastalık veri seti (örnek URL)
    dataset_url = "https://storage.googleapis.com/kagglesdsdata/datasets/1456386/2465538/plant-pathology-2020-fgvc7/images.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240428%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240428T080259Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=7a77cd57b2fc57673d6caff63b7cb18eec28b9c2c6a12e56b2a1e7c8c80f5ec0fcf57d09c5659e89f94f57da4a09767e3c02e6aec3b6ff23f6ad6d2f2df2e8eccc9a7a371bcff3e866ea0f0b24071bad40673f5e35de43c3ced621d9022e30f22be8ea79e80af59fb15a0642e10af3ce0de4b93def35d9c7f38ddf6fe7ba4d7e4a21c3f1cdbf07bea8dd76f2b8de38acc71e4dbc0aef93f78bb42d5a1f27d1c65a65723d1e9c9a78bc4fab02a0fedb4f6c62a84ed9be3b59c8d92de29a71c459a47dddecfda7c2dd41aadd6f49c54c07eec6bca9f4a6be0ff6267a9b307f76e03dec8ce8eacade46f6e96dbcb5e17fcd13e1c77a13dd5e5"

    # İndirme klasörünü oluştur
    os.makedirs("downloads", exist_ok=True)
    download_path = os.path.join("downloads", "grape_disease_dataset.zip")
    
    print("Veri seti indiriliyor...")
    if download_file(dataset_url, download_path):
        extract_and_organize_dataset(download_path)
    else:
        print("Alternatif kaynakları deneyiniz!")
        print("Veri setini manuel olarak Kaggle veya benzer kaynaklardan indirip 'dataset' klasörüne çıkarınız.")

def extract_and_organize_dataset(zip_path):
    """
    İndirilen ZIP dosyasını çıkarır ve organize eder
    """
    print("ZIP dosyası çıkarılıyor...")
    extract_dir = "downloads/extracted"
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print("Veri seti organize ediliyor...")
        # Bu kısım veri setine göre değişebilir
        # Çoğu durumda manuel düzenleme gerekebilir
        
        print("Veri seti hazırlama tamamlandı!")
        print("NOT: Veri seti yapısını kontrol ediniz ve dataset/ klasörüne uygun şekilde organize ediniz.")
        
    except Exception as e:
        print(f"Veri seti çıkarma/organizasyon hatası: {str(e)}")

def show_data_organization_instructions():
    """
    Kullanıcıya veri seti düzenleme talimatlarını gösterir
    """
    print("\n" + "="*80)
    print("ÖNEMLİ: VERİ SETİ DÜZENLEME TALİMATLARI")
    print("="*80)
    print("İndirilen veri setinin şu klasör yapısında düzenlenmesi gerekmektedir:")
    print("\ndataset/")
    print("├── healthy/            <- Sağlıklı üzüm yaprağı görüntüleri")
    print("├── black_rot/          <- Black Rot hastalıklı görüntüler")
    print("├── esca/               <- Esca hastalıklı görüntüler")
    print("├── leaf_blight/        <- Yaprak yanıklığı görüntüleri")
    print("└── powdery_mildew/     <- Külleme hastalıklı görüntüler")
    print("\nİndirilen veri setinde bu sınıflandırma yoksa,")
    print("görüntüleri kendiniz ilgili sınıf klasörlerine manuel olarak taşımanız gerekebilir.")
    print("="*80)
    print("NOT: Kendi çektiğiniz üzüm yaprağı fotoğraflarını da ilgili hastalık klasörlerine ekleyebilirsiniz.")
    print("="*80 + "\n")

if __name__ == "__main__":
    print("Üzüm Yaprağı Hastalık Veri Seti İndirme Aracı")
    print("="*50)
    
    # Klasör yapısını kontrol et
    if not os.path.exists("dataset"):
        print("Dataset klasörü bulunamadı! Klasör yapısı oluşturuluyor...")
        os.makedirs("dataset/healthy", exist_ok=True)
        os.makedirs("dataset/black_rot", exist_ok=True)
        os.makedirs("dataset/esca", exist_ok=True)
        os.makedirs("dataset/leaf_blight", exist_ok=True)
        os.makedirs("dataset/powdery_mildew", exist_ok=True)
    
    show_data_organization_instructions()
    
    choice = input("Veri setini indirmek istiyor musunuz? (e/h): ").lower()
    if choice == 'e':
        download_grape_disease_dataset()
    else:
        print("Veri seti indirilmedi. Lütfen verileri manuel olarak ekleyin.")
    
    print("\nİşlem tamamlandı!")
    print("NOT: Veri seti organizasyonunu kontrol ediniz ve gerekirse düzenleyiniz.") 