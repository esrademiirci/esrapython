import os
import cv2
import argparse
from tqdm import tqdm

def resize_images(input_folder, output_folder=None, size=(224, 224)):
    """
    Belirtilen klasördeki tüm görüntüleri yeniden boyutlandırır
    
    Args:
        input_folder: Görüntülerin bulunduğu kaynak klasör
        output_folder: Yeniden boyutlandırılmış görüntülerin kaydedileceği klasör (None ise aynı klasöre kaydedilir)
        size: Hedef boyut (genişlik, yükseklik)
    """
    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    # Klasördeki görüntü dosyalarını bul
    image_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_files.append(os.path.join(root, file))
    
    print(f"Toplam {len(image_files)} görüntü dosyası bulundu.")
    
    # Her bir görüntüyü yeniden boyutlandır
    for img_path in tqdm(image_files, desc="Görüntüler yeniden boyutlandırılıyor"):
        try:
            # Görüntüyü oku
            img = cv2.imread(img_path)
            if img is None:
                print(f"Hata: {img_path} dosyası okunamadı.")
                continue
            
            # Yeniden boyutlandır
            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            
            # Çıktı yolunu belirleme
            rel_path = os.path.relpath(img_path, input_folder)
            output_path = os.path.join(output_folder, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Yeniden boyutlandırılmış görüntüyü kaydet
            cv2.imwrite(output_path, resized_img)
            
        except Exception as e:
            print(f"Hata: {img_path} dosyası işlenirken hata oluştu: {str(e)}")
    
    print("İşlem tamamlandı!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Görüntüleri yeniden boyutlandır")
    parser.add_argument("--input", type=str, default="dataset", help="Kaynak klasör yolu")
    parser.add_argument("--output", type=str, default=None, help="Hedef klasör yolu (varsayılan: aynı klasör)")
    parser.add_argument("--width", type=int, default=224, help="Hedef genişlik")
    parser.add_argument("--height", type=int, default=224, help="Hedef yükseklik")
    
    args = parser.parse_args()
    
    resize_images(args.input, args.output, size=(args.width, args.height)) 