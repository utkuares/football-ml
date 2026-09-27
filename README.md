# ⚽ Süper Lig Maç Sonucu Tahmin Sistemi (Football ML)

Bu proje, Türkiye Süper Lig geçmiş maç verilerini kullanarak maç sonuçlarını (**H: Ev Sahibi Galibiyeti**, **D: Beraberlik**, **A: Deplasman Galibiyeti**) makine öğrenmesi modelleri ile tahmin eden bir veri bilimi ve makine öğrenmesi projesidir.

---

## 📌 1. Projenin Amacı

Futbol maçlarının sonuçlarını takımların geçmiş performansları, son 5 maçlık form durumları, attıkları/yedikleri gol sayıları ve ev sahibi/deplasman avantajlarını analiz ederek tahmin etmek.

---

## 📊 2. Kullanılan Veri Seti (`tsl_dataset.csv`)

Veri seti Türkiye Süper Lig tarihindeki **17.898** maça ait bilgileri içermektedir. Veri setinde yer alan temel bilgiler:
- **Tarih, Sezon, Hafta**
- **Ev Sahibi ve Deplasman Takımları**
- **Atılan Gol Sayıları (İlk Yarı ve Maç Sonu)**
- **Kırmızı Kart Sayıları, Seyirci/Nötr Saha Bilgisi**

---

## 🛠️ 3. Oluşturulan Özellikler (Feature Engineering)

Data leakage (veri sızıntısı) önlemek amacıyla özellikler yalnızca maç öncesi verilere dayanarak türetilmiştir:

1. **Önceki Maç Sayısı (`home_previous_matches`, `away_previous_matches`)**: Takımların ligdeki tecrübesi.
2. **Son 5 Maç Galibiyet Sayısı (`home_last5_wins`, `away_last5_wins`)**: Takımların son 5 maçtaki galibiyet sayısı.
3. **Son 5 Maç Puanı (`home_last5_points`, `away_last5_points`)**: Takımların son 5 maçta topladığı puan.
4. **Son 5 Maç Attığı ve Yediği Goller (`home_last5_goals_for`, `home_last5_goals_against`, `away_last5_goals_for`, `away_last5_goals_against`)**: Takımların hücum ve savunma formları.
5. **One-Hot Encoding**: Takım isimleri kategorik değişkenden sayısal vektörlere dönüştürülmüştür.

---

## 🤖 4. Kullanılan Modeller ve Sonuçlar

Tüm modeller aynı eğitim/test ayrımı (%80 Train, %20 Test) üzerinde test edilmiştir. Logistic Regression modelinde veri sızıntısını önlemek için `StandardScaler` yalnızca eğitim verisine fit edilip test verisine transform uygulanmıştır.

| Model | Accuracy (%) | Açıklama |
| :--- | :---: | :--- |
| **Gradient Boosting** | **%53.07** | En yüksek doğruluğu veren model. |
| **Logistic Regression** | **%52.85** | Standardize edilmiş verilerle hızlı ve stabil performans. |
| **Random Forest** | **%48.55** | Dengelenmiş sınıf ağırlıklarıyla temel karar ağacı topluluğu. |

---

## 📈 5. Görselleştirmeler

Proje çalıştırıldığında aşağıdaki grafikler otomatik olarak üretilip proje dizinine kaydedilir:

1. `match_distribution.png`: Süper Lig maç sonuçlarının dağılımı (Ev sahibi galibiyeti baskınlığı).
2. `model_comparison.png`: Modellerin başarım (Accuracy) oranlarının karşılaştırması.
3. `feature_importance.png`: Karar ağacı modellerine göre en belirleyici 15 özellik.

---

## 🔮 6. Örnek Tahmin (Galatasaray vs Fenerbahçe)

Sistem, belirli bir tarihteki takımların geçmiş performanslarını çekerek yeni maçlar için tahmin üretebilir:

* **Örnek Maç:** Galatasaray (Ev Sahibi) vs Fenerbahçe (Deplasman) [01.05.2019]
* **Geçmiş Form:**
  * Galatasaray: 3 Galibiyet, 11 Puan (Attığı: 10, Yediği: 4)
  * Fenerbahçe: 1 Galibiyet, 6 Puan (Attığı: 5, Yediği: 5)
* **Model Tahminleri:**
  * **Random Forest:** `H` (Ev sahibi kazanır)
  * **Logistic Regression:** `H` (Ev sahibi kazanır)
  * **Gradient Boosting:** `H` (Ev sahibi kazanır)

---

## 🚀 7. Projenin Çalıştırılması

Gerekli kütüphaneleri yükleyin ve projeyi çalıştırın:

```bash
# Kütüphanelerin yüklenmesi
pip install pandas numpy scikit-learn matplotlib seaborn

# Projenin çalıştırılması
python3 main.py
```

---

## 📦 8. Kullanılan Kütüphaneler

* **Pandas & NumPy:** Veri işleme ve zaman serisi düzenlemeleri.
* **Scikit-Learn:** Veri ölçekleme (`StandardScaler`), model eğitimi, değerlendirme metrikleri ve train/test ayrımı.
* **Matplotlib & Seaborn:** Veri ve model performans görselleri.
