import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np



df = pd.read_excel('Test/data/train.xlsx')

df['datetime'] = pd.to_datetime(df['year'].astype(str) + '-' + 
                                df['month'].astype(str) + '-' + 
                                df['day'].astype(str) + ' ' + 
                                df['time'].astype(str) + ":00")

df['hour'] = df['datetime'].dt.hour   # Saati ayırıyoruz
df['month'] = df['datetime'].dt.month # Ayı ayırıyoruz



# 4. MODELİN GİRDİ VE ÇIKTILARINI AYARLIYORUZ

ss=df["wind_speed_50m:ms"].rolling(window=3).std()#standart sapma(türbilans yoğunluğunda kullanmak için(ss/mv))
mv=df["wind_speed_50m:ms"].rolling(window=3).mean()#ortalama hız değeri (mv)
df["turbilans_yogunlugu"] = ss / (mv +0.00001)


target = 'wind_production'
df["wind_speed_cubed"] = df["wind_speed_50m:ms"] ** 3  # Rüzgar hızı küpü
df["pre_predict"] = df["wind_production"].shift(1)  # Bir önceki saat rüzgar enerjisi üretimi
df["rolling_wind"] = df["wind_speed_50m:ms"].rolling(window=3).mean()  # 3 saatlik hareketli ortalama
df["temp_kelvin"] = df["t_50m:C"] + 273
df["1/kelvin"] = 1/(df["t_50m:C"] +273)
df['wind_dir_sin'] = np.sin(np.radians(df['wind_dir_50m:d'])) # 360 ile 1 derecenin birbirine yakın olduğunu öğretme
df['wind_dir_cos'] = np.cos(np.radians(df['wind_dir_50m:d']))
df['season'] = (df['month'] % 12 + 3) // 3 # Mevsimi belirleme

# Mantık: Hızın 3.5 ile 25.5 arasında olduğu satırları al, gerisini çöpe at.
df = df[(df['wind_speed_50m:ms'] >= 3.5) & (df['wind_speed_50m:ms'] <= 25.5)]

df = df.dropna() # boş kalan satırları temizler
df = df.reset_index(drop=True) # İndeksleri Sıfırla (Satır sildiğimiz için numaralar kaymasın diye 1, 5, 8... gitmesin)
# Remove extreme outliers in production data
Q1 = df['wind_production'].quantile(0.01)
Q3 = df['wind_production'].quantile(0.99)
df['wind_production'] = df['wind_production'].clip(Q1,Q3)  # Hatalı verileri temizliyor

# Kullanacağımız Veriler (Sorular)
features = [  't_50m:C', "wind_dir_sin","wind_dir_cos", "wind_speed_cubed", "pre_predict", 
            "rolling_wind", "temp_kelvin","total_cloud_cover:p",
            "effective_cloud_cover:p",
            "global_rad_1h:Wh","wind_dir_50m:d", "season","hour","month","wind_speed_50m:ms"]



X = df[features]  
y = df[target]    

# Veriyi %80 Eğitim, %20 Test olarak bölüyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)



# Adım 1: Modeli Oluştur (100 tane ağaç kullan diyoruz)
rf_model = RandomForestRegressor(n_estimators=1500,
                                  random_state=42,
                                  max_depth=25,
                                  min_samples_split=5,
                                  min_samples_leaf=10)

print("Yapay Zeka eğitiliyor... (Biraz sürebilir)")

# Adım 2: Modeli Eğit (Ders çalıştır)
rf_model.fit(X_train, y_train) 

print("Eğitim tamamlandı!")

onem_dereceleri = rf_model.feature_importances_

# 2. Bunları güzel bir tabloya döküyoruz
feature_importance_df = pd.DataFrame({
    'Faktor': features,  
    'Onem_Puani': onem_dereceleri
})

# 3. Puana göre sıralıyoruz (En önemli en üstte)
feature_importance_df = feature_importance_df.sort_values(by='Onem_Puani', ascending=False)

# 4. Ekrana Yazdırıyoruz
print(feature_importance_df)
print("=============================")

# Adım 3: Tahmin Yap
tahminler = rf_model.predict(X_test)

# ---------------------------------------------------------
basari_orani = r2_score(y_test, tahminler)

# MAE (Ortalama Hata - MW cinsinden)
mae = mean_absolute_error(y_test, tahminler)

# --- DOĞRULUK YÜZDESİ HESABI ---
# 1. Gerçek verilerin ortalaması (Santral ortalama ne üretiyor?)
gercek_ortalama = np.mean(y_test)

# 2. Hata Yüzdesi (Hata / Ortalama * 100)
yuzde_hata = (mae / gercek_ortalama) * 100

# 3. Doğruluk Yüzdesi (100 - Hata Yüzdesi)
dogruluk_yuzdesi = 100 - yuzde_hata

# 5. BAŞARIYI ÖLÇÜYORUZ
basari_orani = r2_score(y_test, tahminler)
hata_miktari = mean_absolute_error(y_test, tahminler)

print(f"Model Başarısı (R2): {basari_orani:.2f}") # 1'e ne kadar yakınsa o kadar iyi
print(f"Ortalama Hata (MAE): {hata_miktari:.2f}")
print(f"MODEL DOĞRULUK ORANI  : %{dogruluk_yuzdesi:.2f}")

# 6. GRAFİKLE GÖSTERİYORUZ
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[:100], label='Gerçek Üretim', color='blue')
plt.plot(tahminler[:100], label='Random Forest Tahmini', color='red', linestyle='--')
plt.title('Random Forest ile Rüzgar Enerjisi Tahmini')
plt.legend()
plt.show()