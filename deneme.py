import pandas as pd
import os
import seaborn as sns
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, explained_variance_score
from time import time

# Kullanıcının masaüstü dizinini alma
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "insurance_modified.csv")

# CSV dosyasını oku
if os.path.exists(file_path):
    data = pd.read_csv(file_path, delimiter=";")
    data = data.dropna()

    print(data.info())
    print(data.head(5))
    print('-' * 90)
    print(f"Veri başarıyla yüklendi. Veri kümesi {data.shape[0]} satır ve {data.shape[1]} sütun içeriyor.")

    # Kozmetik Durum
    def ekd_category(kozmetik_durum):
        if kozmetik_durum < 19.9:
            return 'İyi'
        elif 19.9 <= kozmetik_durum <= 30:
            return 'Normal'
        else:
            return 'Kötü'

    # Evin Yaşı
    def age_category(evin_yasi):
        age_dict = {
            0: '0-4', 1: '5-9', 2: '10-14', 3: '15-19', 4: '20-24',
            5: '25-29', 6: '30-34', 7: '35-39', 8: '40+'
        }
        index = evin_yasi // 5
        index = min(index, 8)
        return age_dict[index]

    # Çocuk Sayısı
    def cocuk_category(cocuk_sayisi):
        if cocuk_sayisi == 0:
            return 'Yok'
        elif cocuk_sayisi == 1:
            return 'Az'
        elif cocuk_sayisi == 2:
            return 'Normal'
        elif cocuk_sayisi == 3:
            return 'Fazla'
        else:
            return 'Çok Fazla'

    data['Katagorik_Evin_Kozmetik_Durumu'] = data['Evin Kozmetik Durumu'].apply(ekd_category)
    data['Katagorik_Evin_Yasi'] = data['Evin Yasi'].apply(age_category)
    data['Katagorik_Cocuk_Sayisi'] = data['Cocuk Sayisi'].apply(cocuk_category)

    kategorik_sütunlar = ['Katagorik_Evin_Kozmetik_Durumu', 'Ev Durumu', 'Evcil Hayvan Sahibi',
                          'Bolge', 'Katagorik_Evin_Yasi', 'Katagorik_Cocuk_Sayisi']

    # Grafikler
    for column in kategorik_sütunlar:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=column, y='Masraf', data=data)
        plt.title(f"{column} - Masraf")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    sns.pairplot(data, height=2)
    plt.show()

    for v in kategorik_sütunlar:
        data[v].value_counts().plot(kind='bar')
        plt.xticks(rotation=0, fontsize=10)
        plt.title(v)
        plt.tight_layout()
        plt.show()

    for v in kategorik_sütunlar:
        group_df = data.groupby(v)['Masraf'].mean()
        group_df.sort_values().plot(kind='bar')
        plt.xlabel(v)
        plt.ylabel('Ortalama Masraf')
        plt.title(f'Ortalama Sigorta Masrafı - {v}')
        plt.tight_layout()
        plt.show()

# Hedef ve özelliklerin ayrılması
target = data['Masraf']
features = data.drop(['Evin Yasi', 'Evin Kozmetik Durumu', 'Cocuk Sayisi', 'Masraf'], axis=1)

# İstatistiksel özetleme
print(target.describe())
print('-' * 90)
print("Statistics for Medical Insurance dataset:\n")
print("Minimum insurance cost: ${:,.2f}".format(np.min(target)))
print("Maximum insurance cost: ${:,.2f}".format(np.max(target)))
print("Mean insurance cost: ${:,.2f}".format(np.mean(target)))
print("Median insurance cost: ${:,.2f}".format(np.median(target)))
print("Standard deviation of insurance costs: ${:,.2f}".format(np.std(target)))

# Kategorik verileri sayısal verilere dönüştürme
output = pd.DataFrame(index=features.index)
for col, col_data in features.items():
    if col_data.dtype == object:
        col_data = col_data.replace(['yes', 'no'], [1, 0])
        col_data = col_data.infer_objects(copy=False)
    if col_data.dtype == object:
        col_data = pd.get_dummies(col_data, prefix=col)
    output = output.join(col_data)

features = output
print(f"Processed feature columns ({len(features.columns)} total features):\n{list(features.columns)}")

# Eğitim ve test verisini ayırma
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=0)
print("Training and testing split was successful.")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Model eğitimi
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)

# Eğitim ve tahmin fonksiyonu
def train_predict_model(clf, X_train, y_train, X_test, y_test):
    print("Training a {} using a training set size of {}...".format(clf.__class__.__name__, len(X_train)))
    start = time()
    clf.fit(X_train, y_train)
    print("Trained model in {:.4f} seconds".format(time() - start))
    print('#' * 50)

    start = time()
    y_pred_train = clf.predict(X_train)
    print("Made predictions for training data in {:.4f} seconds.".format(time() - start))
    print("R^2 score for training set: {:.4f}".format(r2_score(y_train, y_pred_train)))
    print("Explained variance score for training set: {:.4f}".format(explained_variance_score(y_train, y_pred_train)))
    print('#' * 50)

    start = time()
    y_pred_test = clf.predict(X_test)
    print("Made predictions for testing data in {:.4f} seconds.".format(time() - start))
    print("R^2 score for testing set: {:.4f}".format(r2_score(y_test, y_pred_test)))
    print("Explained variance score for testing set: {:.4f}".format(explained_variance_score(y_test, y_pred_test)))
    print('#' * 50)

# Farklı modellerin eğitimi
clf_a = DecisionTreeRegressor(random_state=0)
clf_b = SVR()
clf_c = KNeighborsRegressor()
clf_d = NuSVR()

for clf in (clf_a, clf_b, clf_c, clf_d):
    for size in (300, 600, 900):
        train_predict_model(clf, X_train[:size], y_train[:size], X_test, y_test)
        print('-' * 80)
    print('+' * 80)

# Modeli kaydet
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("model_columns.pkl", "wb") as cols_file:
    pickle.dump(features.columns.tolist(), cols_file)

# Örnek müşteri verisiyle tahmin
client_data = [
    [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
]

client_df = pd.DataFrame(client_data, columns=features.columns)

# Eksik sütun varsa tamamla
missing_cols = set(features.columns) - set(client_df.columns)
for col in missing_cols:
    client_df[col] = 0
client_df = client_df[features.columns]

predictions = model.predict(client_df)
print("Predictions for client data:", predictions)
