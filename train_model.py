import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone


def dodaj_znacilke(tabela):
    tabela = tabela.copy()

    tabela["dolzina_naslova"] = (
        tabela["product_title"]
        .astype(str)
        .str.len()
    )

    tabela["stevilo_besed"] = (
        tabela["product_title"]
        .astype(str)
        .str.split()
        .str.len()
    )

    tabela["vsebuje_stevilko"] = (
        tabela["product_title"]
        .astype(str)
        .str.contains(r"\d", regex=True)
        .astype(int)
    )

    tabela["stevilo_posebnih_znakov"] = (
        tabela["product_title"]
        .astype(str)
        .apply(
            lambda naslov: sum(
                1
                for znak in naslov
                if not znak.isalnum()
                and not znak.isspace()
            )
        )
    )

    tabela["dolzina_najdaljse_besede"] = (
        tabela["product_title"]
        .astype(str)
        .apply(
            lambda naslov: max(
                (
                    len(beseda)
                    for beseda in naslov.split()
                ),
                default=0
            )
        )
    )

    tabela["stevilo_velikih_besed"] = (
        tabela["product_title"]
        .astype(str)
        .apply(
            lambda naslov: sum(
                1
                for beseda in naslov.split()
                if len(beseda) > 1
                and beseda.isupper()
            )
        )
    )

    return tabela


print("Nalaganje podatkov...")

podatki = pd.read_csv(
    "data/products.csv"
)

podatki.columns = (
    podatki.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

podatki = podatki.dropna(
    subset=[
        "product_title",
        "category_label"
    ]
).copy()

podatki["product_title"] = (
    podatki["product_title"]
    .astype(str)
    .str.strip()
)

podatki["category_label"] = (
    podatki["category_label"]
    .astype(str)
    .str.strip()
)

podatki = podatki[
    (podatki["product_title"] != "")
    &
    (podatki["category_label"] != "")
].copy()

podatki = podatki.drop_duplicates(
    subset=[
        "product_title",
        "category_label"
    ]
).copy()

# Odstranimo kategorije z manj kot 2 primeroma.
stevilo_po_kategoriji = (
    podatki["category_label"]
    .value_counts()
)

redke_kategorije = (
    stevilo_po_kategoriji[
        stevilo_po_kategoriji < 2
    ].index
)

podatki = podatki[
    ~podatki["category_label"].isin(
        redke_kategorije
    )
].copy()

podatki = dodaj_znacilke(
    podatki
)

stevilcne_znacilke = [
    "dolzina_naslova",
    "stevilo_besed",
    "vsebuje_stevilko",
    "stevilo_posebnih_znakov",
    "dolzina_najdaljse_besede",
    "stevilo_velikih_besed"
]

vhodni_stolpci = [
    "product_title"
] + stevilcne_znacilke

vhodni_podatki = podatki[
    vhodni_stolpci
]

ciljna_kategorija = podatki[
    "category_label"
]

(
    vhod_trening,
    vhod_test,
    kategorija_trening,
    kategorija_test
) = train_test_split(
    vhodni_podatki,
    ciljna_kategorija,
    test_size=0.20,
    random_state=42,
    stratify=ciljna_kategorija
)

predobdelava = ColumnTransformer(
    transformers=[
        (
            "naslov",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=30000
            ),
            "product_title"
        ),
        (
            "stevilcne_znacilke",
            MinMaxScaler(),
            stevilcne_znacilke
        )
    ]
)

modeli = {
    "Logistična regresija": LogisticRegression(
        max_iter=1000
    ),
    "Naivni Bayes": MultinomialNB(),
    "Linearni SVM": LinearSVC()
}

najboljsi_f1 = -1
najboljse_ime = None
najboljsi_algoritem = None

for ime_modela, algoritem in modeli.items():

    print(
        f"\nTreniranje: {ime_modela}"
    )

    cevovod = Pipeline([
        (
            "predobdelava",
            clone(predobdelava)
        ),
        (
            "klasifikator",
            clone(algoritem)
        )
    ])

    cevovod.fit(
        vhod_trening,
        kategorija_trening
    )

    napovedi = cevovod.predict(
        vhod_test
    )

    tocnost = accuracy_score(
        kategorija_test,
        napovedi
    )

    f1_macro = f1_score(
        kategorija_test,
        napovedi,
        average="macro"
    )

    print(
        f"Točnost: {tocnost:.4f}"
    )

    print(
        f"F1 macro: {f1_macro:.4f}"
    )

    if f1_macro > najboljsi_f1:
        najboljsi_f1 = f1_macro
        najboljse_ime = ime_modela
        najboljsi_algoritem = clone(
            algoritem
        )


print(
    "\nNajboljši model:",
    najboljse_ime
)

print(
    "Najboljši F1 macro:",
    round(najboljsi_f1, 4)
)

# Ponovno treniranje na vseh podatkih.
finalni_model = Pipeline([
    (
        "predobdelava",
        clone(predobdelava)
    ),
    (
        "klasifikator",
        najboljsi_algoritem
    )
])

finalni_model.fit(
    vhodni_podatki,
    ciljna_kategorija
)

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    finalni_model,
    "model/model_kategorije.pkl"
)

print(
    "\nModel uspešno shranjen v:"
)

print(
    "model/model_kategorije.pkl"
)
