import joblib
import pandas as pd


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


model = joblib.load(
    "model/model_kategorije.pkl"
)

print(
    "Model je uspešno naložen."
)

print(
    "Za izhod napiši 'izhod'."
)

while True:

    naslov = input(
        "\nVnesi naziv izdelka: "
    )

    if naslov.strip().lower() in [
        "izhod",
        "exit"
    ]:
        print(
            "Program se zaključuje."
        )
        break

    if not naslov.strip():
        print(
            "Naslov ne sme biti prazen."
        )
        continue

    vhod = pd.DataFrame({
        "product_title": [
            naslov
        ]
    })

    vhod = dodaj_znacilke(
        vhod
    )

    napoved = model.predict(
        vhod
    )[0]

    print(
        f"Predvidena kategorija: {napoved}"
    )
