import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, ID_COLUMN, TARGET, RANDOM_STATE, TEST_SIZE


def cargar_datos():
    datos = pd.read_csv(DATA_PATH)

    if not {ID_COLUMN, TARGET}.issubset(datos.columns):
        raise ValueError("Falta el identificador o la columna objetivo.")

    if datos.isna().any().any():
        raise ValueError("Hay valores ausentes.")

    if not datos[TARGET].isin([0, 1]).all():
        raise ValueError("El objetivo debe contener solo 0 y 1.")

    return datos


def preparar_particion():
    datos = cargar_datos()
    X = datos.drop(columns=[ID_COLUMN, TARGET])
    y = datos[TARGET]

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preparar_particion()

    print(f"Entrenamiento: {len(X_train)} | Test: {len(X_test)}")

    print("\nProporciones por clase:")
    print(pd.DataFrame({
        "train": y_train.value_counts(normalize=True),
        "test": y_test.value_counts(normalize=True)
    }).sort_index())