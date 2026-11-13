from sklearn.model_selection import train_test_split
import pandas as pd

from config import DATA_PATH, ID_COLUMN, TARGET, RANDOM_STATE, TEST_SIZE


def cargar_datos():
    datos = pd.read_csv(DATA_PATH)
    return datos


def preparar_particion():
    datos = cargar_datos()
    
    # del conjunto de entrenamiento eliminamos el ID y el objetivo
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
    print(pd.DataFrame({
        "train": y_train.value_counts(normalize=True),
        "test": y_test.value_counts(normalize=True)
    }).sort_index())
