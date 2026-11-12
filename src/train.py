import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from config import MODELS_DIR, RANDOM_STATE
from data import preparar_particion
from feature_selection import crear_modelo, seleccionar_variables


def elegir_umbral(modelo, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    probabilidades = cross_val_predict(
        modelo, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    resultados = []

    for umbral in [0.3, 0.5, 0.7]:
        pred = (probabilidades >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

        resultados.append({
            "umbral": umbral,
            "coste": 70 * fn + 3 * fp
        })

    tabla = pd.DataFrame(resultados)
    print(tabla.to_string(index=False))

    return tabla.loc[tabla["coste"].idxmin(), "umbral"]


if __name__ == "__main__":
    X_train, _, y_train, _ = preparar_particion()
    variables = seleccionar_variables(X_train, y_train)
    X_train = X_train[variables]

    modelos = {
        "logistica": crear_modelo(),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    }

    MODELS_DIR.mkdir(exist_ok=True)

    for nombre, modelo in modelos.items():
        umbral = elegir_umbral(modelo, X_train, y_train)
        modelo.fit(X_train, y_train)

        with open(MODELS_DIR / f"{nombre}.pickle", "wb") as archivo:
            pickle.dump({
                "pipeline": modelo,
                "variables": variables,
                "umbral": umbral
            }, archivo)

        print(f"{nombre}: umbral elegido = {umbral}")

    logistica = modelos["logistica"]
    print("\nCoeficientes de la regresión:")
    print(pd.DataFrame({
        "variable": variables,
        "coeficiente": logistica.named_steps["logisticregression"].coef_[0]
    }).round(4).to_string(index=False))