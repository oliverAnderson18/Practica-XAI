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

    # obtiene las probabilidades OOF
    probabilidades = cross_val_predict(modelo, X, y, cv=cv,
                                       method="predict_proba")[:, 1]
    resultados = {}

    # comparamos con los umbrales de la anterior practica
    for umbral in (0.3, 0.5, 0.7):
        prediccion = (probabilidades >= umbral).astype(int)

        # solo usamos FP, FN porque son los errores de coste
        _, fp, fn, _ = confusion_matrix(y, prediccion).ravel()
        resultados[umbral] = 70 * fn + 3 * fp

    print(pd.Series(resultados, name="coste"))
    return min(resultados, key=resultados.get)


if __name__ == "__main__":
    X_train, _, y_train, _ = preparar_particion()

    # FFS selecciona las variables que se usaran
    variables = seleccionar_variables(X_train, y_train)
    X_train = X_train[variables]

    # modelo interpretable y mas complejo para explicar con SHAP
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

    for nombre, modelo in modelos.items():

        # elegimos el umbral antes de entrenar el modelo final
        umbral = elegir_umbral(modelo, X_train, y_train)
        modelo.fit(X_train, y_train)

        ruta = MODELS_DIR / f"{nombre}.pickle"
        with open(ruta, "wb") as archivo:
            pickle.dump({
                "pipeline": modelo,
                "variables": variables,
                "umbral": umbral
            }, archivo)

        print(f"{nombre}: umbral {umbral} | guardado en {ruta}")

    logistica = modelos["logistica"]
    print(pd.DataFrame({
        "variable": variables,
        "coeficiente": logistica.named_steps["logisticregression"].coef_[0]
    }).round(4).to_string(index=False))
