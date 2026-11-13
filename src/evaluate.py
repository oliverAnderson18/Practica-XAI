import pickle

from sklearn.metrics import classification_report, confusion_matrix

from config import MODELS_DIR
from data import preparar_particion


if __name__ == "__main__":
    _, X_test, _, y_test = preparar_particion()

    for nombre in ["logistica", "random_forest"]:
        with open(MODELS_DIR / f"{nombre}.pickle", "rb") as archivo:
            resultado = pickle.load(archivo)

        modelo = resultado["pipeline"]
        variables = resultado["variables"]
        
        probabilidades = modelo.predict_proba(X_test[variables])[:, 1]
        predicciones = (probabilidades >= resultado["umbral"]).astype(int)
 
        print(f"\n--- {nombre.upper()} | umbral {resultado['umbral']} ---")

        # muestra TN, FP, FN, TP
        print(confusion_matrix(y_test, predicciones, labels=[0, 1]))

        # muestra precision, recall, F1 y support
        print(classification_report(
            y_test,
            predicciones,
            labels=[0, 1],
            target_names=["Se presenta", "No se presenta"],
            digits=4,
            zero_division=0))
