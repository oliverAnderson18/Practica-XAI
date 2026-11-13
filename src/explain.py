import pickle
import shap
import matplotlib.pyplot as plt

from config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from data import preparar_particion


if __name__ == "__main__":
    X_train, X_test, _, _ = preparar_particion()

    # cargamos el RF
    with open(MODELS_DIR / "random_forest.pickle", "rb") as archivo:
        resultado = pickle.load(archivo)

    # extraemos el bosque del pipeline y las variables seleccionadas
    bosque = resultado["pipeline"].steps[-1][1]
    variables = resultado["variables"]

    # usamos 100 reservas como referencia del comportamiento habitual
    referencia = X_train[variables].sample(100, random_state=RANDOM_STATE)

    # se explican 200 reservas de test
    ejemplos = X_test[variables].sample(200, random_state=RANDOM_STATE)

    # creamos el SHAP especifico de modelos basados en arboles
    explicador = shap.TreeExplainer(bosque, data=referencia,
                                    feature_perturbation="interventional",
                                    model_output="probability")

    clase = list(bosque.classes_).index(1)
    valores = explicador(ejemplos)[:, :, clase]

    shap.plots.beeswarm(valores, show=False)
    plt.savefig(REPORTS_DIR / "shap_global.png", dpi=200,
                bbox_inches="tight")
    plt.close()

    shap.plots.waterfall(valores[0], show=False)
    plt.savefig(REPORTS_DIR / "shap_local.png", dpi=200,
                bbox_inches="tight")
    plt.close()

    print("Reserva explicada:", ejemplos.index[0])
    print("Gráficos guardados en:", REPORTS_DIR)
