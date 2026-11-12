import pickle
import shap
import matplotlib.pyplot as plt

from config import MODELS_DIR, REPORTS_DIR, RANDOM_STATE
from data import preparar_particion


def guardar_figura(nombre):
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / nombre, dpi=200)
    plt.close()


if __name__ == "__main__":
    X_train, X_test, _, _ = preparar_particion()

    with open(MODELS_DIR / "random_forest.pickle", "rb") as archivo:
        resultado = pickle.load(archivo)

    bosque = resultado["pipeline"].steps[-1][1]
    variables = resultado["variables"]
    referencia = X_train[variables].sample(100, random_state=RANDOM_STATE)
    ejemplos = X_test[variables].sample(200, random_state=RANDOM_STATE)

    explicador = shap.TreeExplainer(
        bosque,
        data=referencia,
        feature_perturbation="interventional",
        model_output="probability"
    )

    clase = list(bosque.classes_).index(1)
    valores = explicador(ejemplos)[:, :, clase]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(valores, show=False)
    guardar_figura("shap_global.png")

    shap.plots.waterfall(valores[0], show=False)
    guardar_figura("shap_local.png")

    print("Reserva explicada:", ejemplos.index[0])
    print("Gráficos guardados en:", REPORTS_DIR)