from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from data import preparar_particion


def crear_modelo():

    # creamos el pipeline para la regresión logística, balanceando por clases
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000,
                           random_state=RANDOM_STATE)
    )


def crear_cv():
    return StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)


def seleccionar_variables(X_train, y_train):

    # FFS: empieza sin variables y añade la que mas mejora el F1 en cada paso
    selector = SequentialFeatureSelector(
        estimator=crear_modelo(),
        direction="forward", # indica que es hacia delante la seleccion
        n_features_to_select="auto",
        tol=0.0001,
        scoring="f1",
        cv=crear_cv(),
        n_jobs=-1
    )
    selector.fit(X_train, y_train)
    return X_train.columns[selector.get_support()].tolist()


if __name__ == "__main__":
    X_train, _, y_train, _ = preparar_particion()
    seleccionadas = seleccionar_variables(X_train, y_train)

    scores = cross_val_score(crear_modelo(), X_train[seleccionadas], y_train,
                             cv=crear_cv(), scoring="f1")

    print("Variables seleccionadas:", seleccionadas)
    print(f"F1 de selección: {scores.mean():.4f} +- {scores.std():.4f}")
