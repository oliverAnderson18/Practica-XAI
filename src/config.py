from pathlib import Path

# parametros que usaremos
RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET = "no_presentado"
ID_COLUMN = "id_reserva"

# rutas principales
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "01_reservas_hotel.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"