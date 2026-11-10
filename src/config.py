from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "raw" / "01_reservas_hotel.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET = "no_presentado"
ID_COLUMN = "id_reserva"


if __name__ == "__main__":
    print("Raiz:", ROOT_DIR)
    print("Dataset:", DATA_PATH)
    print("Existe el csv:", DATA_PATH.is_file())