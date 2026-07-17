from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "DR Detection API"
    DEBUG: bool = False

    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = "Ratheesh@1703"
    DB_NAME: str = "dr_detection"

    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    CHECKPOINT_PATH: Path = BASE_DIR / "checkpoints" / "best_fold0.pth"

    IMG_SIZE: int = 224
    NUM_CLASSES: int = 5
    BACKBONE: str = "efficientnet_b4"
    DROPOUT: float = 0.3

    class Config:
        env_file = ".env"


settings = Settings()
