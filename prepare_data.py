"""
prepare_data.py
---------------
Dieses Script bereitet den Kaggle-Rohdatensatz für das Training vor.
Es wählt zufällig Bilder aus, mischt sie und kopiert sie
in die train/val/test Ordnerstruktur.
 
Ablauf:
    1. Rohdaten aus raw_data/ einlesen
    2. Zufällig mischen (reproduzierbar per Seed)
    3. Auf IMAGES_PER_CLASS Bilder begrenzen
    4. In train / val / test aufteilen
    5. In data/ kopieren
 
Ausführen:
    uv run python prepare_data.py
"""
 
import os
import random
import shutil
 
 
# ---------------------------------------------------------------------------
# Konfiguration – hier kannst du alle Einstellungen anpassen
# ---------------------------------------------------------------------------
 
SOURCE_DIR = "raw_data"         # Ordner mit den Kaggle-Rohdaten
TARGET_DIR = "data"             # Zielordner für train/val/test
 
IMAGES_PER_CLASS = 1500         # Wie viele Bilder pro Klasse verwendet werden
TRAIN_SPLIT = 0.70              # 70% für Training
VAL_SPLIT = 0.15                # 15% für Validation
TEST_SPLIT = 0.15               # 15% für Test
 
SEED = 42                       # Seed für Reproduzierbarkeit
 
# Mapping: Zielordnername (links) → Kaggle-Ordnername (rechts)
CLASSES = {
    "cubism":  "Cubism",
    "pop_art": "Pop_Art",
    "realism": "Realism",
}
 
# Erlaubte Bilddateiformate
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
 
 
# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
 
def get_image_files(folder: str) -> list[str]:
    """Gibt eine Liste aller Bilddateinamen in einem Ordner zurück."""
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
 
 
def split_images(images: list[str]) -> dict[str, list[str]]:
    """Teilt eine Liste von Bildern in train/val/test auf."""
    n_train = int(len(images) * TRAIN_SPLIT)
    n_val = int(len(images) * VAL_SPLIT)
 
    return {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }
 
 
def copy_images(images: list[str], source_dir: str, target_dir: str) -> None:
    """Kopiert eine Liste von Bildern von source_dir nach target_dir."""
    os.makedirs(target_dir, exist_ok=True)
 
    for filename in images:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        shutil.copy(source_path, target_path)
 
 
# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
 
def main():
    print("Starte Datenvorbereitung...")
    print(f"Seed: {SEED} | Bilder pro Klasse: {IMAGES_PER_CLASS}")
    print(f"Aufteilung: {int(TRAIN_SPLIT*100)}% Train / "
          f"{int(VAL_SPLIT*100)}% Val / "
          f"{int(TEST_SPLIT*100)}% Test\n")
 
    random.seed(SEED)
 
    for target_class, source_class in CLASSES.items():
 
        source_dir = os.path.join(SOURCE_DIR, source_class)
 
        # Prüfen ob der Quellordner existiert
        if not os.path.exists(source_dir):
            print(f"FEHLER: Ordner nicht gefunden – {source_dir}")
            continue
 
        # Bilder einlesen und mischen
        all_images = get_image_files(source_dir)
        random.shuffle(all_images)
 
        # Auf gewünschte Anzahl begrenzen
        selected_images = all_images[:IMAGES_PER_CLASS]
 
        if len(selected_images) < IMAGES_PER_CLASS:
            print(f"WARNUNG: {source_class} hat nur "
                  f"{len(selected_images)} Bilder (weniger als {IMAGES_PER_CLASS})")
 
        # In splits aufteilen und kopieren
        splits = split_images(selected_images)
 
        for split_name, images in splits.items():
            target_dir = os.path.join(TARGET_DIR, split_name, target_class)
            copy_images(images, source_dir, target_dir)
            print(f"  {target_class:10} / {split_name:5}: {len(images):4} Bilder kopiert")
 
        print()
 
    print("Fertig! Datensatz liegt in data/")
 
 
if __name__ == "__main__":
    main()
 