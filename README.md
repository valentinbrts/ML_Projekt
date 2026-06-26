# 🎨 Epochen Klassifikator

## 🚀 Installation & Ausführung

**Voraussetzung:** Python 3.14 (siehe `.python-version`)

### 1. Repository klonen

```bash
git clone <repo-url>
cd ml-projekt
```

### 2. Abhängigkeiten installieren

Es gibt zwei Wege – je nachdem welches Tool ihr nutzt:

**Option A – uv (empfohlen)**

Falls uv noch nicht installiert ist:
```bash
winget install astral-sh.uv   # Windows
# curl -LsSf https://astral.sh/uv/install.sh | sh  # Mac/Linux
```

Dann Abhängigkeiten installieren:
```bash
uv sync
```

Das liest `pyproject.toml` und `uv.lock` und erstellt automatisch eine `.venv` mit exakt denselben Paketversionen.

**Option B – pip**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 3. In VS Code öffnen

* Jupyter-Extension installieren (`ms-toolsai.jupyter`)
* Projektordner direkt (nicht einen Unterordner) in VS Code öffnen
* Notebook öffnen → oben rechts „Select Kernel" → „Python Environments" → `.venv` auswählen

### 4. Datensatz einrichten

1. WikiArt-Datensatz von Kaggle herunterladen: https://www.kaggle.com/datasets/steubk/wikiart
2. Relevante Epochen-Ordner in `raw_data/` ablegen.

---

## 📌 Projektübersicht

Dieses Projekt klassifiziert Gemälde mittels **Transfer Learning mit ResNet50 (PyTorch)** in eine von sechs Kunstepochen. Zusätzlich zur Trainingspipeline gibt es eine interaktive **Streamlit-App**, mit der eigene Bilder hochgeladen und live klassifiziert werden können.

**Klassen:**

* Barock (Baroque)
* Kubismus (Cubism)
* Minimalismus (Minimalism)
* Pop-Art (Pop Art)
* Realismus (Realism)
* Romantik (Romanticism)

## 👥 Team

* Valentin Bridts
* Eric Schmidt
* Anna Julitz

---

## 📁 Projektstruktur

```text
ml-projekt/
│
├── data/                       # Finaler, sortierter Datensatz (nicht in Git)
│   ├── train/<klasse>/
│   ├── val/<klasse>/
│   └── test/<klasse>/
│
├── raw_data/                   # Unsortierte Rohdaten direkt von Kaggle (nicht in Git)
│   └── <klasse>/
│
├── model/
│   └── best_model_run4_resnet50.pth
│
├── notebooks/
│   ├── notebook_V1.ipynb
│   ├── notebook_V2.ipynb       # finale, verwendete Trainingsversion
│   └── notebook.ipynb
│
├── outputs/
│   ├── confusion_matrix.png
│   └── training_curves.png
│
├── ML_Projekt run analysis.xlsx  # Übersicht aller Trainingsläufe (Accuracy, Confusion Matrix, etc.)
├── app.py                      # Streamlit-App
├── prepare_data.py             # Sortiert raw_data/ in data/train|val|test
│
├── pyproject.toml              # Abhängigkeiten (ersetzt requirements.txt)
├── uv.lock                     # exakte, reproduzierbare Versionen
├── .python-version
├── .gitignore
└── README.md
```

> `data/`, `raw_data/` und `model/` sind aus Git ausgeschlossen (siehe `.gitignore`), da sie zu groß sind. Die `.gitkeep`-Dateien sorgen dafür, dass die Ordnerstruktur trotzdem sichtbar bleibt.

---

## 📊 Datensatz

Basis ist das **WikiArt-Datenset (Kaggle)**: https://www.kaggle.com/datasets/steubk/wikiart

| Klasse       | Train | Val | Test |
| ------------ | ----: | --: | ---: |
| Barock       |  1050 | 225 |  225 |
| Kubismus     |  1050 | 225 |  225 |
| Minimalismus |   935 | 200 |  202 |
| Pop-Art      |  1038 | 222 |  223 |
| Realismus    |  1050 | 225 |  225 |
| Romantik     |  1050 | 225 |  225 |

**Gesamt:** 6.173 (Train) + 1.322 (Val) + 1.325 (Test) = **8.820 Bilder**

### Vorgehen zur Datensatzerstellung

1. Datensatz von Kaggle herunterladen.
2. Mit `prepare_data.py` eine begrenzte, ausgewogene Menge pro Klasse aus `raw_data/` auswählen.
3. Bilder automatisch in `data/train`, `data/val`, `data/test` (jeweils mit Unterordner pro Klasse) einsortieren.

---

## 🧠 Modell & Training

### Architektur

* **Backbone:** ResNet50, vortrainiert auf ImageNet (Torchvision)
* Alle Convolutional-Layer eingefroren, nur der Klassifikationskopf wird trainiert:

```python
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(2048, 6)
)
```

### Datenvorverarbeitung

**Training (6 Schritte):**

1. Resize (224 × 224)
2. Random Horizontal Flip
3. Random Rotation (±15°)
4. Color Jitter (Brightness 0.3, Contrast 0.3, Saturation 0.2)
5. ToTensor
6. ImageNet-Normalisierung

**Validation / Test (3 Schritte):**

1. Resize (224 × 224)
2. ToTensor
3. ImageNet-Normalisierung

### Trainingskonfiguration

| Parameter      | Wert                                                    |
| -------------- | -------------------------------------------------------- |
| Optimizer      | Adam (nur `model.fc.parameters()`)                       |
| Learning Rate  | 0.001                                                     |
| Loss-Funktion  | CrossEntropyLoss                                          |
| Batch Size     | 64                                                        |
| Max. Epochen   | 50                                                        |
| Early Stopping | Patience = 10                                              |
| LR-Scheduler   | ReduceLROnPlateau (mode="min", patience=3, factor=0.5)    |

### Hardware

Entwicklung lokal in **VS Code mit Jupyter-Extension**, rechenintensives Training auf **Google Colab (GPU)**. Das Notebook erkennt automatisch die verfügbare Hardware:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 📈 Ergebnisse

| Kennzahl    | Wert    |
| ----------- | ------- |
| Accuracy    | 69.96 % |
| Macro F1    | 0.70    |
| Weighted F1 | 0.70    |

Das beste Modell wird anhand des niedrigsten Validation Loss ausgewählt und automatisch gespeichert:

```text
model/best_model_run4_resnet50.pth
```

Enthaltene Informationen im Checkpoint: `model_state_dict`, Epoche, Validation Loss.

Visualisierungen zu Trainingsverlauf und Fehlklassifikationen liegen in `outputs/`:

* `training_curves.png` – Verlauf von Train-/Val-Loss über die Epochen
* `confusion_matrix.png` – Confusion Matrix auf dem Testset

Eine detaillierte Übersicht aller durchgeführten Trainingsläufe (inkl. Accuracy, Confusion Matrix und weiterer Evaluationswerte je Run) befindet sich in `ML_Projekt run analysis.xlsx`. Die oben angegebenen finalen Werte stammen aus **Run 4** (`best_model_run4_resnet50.pth`), dem besten Lauf in dieser Übersicht.

---

## 🖥️ Streamlit-App

`app.py` lädt den gespeicherten Checkpoint (`model/best_model_run4_resnet50.pth`) und führt Inferenz auf hochgeladenen Bildern durch. Funktionen:

* Bild hochladen
* Vorhersage der Kunstepoche mit Wahrscheinlichkeit
* Balkendiagramm der Wahrscheinlichkeiten über alle Klassen
* Anzeige der Confusion Matrix
* Anzeige der Trainingskurven

**Wichtig:** Die App ist unabhängig vom Notebook – die einzige Verbindung ist die gespeicherte `.pth`-Datei. Der Ablauf ist:

```
notebook_V2.ipynb → trainiert ResNet50 → speichert .pth
                                              ↓
                                  app.py lädt .pth → Inferenz auf Nutzerbild
```

---

## 🚀 Installation & Ausführung

Das Projekt nutzt **uv** als Dependency-Manager. `pyproject.toml` übernimmt die Funktion einer `requirements.txt` und listet alle benötigten Pakete; `uv.lock` fixiert die exakten Versionen für volle Reproduzierbarkeit.

**Voraussetzung:** Python 3.14 (siehe `.python-version`)

### 1. Repository klonen

```bash
git clone <repo-url>
cd ml-projekt
```

### 2. Abhängigkeiten installieren

```bash
uv sync
```

Das liest `pyproject.toml` und `uv.lock` und erstellt automatisch eine `.venv` mit exakt denselben Paketversionen.

### 3. In VS Code öffnen

* Jupyter-Extension installieren
* Projektordner direkt (nicht einen Unterordner) in VS Code öffnen
* Notebook öffnen → „Select Kernel“ → Python Environments → `.venv` auswählen

### 4. Datensatz einrichten

Siehe Abschnitt [Datensatz](#-datensatz) – Rohdaten herunterladen, `prepare_data.py` ausführen.

### 5. App starten

```bash
streamlit run app.py
```

---

## 🛠️ Technologien

* Python 3.14
* PyTorch / Torchvision
* Streamlit
* NumPy, Pandas
* Scikit-learn
* Matplotlib, Seaborn
* VS Code + Jupyter-Extension
* Google Colab (GPU-Training)
* uv (Dependency-Management)

---

## 🧩 Herausforderungen

* Ähnlichkeit einzelner Kunstepochen (z. B. Realismus vs. Romantik) erschwert die Trennung
* Begrenzte lokale Rechenleistung → Training größtenteils über Google Colab
* Ausbalancieren der Klassengrößen bei der Datensatzauswahl

---

## ⚠️ Limitationen

Vergleichsweise kleiner Datensatz pro Klasse (zwischen 935 und 1050 Trainingsbildern)
Modell wurde nur auf WikiArt-Bildern getestet, nicht auf Echtweltfotos oder Bildern aus anderen Quellen
Nur der Klassifikationskopf trainiert (eingefrorenes Backbone) – möglicherweise nicht optimal für stilistisch sehr ähnliche Epochen
Kein systematischer Vergleich mit anderen Architekturen (z. B. EfficientNet, ViT)

---

## 🔭 Mögliche Erweiterungen

* Fine-Tuning weiterer ResNet-Layer statt nur des Klassifikationskopfs
* Systematische Hyperparameter-Optimierung
* Vergleich mit anderen Architekturen (z. B. EfficientNet, Vision Transformer)
* Erweiterung der App um Top-3-Vorhersagen oder Grad-CAM-Heatmaps

---

## 📌 Autoren

Projekt im Rahmen ML4B (Machine Learning for Business), SoSe 2026, FAU – Klassifikation von Kunstepochen mittels Transfer Learning und Convolutional Neural Networks.
