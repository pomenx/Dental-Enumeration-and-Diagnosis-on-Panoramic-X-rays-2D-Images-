# Dental-Enumeration-and-Diagnosis-on-Panoramic-X-rays-2D-Images

Questo progetto combina **HierarchicalDet** con **MedSAM** e **YOLO** per realizzare un sistema avanzato di **rilevamento e segmentazione su radiografie panoramiche dentali a raggi X**.
Propone un framework per la rilevazione e segmentazione dei denti, assegando 3 classi al dente rilevato: Quadrante , Enumerazione e Diagnosi.

---

## 🛠️ Installazione e configurazione

### 1. Requisiti

Assicurati di avere **Python 3.10** installato sul tuo sistema.  
Puoi verificarlo con:

```bash
python --version
```

Se non hai Python 3.10, puoi scaricarlo da [python.org/downloads](https://www.python.org/downloads/).


### 2. Clonare la repository

```bash
git clone https://github.com/tuo-username/Dental-Enumeration-and-Diagnosis-on-Panoramic-X-rays-2D-Images.git
cd Dental-Enumeration-and-Diagnosis-on-Panoramic-X-rays-2D-Images
```

### 3. Creare e attivare un ambiente virtuale

Si consiglia di creare un ambiente virtuale per evitare conflitti tra pacchetti:

```bash
python -m venv HierSeg
source HierSeg/bin/activate      # Su Linux/Mac
HierSeg\Scripts\activate       # Su Windows
```

### 4. Installare le dipendenze

Installa tutte le dipendenze necessarie tramite `pip`:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```
---

## 🚀 Esecuzione della Demo

Dopo aver configurato l’ambiente e i file di configurazione, puoi eseguire la demo con il seguente comando scegliendo quale modello utilizzare per la segmentazione, 'medsam' o 'yolo', modificando --seg_model:

```bash
python demo.py --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml --input input/test_0.png --nclass 3 --seg_model medsam --opts MODEL.WEIGHTS weights/HierarchicalDet/disease/model_final.pth   
```

---

## 🧩 Argomenti principali

| Argomento       | Descrizione                                                 |
| --------------- | ----------------------------------------------------------- |
| `--config-file` | File di configurazione del modello HierarchicalDet          |
| `--input`       | Percorso alle immagini di input (supporta wildcard `*.png`) |
| `--nclass`      | Numero di classi                                            |
| `--seg_model`   | Modello di segmentazione ('medsam' o 'yolo')                |
| `--opts`        | Opzioni aggiuntive (pesi del modello HierarchicalDet)       |

---

## 📚 Riferimenti

* [MedSAM](https://github.com/bowang-lab/MedSAM)
* [HierarchicalDet](https://github.com/facebookresearch/detectron2/projects)
* [YOLO](https://github.com/ultralytics/ultralytics)

