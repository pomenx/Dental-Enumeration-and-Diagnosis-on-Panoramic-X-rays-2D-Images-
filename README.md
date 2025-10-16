# Dental-Enumeration-and-Diagnosis-on-Panoramic-X-rays-2D-Images

Questo progetto combina **HierarchicalDet** con **MedSAM** per realizzare un sistema avanzato di **rilevamento e segmentazione di immagini mediche**.
È progettato per analizzare dataset di immagini radiologiche o istologiche e fornire risultati precisi di classificazione e localizzazione di lesioni.

---

## 🛠️ Installazione e configurazione

### 1. Clonare la repository

```bash
git clone https://github.com/tuo-username/Dental-Enumeration-and-Diagnosis-on-Panoramic-X-rays-2D-Images.git
cd Dental-Enumeration-and-Diagnosis-on-Panoramic-X-rays-2D-Images
```

### 2. Creare e attivare l'ambiente Conda

Assicurati di avere [Conda](https://docs.conda.io/en/latest/) installato. Poi esegui:

```bash
conda env create -f environment.yml
conda activate HierarchicalSeg
```

Questo comando creerà un environment chiamato `HierarchicalSeg` con tutte le dipendenze necessarie.

---

## 🚀 Esecuzione della Demo

Dopo aver configurato l’ambiente e i file di configurazione, puoi eseguire la demo con il seguente comando:

```bash
python demo.py --config-file configs/diffdet.custom.swinbase.nonpretrain.yaml --input input/test_0.png --nclass 3 --seg_model medsam --opts MODEL.WEIGHTS weights/HierarchicalDet/disease2/model_final.pth   
```

---

## 🧩 Argomenti principali

| Argomento       | Descrizione                                                 |
| --------------- | ----------------------------------------------------------- |
| `--config-file` | File di configurazione del modello HierarchicalDet          |
| `--input`       | Percorso alle immagini di input (supporta wildcard `*.png`) |
| `--nclass`      | Numero di classi nel dataset                                |
| `--opts`        | Opzioni aggiuntive (pesi del modello HierarchicalDet)       |

---

## 📚 Riferimenti

* [MedSAM](https://github.com/bowang-lab/MedSAM)
* [HierarchicalDet](https://github.com/facebookresearch/detectron2/projects)
* [YOLO](https://github.com/ultralytics/ultralytics)

