# Dental-Enumeration-and-Diagnosis-on-Panoramic-X-rays-2D-Images-

Questo progetto combina **HierarchicalDet** con **MedSAM** per realizzare un sistema avanzato di **rilevamento e segmentazione di immagini mediche**.  
È progettato per analizzare dataset di immagini radiologiche o istologiche e fornire risultati precisi di classificazione e localizzazione di lesioni.

---

## 📦 Installazione dell’ambiente Conda

1. **Clona la repository:**
   ```bash
   git clone https://github.com/tuo-utente/tuo-progetto.git
   cd tuo-progetto
   ```

2. **Crea un nuovo ambiente Conda:**
   ```bash
   conda create -n hierdet python=3.10 -y
   ```

3. **Attiva l’ambiente:**
   ```bash
   conda activate hierdet
   ```

4. **Installa le dipendenze:**
  ```bash
   conda env create -f env.yml
   ```

---

## 🚀 Esecuzione della Demo

Dopo aver configurato l’ambiente e i file di configurazione, puoi eseguire la demo con il seguente comando:

```bash
python HierarchicalDet_MedSAM.py   --config-file C:/Users/Admin/Desktop/VA_project/HierarchicalDet/configs/diffdet.custom.swinbase.nonpretrain.yaml   --input C:/Users/Admin/Desktop/VA_project/DENTEX/disease/input/*.png   --nclass 3   --opts MODEL.WEIGHTS C:/Users/Admin/Desktop/VA_project/HierarchicalDet/pesi/disease2/model_final.pth
```

---

## 🧩 Argomenti principali

| Argomento | Descrizione |
|------------|-------------|
| `--config-file` | File di configurazione del modello HierarchicalDet |
| `--input` | Percorso alle immagini di input (supporta wildcard `*.png`) |
| `--nclass` | Numero di classi nel dataset |
| `--opts` | Opzioni aggiuntive (pesi del modello HierarchicalDet) |


## ⚙️ Requisiti hardware consigliati

- GPU NVIDIA con almeno **8 GB VRAM**
- **CUDA 11.8** o superiore
- **Driver NVIDIA** aggiornati
- **Python ≥ 3.9**

---

## 📚 Riferimenti
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [HierarchicalDet](https://github.com/facebookresearch/detectron2/projects)
- [YOLO](https://github.com/ultralytics/ultralytics)
