# MKUNet — Multi-Kernel U-Net for Medical Image Segmentation

> Projet réalisé dans le cadre du cours **INF8225** — Modèles probabilistes et apprentissage  
> Polytechnique Montréal

---

## 📁 Structure du dépôt

```
ton-repo/
│
├── INF8225_MKUNet.ipynb          # Notebook principal (53 cellules)
│                                 # Entraînement, évaluation, visualisation
│
├── mkunet_network.py             # Architecture complète du modèle
│                                 # Modules : MKDC, MKIR, CA, SA, GAG, MKUNet
│
├── requirements.txt              # Dépendances Python
│
├── utils/
│   ├── dataloader.py             # SegDataset + build_loader (6 datasets)
│   └── utils.py                  # structure_loss, dice_score, iou_score,
│                                 # RunningAverage, helpers divers
│
├── data/                         # ⚠️ À créer manuellement (voir ci-dessous)
│   ├── BUSI/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   └── test/
│   │       ├── images/
│   │       └── masks/
│   ├── ClinicDB/                 # (même structure que BUSI)
│   ├── ColonDB/                  # (même structure que BUSI)
│   ├── ISIC18/                   # (même structure que BUSI)
│   ├── DSB18/                    # (même structure que BUSI)
│   └── EM/                       # (même structure que BUSI)
│
├── checkpoints/
│   └── <RUN_ID>_best.pth         # Meilleurs poids sauvegardés automatiquement
│
└── results/
    ├── summary_all_datasets.xlsx        # Métriques agrégées (Dice, IoU, …)
    ├── curves_<RUN_ID>.png              # Courbes loss / métriques par epoch
    ├── qualitative_<DATASET>_test.png   # Prédictions qualitatives par dataset
    └── comparison_all_datasets.png      # Comparaison visuelle inter-datasets
```

---

## 📦 Téléchargement des données

Toutes les données nécessaires sont regroupées dans une archive disponible sur Google Drive :

🔗 **[Télécharger les données (Google Drive)](https://drive.google.com/file/d/1kd5ZynM48Q_2WjK9xmxi2dqLDQzUaTBr/view?usp=sharing)**

### Étapes d'installation

**1. Télécharger l'archive** depuis le lien ci-dessus.

**2. Décompresser l'archive** à la racine du dépôt :

```bash
unzip data.zip -d ton-repo/
```

> L'archive doit générer directement le dossier `data/` avec les 6 sous-dossiers de datasets.

**3. Vérifier la structure** obtenue :

```bash
ls ton-repo/data/
# → BUSI  ClinicDB  ColonDB  ISIC18  DSB18  EM
```

---

## 🗄️ Datasets inclus

| Dataset     | Domaine                        | Tâche                          |
|-------------|--------------------------------|--------------------------------|
| **BUSI**    | Échographie mammaire           | Segmentation de tumeurs        |
| **ClinicDB**| Coloscopie                     | Détection de polypes           |
| **ColonDB** | Coloscopie                     | Détection de polypes           |
| **ISIC18**  | Dermatologie                   | Segmentation de lésions        |
| **DSB18**   | Microscopie cellulaire         | Segmentation de noyaux         |
| **EM**      | Microscopie électronique       | Segmentation de membranes      |

---

## ⚙️ Installation des dépendances

```bash
pip install -r requirements.txt
```

---

## 🚀 Utilisation

Ouvrir et exécuter le notebook principal :

```bash
jupyter notebook INF8225_MKUNet.ipynb
```

Le notebook couvre :
- Le chargement des données via `utils/dataloader.py`
- La définition et l'instanciation de MKUNet (`mkunet_network.py`)
- L'entraînement avec sauvegarde automatique du meilleur modèle dans `checkpoints/`
- L'évaluation sur les sets de test (Dice, IoU)
- La génération des figures et du fichier Excel dans `results/`

---

## 🧠 Architecture — MKUNet

Le modèle repose sur une architecture de type **U-Net** enrichie de plusieurs modules :

| Module   | Rôle                                                              |
|----------|-------------------------------------------------------------------|
| **MKDC** | Multi-Kernel Dilated Convolution — capture multi-échelle          |
| **MKIR** | Multi-Kernel Inverted Residual — extraction de features légère    |
| **CA**   | Channel Attention — pondération des canaux informatifs            |
| **SA**   | Spatial Attention — focalisation sur les régions pertinentes      |
| **GAG**  | Guided Attention Gate — fusion encoder/decoder guidée             |

---

## 📊 Sorties générées

Après entraînement complet, le dossier `results/` contiendra :

- **`summary_all_datasets.xlsx`** — tableau récapitulatif des métriques sur tous les datasets
- **`curves_<RUN_ID>.png`** — courbes d'entraînement (loss, Dice, IoU par epoch)
- **`qualitative_<DATASET>_test.png`** — visualisations côte-à-côte (image / masque GT / prédiction)
- **`comparison_all_datasets.png`** — comparaison visuelle des résultats entre datasets

---

## 📝 Notes

- Les dossiers `checkpoints/` et `results/` sont créés automatiquement au premier lancement.
- Seul le dossier `data/` doit être créé manuellement en décompressant l'archive Drive.
- Chaque run est identifié par un `RUN_ID` horodaté pour faciliter la reproductibilité.
