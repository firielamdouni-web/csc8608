# TP1 - Segmentation Interactive avec SAM

## Description
Ce projet implémente une mini-application de segmentation interactive d'images utilisant SAM (Segment Anything Model).

## Structure du projet
```
TP1/
├── data/
│   └── images/          # Images à segmenter
├── src/
│   ├── app.py          # Interface Streamlit
│   ├── sam_utils.py    # Utilitaires SAM
│   ├── geom_utils.py   # Calculs géométriques
│   └── viz_utils.py    # Visualisation
├── outputs/
│   ├── overlays/       # Images avec masques
│   └── logs/           # Logs d'exécution
├── report/
│   └── report.md       # Rapport du TP
├── requirements.txt     # Dépendances
└── README.md           # Ce fichier
```

## Installation

1. Réserver un nœud GPU via SLURM
2. Activer l'environnement conda existant
3. Installer les dépendances : `pip install -r requirements.txt`

## Utilisation

Lancer l'application Streamlit :
```bash
streamlit run TP1/src/app.py --server.port <PORT> --server.address 0.0.0.0
```

Puis créer un tunnel SSH depuis votre machine locale :
```bash
ssh -L <PORT>:localhost:<PORT> nodeX_tsp
```

## Auteur
CSC 8608 - Deep Learning
