# Reachy_mirror: a mirroring application for Reachy
Version en français en dessous
## Requirements
- python3
- pip
- wget
- a Reachy robot connected to a local network
- a computer connected to the same local network (can be reachy's internal computer)
## Instalation
### 1. download the model:
```bash
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```
### 2. install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python3 reachy_mirror.py [-h] [-m] [-d] ip
```

```
Mirror the movement of the arms of the person in front of the robot

positional arguments:
  ip              reachy's IP adress

options:
  -h, --help      Show this help message and exit
  -m, --mirrored  Disable the mirror effet (with this flag, mooving the left arm will moove the robot's left arm instead of the right arm)
  -d, --debug     Activate debug mode (run localy and uses the PC webcam)

To exit the program when running, press q
```

### To quit the program, press q

</br>
</br>

# Reachy_mirror: une application "miroir" pour Reachy
English version on top
## Prérequis
- python3
- pip
- wget
- Un robot Reachy connecté à un réseau local
- Un ordinateur connecté au même réseau local (peut être l'ordinateur interne de Reachy)
## Instalation
### 1. télécharger le modèle:
```bash
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```
### 2. installer les dépendances:
```bash
pip install -r requirements.txt
```

## Utilisation
```bash
python3 reachy_mirror.py [-h] [-m] [-d] ip
```

```
Reproduit le mouvements des bras de la personne se trouvant en face du robot

arguments positionels:
  ip              l'addresse IP de Reachy

options:
  -h, --help      Montre ce message et termine le programme
  -m, --mirrored  Déactive l'effet miroir (avec ce drapeau, bouger le bras gauche bougera le bras gauche du robot au lieu du bras droit)
  -d, --debug     Active le mode de débogueage (s'exécute localement et utilise la caméra du PC)

Pour quitter le programme, appuyez sur q
```

### Pour quitter le programme, appuyez sur q