# AI University Project: Classifying Hand-written Digits

This project develops a Super Ensembled deep learning model to classify hand-written digits

**Key Features:**

* Implements VGG19 & ResNET50 finetunned models with a Trained CNN model.
* Achieves an accuracy of **99.20%** on the test dataset.
* Provides insights into the key factors from data data loading all the way to deployment.

**Installation:**


**No Docker :**
1. Clone the repository: `git clone https://github.com/Ouassim-Hamdani/Interface-Hand-Digit-Recognizer.git` or extract zip.
2. Install dependencies: `pip install -r requirements.txt` or `make install`  if you have `make` installed.
3. Either : 
    - Run the main script: `python src/main.py` or `make run-main`.
    - Run the interface : `streamlit run src/app.py` or `make run-app`.

For a detailed description, analysis, and results, please refer to the full report: [report.md](report/report.md) or report.pdf in `report/report.pdf`.

**Docker :**
1. Clone the repository: `git clone https://github.com/Ouassim-Hamdani/Interface-Hand-Digit-Recognizer.git` or extract zip.
2. Run `docker compose up --build` or `make docker` if you have `make` installed.

**Note about Linux** : In case of a library error, run this command `apt-get install -y libgl1-mesa-glx` to install the necessary packages to execute (Already included in Docker).



**Project Folder Architecture**

```
├── notebooks
│   └── training_notebook.ipynb       # Jupyter notebook for training the model.
├── models
│   └── CNN.keras                     # Trained CNN Keras model.
│   └── RESNET50.keras                # Trained ResNET50 Keras model.
│   └── VGG19.keras                   # Trained VGG19 Keras model.
├── data
│   └── train.csv                     # Dataset used for training (CSV format).
│   └── test_gen.csv                  # Generated Dataset used for testing (CSV format).
├── report
│   ├── report.pdf                    # Project report (PDF).
│   ├── report.md                     # Project report (Markdown).
│   ├── report.html                   # Project report (HTML).
│   └── figures                       # Figures Folder used in the report.
├── src
│   ├── main.py                       # Main script for running the project & visualizing.
│   ├── app.py                        # Script for the web interface application (Streamlit).
│   ├── model.py                      # Ensembled Model architecture, class and prediction functions.
│   └── utils.py                      # Utility functions.
├── Dockerfile                        # Instructions for building a Docker image.
├── docker-compose.yaml               # Instructions for orchestrating Docker containers.
├── Makefile                          # Shortcuts for commands to build & run project.
└── requirements.txt                  # Python packages required to run the project.
```


**Author :** Ouassim HAMDANI

**Class :** Master 1 IIA - Multi-source Data Extraction