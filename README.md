# AIC354 Assignment 2 - End-to-End Machine Learning System (FA24-BSE-068)

| Task | Folder | Notebook | Model | Test accuracy | Live app |
| --- | --- | --- | --- | --- | --- |
| 1. Titanic Survival | `Task1_Titanic` | `Titanic_Passenger_Survival_Recommended.ipynb` | `svc_trained_model.pkl` | 0.80 | https://fa24-bse-068-titanic.streamlit.app/ |
| 2. Heart Disease | `Task2_HeartDisease` | `Heart_Disease_Prediction.ipynb` | `heart_disease_svc_trained_model.pkl` | 0.85 | https://fa24-bse-068-heart-disease.streamlit.app/ |

Heart Disease inputs (4 UCI attributes): ChestPainType (`cp`), ExerciseInducedAngina (`exang`), MajorVessels (`ca`), Thalassemia (`thal`).

## Before submitting
1. Replace `[Your Name]` in the first cell of both notebooks and in the Developer table of the Heart Disease notebook.
2. Deploy both apps (steps below). If Streamlit gives a different URL, paste the real link into the first cell of the matching notebook.

## Deploy on Streamlit Community Cloud
1. Create a **public** GitHub repository named `ML-Assignment02-FA24-BSE-068`.
2. Upload everything in this folder: on the repo page click **Add file -> Upload files**, drag in `Task1_Titanic`, `Task2_HeartDisease` and `README.md`, then **Commit changes**.
   Or with git, from this folder:
   ```
   git init
   git add .
   git commit -m "Assignment 2"
   git branch -M main
   git remote add origin https://github.com/<your-username>/ML-Assignment02-FA24-BSE-068.git
   git push -u origin main
   ```
3. Open https://share.streamlit.io, sign in with GitHub, click **Create app -> Deploy a public app from GitHub**.
4. Titanic app: Repository = your repo, Branch = `main`, Main file path = `Task1_Titanic/app.py`, App URL = `fa24-bse-068-titanic`. Open **Advanced settings**, pick **Python 3.12**, click **Deploy**.
5. Heart Disease app: repeat step 4 with Main file path = `Task2_HeartDisease/app.py` and App URL = `fa24-bse-068-heart-disease`.

Each app folder has its own `requirements.txt`; `scikit-learn==1.7.2` must stay pinned because the `.pkl` models were saved with that version.

## Upload to Google Classroom
- `Task1_Titanic/`: `Titanic_Passenger_Survival_Recommended.ipynb`, `sample-data.csv`, `svc_trained_model.pkl`, `app.py`
- `Task2_HeartDisease/`: `Heart_Disease_Prediction.ipynb`, `processed.cleveland.data`, `sample-data.csv`, `heart_disease_svc_trained_model.pkl`, `app.py`

## Run locally
```
pip install numpy pandas scikit-learn==1.7.2 matplotlib seaborn prettytable notebook streamlit
jupyter notebook                      # open a notebook -> Kernel -> Restart & Run All
streamlit run Task2_HeartDisease/app.py
```
