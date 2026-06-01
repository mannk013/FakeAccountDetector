# 🤖 Social Media Fake Account Detector

## Project Overview
A Machine Learning project that detects fake accounts on social media platforms.
The model analyzes account features such as followers count, posts, and profile settings
to predict whether an account is fake or real.

## Demo
> App built with Streamlit — enter account details and get instant prediction.

## Dataset
- **Source:** Kaggle — Fake and Real Social Media Accounts
- **Size:** 5,000 accounts (2,500 fake + 2,500 real)
- **Balance:** Perfectly balanced (50/50)

## Tech Stack
- **Python** — core language
- **Pandas & NumPy** — data manipulation
- **Scikit-learn** — ML models
- **Matplotlib & Seaborn** — visualizations
- **Streamlit** — web application

## ML Models
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 83% | 0.896 |
| **Random Forest** | **91%** | **0.955** |

## Key Findings
- Fake accounts tend to have significantly fewer posts and likes
- Real accounts are more likely to have location enabled
- `statuses_count`, `favourites_count` and `followers_count` are the most important features

## Framework
This project follows the **OSEMN** framework:
- **O**btain — Load dataset
- **S**crub — Clean data, handle missing values
- **E**xplore — EDA with visualizations
- **M**odel — Train and evaluate ML models
- **i**Nterpret — Conclusions + Streamlit app

## Project Structure
fake_account_detector/
│
├── Data/
│   ├── fake_users.csv
│   └── real_users.csv
│
├── fake_account_detector.ipynb  # EDA + ML notebook
├── app.py                       # Streamlit application
└── README.md


## How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn streamlit

# Run the app
streamlit run app.py
```

## Author
**Michał Mańkowski**  
