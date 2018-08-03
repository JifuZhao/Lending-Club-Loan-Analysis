<img src="./result/logo.png" width=500px>

# Lending Club Loan Analysis and Modeling
LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. Lending Club operates an online lending platform that enables borrowers to obtain a loan, and investors to purchase notes backed by payments made on loans. Lending Club is the world's largest peer-to-peer lending platform. (from [Wikipedia](https://en.wikipedia.org/wiki/Lending_Club)).

The goal of this project is to analyze and model Lending Club's issued loans. A summary of the whole projects can be found in the corresponding Jupyter notebook: [0. Summary.ipynb](https://github.com/JifuZhao/Lending-Club-Loan-Analysis/blob/master/0.%20Summary.ipynb).


***
## Data
The loan data is available through multiple sources, including [Kaggle Lending Club Loan Data](https://www.kaggle.com/wendykan/lending-club-loan-data), [All Lending Club Load Data](https://www.kaggle.com/wordsforthewise/lending-club), or [Lending Club Statistics](https://www.lendingclub.com/info/download-data.action). In this project, I use the data from [Kaggle Lending Club Loan Data](https://www.kaggle.com/wendykan/lending-club-loan-data), which contains the issued load data from 2007 to 2015. In addition, I also use the issued loan data from 2016 from [Lending Club Statistics](https://www.lendingclub.com/info/download-data.action).

The data collection and concatenation process can be found in the corresponding notebook: [1. Data Collection and Concatenation.ipynb](https://github.com/JifuZhao/Lending-Club-Loan-Analysis/blob/master/1.%20Data%20Collection%20and%20Concatenation.ipynb).


***
## Data Cleaning
- Notebook: [2. Data Cleaning.ipynb](https://github.com/JifuZhao/Lending-Club-Loan-Analysis/blob/master/2.%20Data%20Cleaning.ipynb)


***
## Feature Engineering
- Notebook: [3. Feature Engineering.ipynb](https://github.com/JifuZhao/Lending-Club-Loan-Analysis/blob/master/3.%20Feature%20Engineering.ipynb)


***
## Visualization
- Categorical and discrete features: [4. Data Visualization - Discrete Variable.ipynb](https://github.com/JifuZhao/Lending-Club-Loan-Analysis/blob/master/4.%20Data%20Visualization%20-%20Discrete%20Variable.ipynb)
- Numerical features: [4. Data Visualization - Numerical Variable.ipynb](https://github.com/JifuZhao/Lending-Club-Loan-Analysis/blob/master/4.%20Data%20Visualization%20-%20Numerical%20Variable.ipynb)
- Summary of influential features: [4. Data Visualization Summary.ipynb](https://github.com/JifuZhao/Lending-Club-Loan-Analysis/blob/master/4.%20Data%20Visualization%20Summary.ipynb)

Since the above notebooks have relatively large file sizes, to view them, there are two suggest ways.
- Download the corresponding html files from folder `./htmls/`
- View the notebook in nbviewer: [nbviewer.jupyter.org/](https://nbviewer.jupyter.org/)

The corresponding nbviewer pages are as follows:
- Categorical and discrete features: [4. Data Visualization - Discrete Variable.ipynb](https://nbviewer.jupyter.org/github/JifuZhao/Lending-Club-Loan-Analysis/blob/master/4.%20Data%20Visualization%20-%20Discrete%20Variable.ipynb)
- Numerical features: [4. Data Visualization - Numerical Variable.ipynb](https://nbviewer.jupyter.org/github/JifuZhao/Lending-Club-Loan-Analysis/blob/master/4.%20Data%20Visualization%20-%20Numerical%20Variable.ipynb)
- Summary of influential features: [4. Data Visualization Summary.ipynb](https://nbviewer.jupyter.org/github/JifuZhao/Lending-Club-Loan-Analysis/blob/master/4.%20Data%20Visualization%20Summary.ipynb)


***
## Machine Learning
For binary classification problems, there are some commonly used algorithms, from the widely used [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression), to tree-based ensemble models, such as [Random Forest](https://en.wikipedia.org/wiki/Random_forest) and [Boosting](https://en.wikipedia.org/wiki/Boosting_(machine_learning) algorithms.

For imbalanced classification problems, despite the naive method, there are several re-sampling based methods, including:
- Without Sampling
- Under-Sampling
- Over-Sampling
- Synthetic Minority Oversampling Technique (SMOTE)
- Adaptive Synthetic (ADASYN) sampling

Here, the performance of several commonly used algorithms under the conditions of without sampling and over-sampling are compared. The metric used here is AUC, or Area Under the ROC Curve.

While the famous [scikit-learn](http://scikit-learn.org/stable/) has been widely used for a lot of problems, it requires manually transformation of categorical variable into numerical format, which is not always a good choice. There are several new packages that naively support categorical features, including [H2O](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html#), [LightGBM](https://lightgbm.readthedocs.io/en/latest/), and [CatBoost](https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/).

In this projects, several widely used algorithms are explored, including:
- Logistic Regression
- Random Forest
- Boosting
- Stacked Models

### Model Performance Comparison

| Model                    | Logistic Regression | Random Forest | Random Forest | Boosting   | Boosting   |
|----------------------------------------------------------------------------------------------------------|
| Package                  | H2O                 | H2O           | LightGBM      | LightGBM   | CatBoost   |
| Without oversampling AUC | 0.6982              | 0.7007        | 0.6882        | **0.7204** | **0.7222** |
| With oversampling AUC    | 0.6982              | **0.7008**    | **0.6893**    | 0.7195     | 0.6814     |


***
### Note:
Detailed analysis can be found in my [blog](https://jifuzhao.github.io/2018/03/20/lending-club.html). Feel free to read through it.

Copyright @ Jifu Zhao 2018
