import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Função para plotar a matriz de scatter plots com cores condicionais
def plot_correlation_with_color(df, real_cols, bin_cols):
    n = len(real_cols)
    fig, axes = plt.subplots(n, n, figsize=(50, 50))
    
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: mostrar o nome da variável
                ax.text(0.5, 0.5, real_cols[i], ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Scatter plot para colunas i e j
                x_col = real_cols[j]
                y_col = real_cols[i]
                
                # Verificar se os valores das colunas binárias são iguais
                same_bin = (df[bin_cols[0]] == df[bin_cols[1]])
                
                # Plotar pontos onde bin1 == bin2 (mesma cor, ex: azul)
                ax.scatter(
                    df[x_col][same_bin],
                    df[y_col][same_bin],
                    color='blue',
                    label=f'{bin_cols[0]} == {bin_cols[1]}',
                    alpha=0.6
                )
                
                # Plotar pontos onde bin1 != bin2 (outra cor, ex: vermelho)
                ax.scatter(
                    df[x_col][~same_bin],
                    df[y_col][~same_bin],
                    color='red',
                    label=f'{bin_cols[0]} != {bin_cols[1]}',
                    alpha=0.6
                )
                
                # Configurar eixos
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_xticks([])
                ax.set_yticks([])
        
    # Ajustar layout
    plt.tight_layout()
    # plt.legend(handles=[
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=f'{bin_cols[0]} == {bin_cols[1]}'),
    #     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=f'{bin_cols[0]} != {bin_cols[1]}')
    # ], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # plt.suptitle(f"Scatter Matrix (Colorized by {bin_cols[0]} vs {bin_cols[1]})", y=1.02)
    plt.show()

def analyze_df(df: pd.DataFrame) -> None:
    for column in df.select_dtypes(include=['bool']).columns:
        df[column] = df[column].astype(int)

    print(df.shape)
    print(df.columns.tolist())
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.corr())

    df.boxplot()
    plt.show()

    pd.plotting.scatter_matrix(df, figsize=(10, 10), diagonal='kde', c=df['Outcome'])
    plt.show()

    df.hist(bins=30, figsize=(10, 10), layout=(3, 3))
    plt.tight_layout()
    plt.show()

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))

    return {
        'True Positive': true_positive,
        'False Positive': false_positive,
        'True Negative': true_negative,
        'False Negative': false_negative
    }

def accuracy_score(a, b, c, d) -> float:
    return (a + c) / (a + b + c + d) if (a + b + c + d) > 0 else 0

def precision_score(a, b) -> float:
    return a / (a + b) if (a + b) > 0 else 0

def recall_score(a, d) -> float:
    return a / (a + d) if (a + d) > 0 else 0

def f1_score(a, b, c) -> float:
    precision = precision_score(a, b)
    recall = recall_score(a, c)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def confusion_matrix_analyze(cm: dict) -> None:
    print("Confusion Matrix Analysis:")
    print(f"\n\t\tPredicted Positive\tPredicted Negative")
    print(f"Actual Positive\t{cm['True Positive']}\t\t\t{cm['False Negative']}")
    print(f"Actual Negative\t{cm['False Positive']}\t\t\t{cm['True Negative']}")
    print("\nMetrics:")
    print(f"  - Accuracy:  {accuracy_score(cm['True Positive'], cm['False Positive'], cm['True Negative'], cm['False Negative']):.4f}")
    print(f"  - Precision: {precision_score(cm['True Positive'], cm['False Positive']):.4f}")
    print(f"  - Recall:    {recall_score(cm['True Positive'], cm['False Negative']):.4f}")
    print(f"  - F1 Score:  {f1_score(cm['True Positive'], cm['False Positive'], cm['False Negative']):.4f}")

def set_columns_type(df: pd.DataFrame) -> pd.DataFrame:
    df['Pregnancies'] = df['Pregnancies'].astype(int)
    df['Glucose'] = df['Glucose'].astype(int)
    df['BloodPressure'] = df['BloodPressure'].astype(int)
    df['SkinThickness'] = df['SkinThickness'].astype(int)
    df['Insulin'] = df['Insulin'].astype(float)
    df['BMI'] = df['BMI'].astype(float)
    df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].astype(float)
    df['Age'] = df['Age'].astype(int)
    return df

def test_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('diabetes_dataset.csv')

    df = df.dropna()

    df = set_columns_type(df)
    df['Outcome'] = df['Outcome'].astype(int)

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.9, random_state=10)
    return X_test, y_test

def train_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('diabetes_dataset.csv')
    X_test, _ = test_dataset()
    df = df.drop(X_test.index)

    df = df.dropna()

    df = set_columns_type(df)
    df['Outcome'] = df['Outcome'].astype(int)

    X_train = df.drop(columns=['Outcome'])
    y_train = df['Outcome']
    return X_train, y_train

def preprocess(
        df: pd.DataFrame, 
        ss_Pregnancies: StandardScaler,
        ss_SkinThickness: StandardScaler,
        ss_Insulin: StandardScaler,
        ss_BMI: StandardScaler,
        ss_Age: StandardScaler,
        fit = False
        ) -> pd.DataFrame:
    if fit:
        ss_Pregnancies.fit(df[['Pregnancies']])
    df['Pregnancies'] = ss_Pregnancies.transform(df[['Pregnancies']])

    df['Glucose'] = df['Glucose'] * 0.5

    df['BloodPressure'] = df['BloodPressure'] + df['BloodPressure'].quantile(0.5)

    df['BloodPressure'] = df['BloodPressure'] * 2

    if fit:
        ss_SkinThickness.fit(df[['SkinThickness']])
    df['SkinThickness'] = ss_SkinThickness.transform(df[['SkinThickness']])

    if fit:
        ss_Insulin.fit(df[['Insulin']])
    df['Insulin'] = ss_Insulin.transform(df[['Insulin']])

    if fit:
        ss_BMI.fit(df[['BMI']])
    df['BMI'] = ss_BMI.transform(df[['BMI']])

    df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'] * 25

    if fit:
        ss_Age.fit(df[['Age']])
    df['Age'] = ss_Age.transform(df[['Age']])

    return df

def prepare_algorithm() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, KNeighborsClassifier, StandardScaler, StandardScaler, StandardScaler, StandardScaler, StandardScaler]:
    X_test, y_test = test_dataset()
    X_train, y_train = train_dataset()

    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    ss_Pregnancies = StandardScaler()
    ss_SkinThickness = StandardScaler()
    ss_Insulin = StandardScaler()
    ss_BMI = StandardScaler()
    ss_Age = StandardScaler()
    
    X_train = preprocess(X_train, ss_Pregnancies, ss_SkinThickness, ss_Insulin, ss_BMI, ss_Age, fit=True)
    X_test = preprocess(X_test, ss_Pregnancies, ss_SkinThickness, ss_Insulin, ss_BMI, ss_Age)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    return X_train, y_train, X_test, y_test, knn, ss_Pregnancies, ss_SkinThickness, ss_Insulin, ss_BMI, ss_Age

if __name__ == "__main__":
    if 'versus-test' in os.sys.argv[1:]:
        X_train, y_train, X_test, y_test, knn, _, _, _, _, _ = prepare_algorithm()
        y_pred = knn.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrix_analyze(cm)

        X_test['Predicted'] = y_test
        X_test['Outcome'] = y_pred
        X_test.astype({'Predicted': int, 'Outcome': int})

        # print("Test set false positive:")
        # print(X_test[X_test['Outcome'] == 0][X_test['Predicted'] == 1])
        # print("Test set false negative:")
        # print(X_test[X_test['Outcome'] == 1][X_test['Predicted'] == 0])

        # plot_correlation_with_color(X_test, 
        #                             real_cols=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        #                             bin_cols=['Outcome', 'Predicted'])

    if 'analyze-train' in os.sys.argv[1:]:
        X_train, y_train, _, _, _, _, _, _, _, _ = prepare_algorithm()
        X_train['Outcome'] = y_train
        analyze_df(X_train)

    if 'analyze-test' in os.sys.argv[1:]:
        _, _, X_test, y_test, _, _, _, _, _, _ = prepare_algorithm()
        X_test['Outcome'] = y_test
        analyze_df(X_test)

    if 'versus-production' in os.sys.argv[1:]:
        _, _, _, _, knn, ss_Pregnancies, ss_SkinThickness, ss_Insulin, ss_BMI, ss_Age = prepare_algorithm()
        df_pred = pd.read_csv('diabetes_app.csv')

        df_pred = set_columns_type(df_pred)
        df_pred = preprocess(df_pred, ss_Pregnancies, ss_SkinThickness, ss_Insulin, ss_BMI, ss_Age)

        y_pred_app = knn.predict(df_pred)

        r = requests.post(
            url = "https://aydanomachado.com/mlclass/01_Preprocessing.php", 
            data = {
                'dev_key': "Crystal Gems",
                'predictions': pd.Series(y_pred_app).to_json(orient='values')
            })

        print(r.text)
