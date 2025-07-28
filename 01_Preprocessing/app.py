import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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

    pd.plotting.scatter_matrix(df, figsize=(10, 10), diagonal='kde')
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
    print(f"True Positive: {cm['True Positive']}")
    print(f"False Positive: {cm['False Positive']}")
    print(f"True Negative: {cm['True Negative']}")
    print(f"False Negative: {cm['False Negative']}")
    print(f"Accuracy: {accuracy_score(cm['True Positive'], cm['False Positive'], cm['True Negative'], cm['False Negative']):.4f}")
    print(f"Precision: {precision_score(cm['True Positive'], cm['False Positive']):.4f}")
    print(f"Recall: {recall_score(cm['True Positive'], cm['False Negative']):.4f}")
    print(f"F1 Score: {f1_score(cm['True Positive'], cm['False Positive'], cm['False Negative']):.4f}")

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['BloodPressure'], inplace=True)
    df.dropna(inplace=True)
    return df

def preprocess(df: pd.DataFrame, l2_normalizer: Normalizer, pca_glucose_insulin: PCA, min_max_glucose_insulin: MinMaxScaler, fit = False) -> pd.DataFrame:
    if fit:
        pca_glucose_insulin.fit(df[['Glucose', 'Insulin']])
    df['glucose_insulin'] = pca_glucose_insulin.transform(df[['Glucose', 'Insulin']])

    if fit:
        l2_normalizer.fit(df)
    df_columns = df.columns
    df = l2_normalizer.transform(df)
    df = pd.DataFrame(df, columns=df_columns)
    
    if fit:
        min_max_glucose_insulin.fit(df[['glucose_insulin']])
    df['glucose_insulin'] = min_max_glucose_insulin.transform(df[['glucose_insulin']])

    return df


if __name__ == "__main__":
    df = pd.read_csv('diabetes_dataset.csv')
    df = prepare(df)

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    l2_normalizer = Normalizer(norm='l2')
    pca_glucose_insulin = PCA(n_components=1)
    min_max_glucose_insulin = MinMaxScaler()
    knn = KNeighborsClassifier(n_neighbors=3)

    if 'analyze' in os.sys.argv[1:]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        
        X_train = preprocess(X_train, l2_normalizer, pca_glucose_insulin, min_max_glucose_insulin, fit=True)
        X_test = preprocess(X_test, l2_normalizer, pca_glucose_insulin, min_max_glucose_insulin)

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrix_analyze(cm)

        X_test['Predicted'] = y_test.reset_index(drop=True)
        X_test['Outcome'] = y_pred
        X_test.astype({'Predicted': 'int', 'Outcome': 'int'})

        print("Test set false positive:")
        print(X_test[X_test['Outcome'] == 0][X_test['Predicted'] == 1])
        print("Test set false negative:")
        print(X_test[X_test['Outcome'] == 1][X_test['Predicted'] == 0])

    if 'analyze-dataset' in os.sys.argv[1:]:
        X_train = preprocess(X, l2_normalizer, pca_glucose_insulin, min_max_glucose_insulin, fit=True)
        analyze_df(X_train)

    if 'send' in os.sys.argv[1:]:
        X_train = preprocess(X, l2_normalizer, pca_glucose_insulin, min_max_glucose_insulin, fit=True)
        knn.fit(X_train, y)

        df_pred = pd.read_csv('diabetes_app.csv')
        df_pred = prepare(df_pred)
        df_pred = preprocess(df_pred, l2_normalizer, pca_glucose_insulin, min_max_glucose_insulin)

        y_pred_app = knn.predict(df_pred)

        r = requests.post(
            url = "https://aydanomachado.com/mlclass/01_Preprocessing.php", 
            data = {
                'dev_key': "Crystal Gems",
                'predictions': pd.Series(y_pred_app).to_json(orient='values')
            })

        print(r.text)