import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def training_model_to_predict_insulin(df: pd.DataFrame) -> LinearRegression:
    df = df.dropna(subset=['Glucose', 'Insulin'])
    X = df['Glucose'].values.reshape(-1, 1)
    y = df['Insulin']

    model = LinearRegression()
    model.fit(X, y)

    return model

def training_model_to_predict_skinthickness(df: pd.DataFrame) -> LinearRegression:
    df = df.dropna(subset=['BMI', 'SkinThickness'])
    X = df['BMI'].values.reshape(-1, 1)
    y = df['SkinThickness']

    model = LinearRegression()
    model.fit(X, y)

    return model

def predict_insulin(model: LinearRegression, df: pd.DataFrame) -> pd.Series:
    df = df.dropna(subset=['Glucose'])
    X = df['Glucose'].values.reshape(-1, 1)
    predictions = model.predict(X)
    return pd.Series(predictions, index=df.index)

def predict_skinthickness(model: LinearRegression, df: pd.DataFrame) -> pd.Series:
    df = df.dropna(subset=['BMI'])
    X = df['BMI'].values.reshape(-1, 1)
    predictions = model.predict(X)
    return pd.Series(predictions, index=df.index)

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

def get_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_limit) | (df[column] > upper_limit)]
    
    return outliers

def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    without_outliers = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)].copy()
    
    return without_outliers

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

    df.dropna(inplace=True)

    df = set_columns_type(df)
    df['Outcome'] = df['Outcome'].astype(int)

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    return X_test, y_test

def train_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('diabetes_dataset.csv')
    X_test, _ = test_dataset()
    df = df.drop(X_test.index)

    # Enrich the dataset
    insulin_model = training_model_to_predict_insulin(df)
    df['Insulin'] = df['Insulin'].fillna(predict_insulin(insulin_model, df))
    skinthickness_model = training_model_to_predict_skinthickness(df)
    df['SkinThickness'] = df['SkinThickness'].fillna(predict_skinthickness(skinthickness_model, df))

    df = remove_outliers(df, 'DiabetesPedigreeFunction')
    df.dropna(inplace=True)

    df = set_columns_type(df)
    df['Outcome'] = df['Outcome'].astype(int)

    X_train = df.drop(columns=['Outcome'])
    y_train = df['Outcome']
    return X_train, y_train

def preprocess(
        df: pd.DataFrame, 
        l2_normalizer: Normalizer, 
        pca_glucose_insulin: PCA, 
        pca_skinthickness_bmi: PCA, 
        min_max_glucose_insulin: MinMaxScaler, 
        min_max_skinthickness_bmi: MinMaxScaler, 
        min_max_dpf: MinMaxScaler, 
        fit = False
        ) -> pd.DataFrame:
    if fit:
        pca_glucose_insulin.fit(df[['Glucose', 'Insulin']])
    df['glucose_insulin'] = pca_glucose_insulin.transform(df[['Glucose', 'Insulin']])
    df.drop(columns=['Glucose', 'Insulin'], inplace=True)

    if fit:
        pca_skinthickness_bmi.fit(df[['SkinThickness', 'BMI']])
    df['skinthickness_bmi'] = pca_skinthickness_bmi.transform(df[['SkinThickness', 'BMI']])
    df.drop(columns=['SkinThickness', 'BMI'], inplace=True)

    if fit:
        l2_normalizer.fit(df)
    df_columns = df.columns
    df = l2_normalizer.transform(df)
    df = pd.DataFrame(df, columns=df_columns)

    if fit:
        min_max_glucose_insulin.fit(df[['glucose_insulin']])
    df['glucose_insulin'] = min_max_glucose_insulin.transform(df[['glucose_insulin']])

    if fit:
        min_max_skinthickness_bmi.fit(df[['skinthickness_bmi']])
    df['skinthickness_bmi'] = min_max_skinthickness_bmi.transform(df[['skinthickness_bmi']])

    if fit:
        min_max_dpf.fit(df[['DiabetesPedigreeFunction']])
    df['DiabetesPedigreeFunction'] = min_max_dpf.transform(df[['DiabetesPedigreeFunction']])

    return df


if __name__ == "__main__":
    df = pd.read_csv('diabetes_dataset.csv')
    X_test, y_test = test_dataset()
    X_train, y_train = train_dataset()

    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    # print("Train set shape:", X_train.shape)
    # print("Test set shape:", X_test.shape)

    l2_normalizer = Normalizer(norm='l2')
    pca_glucose_insulin = PCA(n_components=1)
    pca_skinthickness_bmi = PCA(n_components=1)
    min_max_glucose_insulin = MinMaxScaler()
    min_max_skinthickness_bmi = MinMaxScaler()
    min_max_dpf = MinMaxScaler()
    
    X_train = preprocess(X_train, l2_normalizer, pca_glucose_insulin, pca_skinthickness_bmi, min_max_glucose_insulin, min_max_skinthickness_bmi, min_max_dpf, fit=True)
    X_test = preprocess(X_test, l2_normalizer, pca_glucose_insulin, pca_skinthickness_bmi, min_max_glucose_insulin, min_max_skinthickness_bmi, min_max_dpf)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    if 'versus-test' in os.sys.argv[1:]:
        y_pred = knn.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        confusion_matrix_analyze(cm)

        X_test['Predicted'] = y_test
        X_test['Outcome'] = y_pred
        X_test.astype({'Predicted': int, 'Outcome': int})

        print("Test set false positive:")
        print(X_test[X_test['Outcome'] == 0][X_test['Predicted'] == 1])
        print("Test set false negative:")
        print(X_test[X_test['Outcome'] == 1][X_test['Predicted'] == 0])

    if 'analyze' in os.sys.argv[1:]:
        X_train['Outcome'] = y_train
        analyze_df(X_train)

    if 'versus-production' in os.sys.argv[1:]:
        df_pred = pd.read_csv('diabetes_app.csv')

        df_pred = set_columns_type(df_pred)
        df_pred = preprocess(df_pred, l2_normalizer, pca_glucose_insulin, pca_skinthickness_bmi, min_max_glucose_insulin, min_max_skinthickness_bmi, min_max_dpf)

        y_pred_app = knn.predict(df_pred)

        r = requests.post(
            url = "https://aydanomachado.com/mlclass/01_Preprocessing.php", 
            data = {
                'dev_key': "Crystal Gems",
                'predictions': pd.Series(y_pred_app).to_json(orient='values')
            })

        print(r.text)