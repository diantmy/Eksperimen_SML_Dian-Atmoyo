import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(df):
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'].fillna(
        df['TotalCharges'].median(),
        inplace=True
    )
    return df

def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def remove_irrelevant_columns(df):
    df.drop('customerID', axis=1, inplace=True)
    return df

def handle_outliers(df):
    numerical_cols = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges'
    ]
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[
            (df[col] >= lower) &
            (df[col] <= upper)
        ]
    return df

def create_binning(df):
    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0,12,24,48,72],
        labels=[
            '0-12',
            '13-24',
            '25-48',
            '49-72'
        ]
    )
    return df

def encode_data(df):
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
    df = pd.get_dummies(
        df,
        drop_first=True
    )
    return df


def scale_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )
    return X_scaled, y

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def save_data(X, y, output_path):
    final_df = pd.concat(
        [
            pd.DataFrame(X),
            pd.Series(y, name='Churn')
        ],
        axis=1
    )
    final_df.to_csv(
        output_path,
        index=False
    )

def main():
    df = load_data(
        '../WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = remove_irrelevant_columns(df)
    df = handle_outliers(df)
    df = create_binning(df)
    df = encode_data(df)
    X, y = scale_data(df)
    X, y = apply_smote(X, y)
    save_data(
        X,
        y,
        'telco_preprocessed.csv'
    )
    print("Preprocessing selesai.")

if __name__ == "__main__":
    main()
