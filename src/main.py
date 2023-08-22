import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_random_state

class RFModel:
    def __init__(self, seed=12345):
        self.seed = seed
        check_random_state(self.seed)
        self.rf_model = RandomForestClassifier(random_state=self.seed)

    def load_data(self, path):
        self.df = pd.read_csv(path, sep=";")
        self.preprocess_data()

    def preprocess_data(self):
        self.df['over50'] = (self.df['age'] == '50plus').astype(int)
        self.df['below21'] = (self.df['age'] == 'below21').astype(int)
        self.df['b21_26'] = self.df['age'].isin(['21', '26']).astype(int)
        self.df['in30s'] = self.df['age'].isin(['31', '36']).astype(int)
        self.df['in40s'] = self.df['age'].isin(['41', '46']).astype(int)
        self.df.drop('age', axis=1, inplace=True)
        factorize_columns = self.df.columns[[0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
        df_encoded = pd.get_dummies(self.df, columns=factorize_columns)
        self.Y = df_encoded.pop('Y')
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(df_encoded, self.Y, test_size=0.3, random_state=self.seed)

    def fit(self):
        param_grid = {
            'n_estimators': [10, 100,500,1000],
            'max_leaf_nodes': [5,10, 20, 30, 50],
            'min_samples_leaf': [100,150, 300, 500]
        }
        grid_rf = GridSearchCV(self.rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3)
        grid_rf.fit(self.X_train, self.y_train)
        print(grid_rf.best_params_)
        self.best_rf_model = grid_rf.best_estimator_

    def predict(self):
        predicted_categories_train = self.best_rf_model.predict(self.X_train)
        predicted_categories_val = self.best_rf_model.predict(self.X_val)
        confusion_mat_train = confusion_matrix(y_true=self.y_train, y_pred=predicted_categories_train)
        confusion_mat_val = confusion_matrix(y_true=self.y_val, y_pred=predicted_categories_val)
        print(confusion_mat_train)
        print(confusion_mat_val)
        error_rate_train = 1 - np.trace(confusion_mat_train) / len(self.y_train)
        error_rate_val = 1 - np.trace(confusion_mat_val) / len(self.y_val)
        print("Train - Error Rate RF : ",error_rate_train)
        print("Validation - Error Rate RF : ",error_rate_val)
        return self.best_rf_model


if __name__ == "__main__":
    seed = 12345
    rf_model = RFModel(seed)
    rf_model.load_data("data.csv")  
    rf_model.fit()
    best_rf_model = rf_model.predict()
