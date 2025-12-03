# churn_model.py - Machine learning model for churn prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

class ChurnModel:
    """Class to train and use a churn prediction model"""
    
    def __init__(self, model_type='logistic'):
        """Initialize the model
        
        Args:
            model_type (str): Type of model to use ('logistic', 'rf', 'gb', 'xgb')
        """
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.features = None
        self.target = 'Churn'
        
        # Define model directory
        self.model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.model_dir, f'churn_{model_type}.pkl')
        self.preprocessor_path = os.path.join(self.model_dir, f'preprocessor_{model_type}.pkl')
    
    def preprocess_data(self, df):
        """Preprocess the data for training or prediction
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            DataFrame: Preprocessed data
        """
        # Define features if not already defined
        if self.features is None:
            # Exclude target and non-predictive columns
            exclude_cols = [self.target, 'customerID', 'customer_id']
            self.features = [col for col in df.columns if col not in exclude_cols]
        
        # Create preprocessor if not already created
        if self.preprocessor is None:
            # Identify numeric and categorical features
            numeric_features = df[self.features].select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = df[self.features].select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Define preprocessing for numeric features
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Define preprocessing for categorical features
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Combine preprocessing steps
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
        
        return df
    
    def train(self, df):
        """Train the model
        
        Args:
            df (DataFrame): Training data with target column
            
        Returns:
            dict: Training metrics
        """
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Split features and target
        X = df[self.features]
        y = df[self.target].map({'Yes': 1, 'No': 0})  # Convert to binary
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit preprocessor
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Create and train model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(random_state=42)
        elif self.model_type == 'gb':
            self.model = GradientBoostingClassifier(random_state=42)
        elif self.model_type == 'xgb':
            self.model = xgb.XGBClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train_processed, y_train)
        
        # Evaluate model
        X_test_processed = self.preprocessor.transform(X_test)
        y_pred = self.model.predict(X_test_processed)
        y_prob = self.model.predict_proba(X_test_processed)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob)
        }
        
        # Save model and preprocessor
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.preprocessor, self.preprocessor_path)
        
        return metrics
    
    def load(self):
        """Load a trained model and preprocessor
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, df):
        """Make predictions on new data
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            DataFrame: Input data with predictions added
        """
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Make sure model is loaded
        if self.model is None:
            self.load()
        
        # Extract features
        X = df[self.features]
        
        # Preprocess and predict
        X_processed = self.preprocessor.transform(X)
        df['churn_probability'] = self.model.predict_proba(X_processed)[:, 1]
        
        # Add risk category
        df['risk_category'] = pd.cut(
            df['churn_probability'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        return df

# Example usage with synthetic data
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic telecom customer data for demonstration"""
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f'CUST{i:04d}' for i in range(1, n_samples + 1)]
    
    # Generate features
    tenure = np.random.randint(1, 73, n_samples)  # 1-72 months
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2])
    online_security = np.random.choice(['Yes', 'No'], n_samples)
    tech_support = np.random.choice(['Yes', 'No'], n_samples)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.5, 0.2])
    monthly_charges = np.random.uniform(20, 120, n_samples)
    total_charges = monthly_charges * tenure * (np.random.uniform(0.9, 1.1, n_samples))
    
    # Generate target (churn) based on features
    churn_prob = 0.2  # Base churn probability
    
    # Adjust based on contract type
    contract_factor = np.where(contract == 'Month-to-month', 0.2, 
                              np.where(contract == 'One year', -0.1, -0.15))
    
    # Adjust based on tenure
    tenure_factor = -0.2 * (tenure / 72)  # Longer tenure reduces churn
    
    # Adjust based on services
    service_factor = np.where(online_security == 'No', 0.1, -0.05) + \
                    np.where(tech_support == 'No', 0.1, -0.05)
    
    # Adjust based on charges
    charge_factor = 0.1 * (monthly_charges - 70) / 50  # Higher charges increase churn
    
    # Calculate final probability
    final_prob = churn_prob + contract_factor + tenure_factor + service_factor + charge_factor
    final_prob = np.clip(final_prob, 0.01, 0.99)  # Ensure probability is between 0.01 and 0.99
    
    # Generate churn based on probability
    churn = np.random.binomial(1, final_prob, n_samples)
    churn = np.where(churn == 1, 'Yes', 'No')
    
    # Create DataFrame
    df = pd.DataFrame({
        'customerID': customer_ids,
        'tenure': tenure,
        'contract': contract,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'InternetService': internet_service,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    
    return df

def example_usage():
    # Generate synthetic data
    df = generate_synthetic_data(1000)
    
    # Create and train model
    model = ChurnModel(model_type='logistic')
    metrics = model.train(df)
    print(f"Training metrics: {metrics}")
    
    # Generate test data
    test_df = generate_synthetic_data(100)
    
    # Make predictions
    results = model.predict(test_df)
    
    # Print sample results
    print("\nSample predictions:")
    sample_cols = ['customerID', 'tenure', 'contract', 'MonthlyCharges', 'churn_probability', 'risk_category']
    print(results[sample_cols].head())
    
    # Calculate accuracy on test data
    y_true = test_df['Churn'].map({'Yes': 1, 'No': 0})
    y_pred = (results['churn_probability'] > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    example_usage()