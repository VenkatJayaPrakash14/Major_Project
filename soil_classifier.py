import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV

class SoilClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.class_labels = None
    
    def preprocess_data(self, df):
        """Preprocess the dataset by encoding labels and scaling features"""
        # Encode labels
        df['encoded_label'] = self.label_encoder.fit_transform(df['label']) + 1
        self.class_labels = self.label_encoder.classes_
        
        # Scale pH values
        df['ph'] = self.scaler.fit_transform(df[['ph']])
        
        # Prepare features and target
        X = df.drop(['label', 'encoded_label'], axis=1)
        y = df['encoded_label']
        self.feature_columns = list(X.columns)
        
        return X, y
    
    def train_model(self, X, y, test_size=0.10):
        """Train the Random Forest model with hyperparameter tuning"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=42
        )
        
        # Define hyperparameter search space
        # Removed potentially problematic parameters
        param_grid = {
            'n_estimators': np.arange(50, 200),
            'max_depth': [None] + list(np.arange(4, 100)),
            'min_samples_split': np.arange(2, 20),
            'min_samples_leaf': np.arange(1, 10)
        }
        
        # Create base model with basic parameters
        rf = RandomForestClassifier(random_state=42)
        
        # Perform randomized search for hyperparameter tuning
        rscv_model = RandomizedSearchCV(
            rf, 
            param_grid, 
            n_iter=20,  # Reduced number of iterations for faster training
            cv=5, 
            n_jobs=-1,
            random_state=42
        )
        rscv_model.fit(X_train, y_train)
        
        # Get the best model
        self.model = rscv_model.best_estimator_
        
        # Evaluate the model
        self._evaluate_model(X_train, X_test, y_train, y_test)
        
        return self.model

    def _evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate model performance on both training and test sets"""
        # Test set evaluation
        y_pred_test = self.model.predict(X_test)
        print("Test Set Evaluation:")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
        print("\nClassification Report:\n", classification_report(y_test, y_pred_test))
        
        # Training set evaluation
        y_pred_train = self.model.predict(X_train)
        print("\nTraining Set Evaluation:")
        print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
        print("\nClassification Report:\n", classification_report(y_train, y_pred_train))
    
    def predict_crops(self, features_dict, threshold=0.15):
        """Predict suitable crops based on soil features"""
        # Create a DataFrame with the provided features
        features_df = pd.DataFrame([features_dict])[self.feature_columns]
        
        # Get probability predictions
        probabilities = self.model.predict_proba(features_df)[0]
        
        # Create crop probability dictionary
        crop_probabilities = {}
        for crop_label, probability in zip(self.class_labels, probabilities):
            crop_probabilities[crop_label] = probability
        
        # Sort and filter based on threshold
        sorted_crops = sorted(crop_probabilities.items(), key=lambda x: x[1], reverse=True)
        recommended_crops = [crop for crop, prob in sorted_crops if prob >= threshold]
        
        return recommended_crops
    
    def save_model(self, model_path='random_forest_model.pkl', metadata_path='model_metadata.json'):
        """Save the trained model and all necessary metadata"""
        # Save the model
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)
        
        # Save metadata including feature columns, class labels, and scaler
        metadata = {
            'feature_columns': self.feature_columns,
            'class_labels': self.class_labels.tolist() if self.class_labels is not None else None,
            'scaler_params': {
                'scale_': self.scaler.scale_.tolist(),
                'min_': self.scaler.min_.tolist(),
            }
        }
        
        with open(metadata_path, 'w') as file:
            json.dump(metadata, file)
    
    @classmethod
    def load_model(cls, model_path='random_forest_model.pkl', metadata_path='model_metadata.json'):
        """Load a saved model and all metadata"""
        classifier = cls()
        
        # Load the model
        with open(model_path, 'rb') as file:
            classifier.model = pickle.load(file)
        
        # Load metadata
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
            classifier.feature_columns = metadata['feature_columns']
            classifier.class_labels = np.array(metadata['class_labels'])
            
            # Reconstruct the scaler
            classifier.scaler = MinMaxScaler()
            classifier.scaler.scale_ = np.array(metadata['scaler_params']['scale_'])
            classifier.scaler.min_ = np.array(metadata['scaler_params']['min_'])
            classifier.scaler.data_min_ = classifier.scaler.min_
            classifier.scaler.data_max_ = classifier.scaler.min_ + 1/classifier.scaler.scale_
            classifier.scaler.data_range_ = classifier.scaler.data_max_ - classifier.scaler.data_min_
        
        return classifier