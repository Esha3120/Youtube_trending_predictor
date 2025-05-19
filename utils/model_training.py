import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def train_trending_model(df, save_model=True):
    """
    Train and save XGBoost model
    Args:
        df: Fully processed DataFrame
        save_model: Whether to save the trained model
    Returns:
        model: Trained XGBoost model
        features: List of feature names
    """
    # Get all features except metadata and target
    features = [col for col in df.columns if col not in [
        'Title', 'Channel', 'Views', 'Likes', 'Comments',
        'Published At', 'Is_Trending', 'Region'
    ]]
    
    X = df[features]
    y = df['Is_Trending']
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )
    
    # Hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        XGBClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Save model and features
    if save_model:
        joblib.dump(best_model, 'models/youtube_trending_model.joblib')
        joblib.dump(features, 'models/youtube_features.joblib')
        print("Model and features saved to models/ directory")
    
    return best_model, features

def load_trending_model():
    """
    Load pre-trained model and features
    Returns:
        model: Pre-trained XGBoost model
        features: List of feature names
    """
    try:
        model = joblib.load('models/youtube_trending_model.joblib')
        features = joblib.load('models/youtube_features.joblib')
        return model, features
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Model files not found. Please train the model first."
        ) from e