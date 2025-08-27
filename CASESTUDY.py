import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Simulate dataset based on Peña-Calvin et al. (2023)
np.random.seed(42)
data = {
    'domain': np.random.choice(['Technology', 'Finance', 'Social', 'Entertainment', 'Legal'], 40),
    'purpose': np.random.choice(['Services', 'Granting', 'Peer Production', 'Funding'], 40),
    'scope_blockchain': np.random.choice([0, 1], 40),
    'scope_web2': np.random.choice([0, 1], 40),
    'scope_physical': np.random.choice([0, 1], 40),
    'scope_web2_physical': np.random.choice([0, 1], 40),
    'voting_participation': np.random.choice(['Universal', 'Restricted'], 40),
    'vote_weight': np.random.choice(['Token-Owned', 'Token-Deposited', 'Uniform'], 40),
    'archetype': np.random.choice(['GEN-1', 'GEN-2', 'GEN-3', 'GEN-4', 'GEN-5', 'GEN-6', 'GEN-OTHER'], 40)
}
df = pd.DataFrame(data)

# Preprocessing
try:
    X = df[['domain', 'purpose', 'scope_blockchain', 'scope_web2', 'scope_physical', 'scope_web2_physical', 
            'voting_participation', 'vote_weight']]
    y = df['archetype']

    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X[['domain', 'purpose', 'voting_participation', 'vote_weight']])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out())
    X_final = pd.concat([X_encoded_df, X[['scope_blockchain', 'scope_web2', 'scope_physical', 'scope_web2_physical']].reset_index(drop=True)], axis=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Cross-validation
    cv_scores = cross_val_score(model, X_final, y, cv=5, scoring='f1_weighted')
    print(f"Cross-validation F1-score: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

    # Feature importance plot
    feature_importance = pd.Series(model.feature_importances_, index=X_final.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title('Feature Importance in DAO Archetype Classification')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

except Exception as e:
    print(f"An error occurred: {str(e)}")