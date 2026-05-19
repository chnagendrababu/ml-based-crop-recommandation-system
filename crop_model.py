import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')


def train_model():
    """Train the crop recommendation model"""
    print("=" * 60)
    print("CROP RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("=" * 60)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    try:
        df = pd.read_csv("Crop_recommendation.csv")
        print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError:
        print("✗ Error: Crop_recommendation.csv not found!")
        return

    target = 'label'
    X = df.drop(target, axis=1)
    y = df[target]

    # Encode target labels
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train model
    print("\n[2/5] Training XGBoost model...")
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(np.unique(y_encoded)),
        random_state=42,
        eval_metric='mlogloss'
    )

    clf.fit(X_train, y_train)
    print("✓ Model training completed")

    # Evaluate model
    print("\n[3/5] Evaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Model Accuracy: {accuracy * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_)
    disp.plot(cmap="viridis", xticks_rotation=90)
    plt.title("Confusion Matrix - Test Data")
    plt.show()

    # Feature importance bar plot
    print("\n[4/5] Feature Importance Visualization...")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance, x='Importance', y='Feature', palette='coolwarm')
    plt.title("Feature Importance")
    plt.show()

    # Save models
    print("\n[5/5] Saving model files...")
    with open("xgb_crop_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le_y, f)

    with open("feature_names.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)

    print("✓ Model saved as 'xgb_crop_model.pkl'")
    print("✓ Encoder saved as 'label_encoder.pkl'")
    print("✓ Features saved as 'feature_names.pkl'")
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    return clf, le_y, X.columns


def predict_crop(input_dict=None):
    """Predict crop based on input parameters"""
    try:
        with open("xgb_crop_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            features = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        return None

    # Input
    if input_dict is None:
        print("\n" + "=" * 60)
        print("ENTER CROP PARAMETERS")
        print("=" * 60)
        user_input = []
        for feature in features:
            while True:
                try:
                    value = float(input(f"{feature:15s}: "))
                    user_input.append(value)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
        user_input = np.array(user_input).reshape(1, -1)
    else:
        try:
            user_input = np.array([input_dict[f] for f in features]).reshape(1, -1)
        except KeyError as e:
            print(f"Error: Missing feature {e} in input dictionary")
            return None

    # Prediction
    predicted_class = model.predict(user_input)
    predicted_proba = model.predict_proba(user_input)
    crop_name = encoder.inverse_transform(predicted_class)[0]
    confidence = predicted_proba[0][predicted_class[0]] * 100

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"🌾 Recommended Crop: {crop_name.upper()}")
    print(f"📊 Confidence: {confidence:.2f}%")

    # Top 3 crops bar chart
    print("\nTop 3 Recommended Crops:")
    top_3_idx = np.argsort(predicted_proba[0])[-3:][::-1]
    top_crops = [(encoder.inverse_transform([i])[0], predicted_proba[0][i] * 100) for i in top_3_idx]

    for i, (crop, prob) in enumerate(top_crops, 1):
        print(f"{i}. {crop:15s}: {prob:.2f}%")

    # Bar graph
    plt.figure(figsize=(6, 4))
    crops, probs = zip(*top_crops)
    sns.barplot(x=list(crops), y=list(probs), palette='crest')
    plt.title("Top 3 Crop Recommendations")
    plt.ylabel("Confidence (%)")
    plt.show()

    return crop_name


def main():
    """Main CLI"""
    print("\n" + "=" * 60)
    print("CROP RECOMMENDATION SYSTEM")
    print("=" * 60)
    print("\nOptions:")
    print("1. Train new model")
    print("2. Make prediction (interactive)")
    print("3. Make prediction (example data)")
    print("4. Exit")

    choice = input("\nEnter your choice (1-4): ")

    if choice == '1':
        train_model()
    elif choice == '2':
        predict_crop()
    elif choice == '3':
        sample_input = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        print("\nUsing example data:")
        for key, value in sample_input.items():
            print(f"  {key:15s}: {value}")
        predict_crop(sample_input)
    elif choice == '4':
        print("\nExiting... Goodbye!")
    else:
        print("\nInvalid choice. Please run again.")


if __name__ == "__main__":
    main()
