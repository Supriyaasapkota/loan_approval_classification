
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report



df = pd.read_csv("loan_approval_cleaned.csv")


X = df.drop(["loan_approved", "points"], axis=1)
y = df["loan_approved"]



numerical_columns = X.select_dtypes(include="number").columns
categorical_columns = X.select_dtypes(include="str").columns



preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__solver": ["liblinear"]
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"]
        }
    ),

    "SVC": (
        SVC(probability=True),
        {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"]
        }
    )
}



trained_models = {}
results = []

best_score = 0
best_model = None
best_model_name = ""



for name, (model, parameters) in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    grid_search = GridSearchCV(
        pipeline,
        parameters,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

   
    best_estimator = grid_search.best_estimator_

    trained_models[name] = best_estimator

   
    y_pred = best_estimator.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

  
    results.append({
        "Model": name,
        "Best CV Accuracy": grid_search.best_score_,
        "Test Accuracy": accuracy,
        "Best Parameters": grid_search.best_params_
    })

    print("\nModel:", name)
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))

    
    if accuracy > best_score:
        best_score = accuracy
        best_model = best_estimator
        best_model_name = name


results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))



results_df.to_csv("model_comparison.csv", index=False)



joblib.dump(trained_models["Logistic Regression"], "logistic_model.joblib")
joblib.dump(trained_models["KNN"], "knn_model.joblib")
joblib.dump(trained_models["SVC"], "svc_model.joblib")



joblib.dump(best_model, "best_model.joblib")


print("\nBest Model:", best_model_name)
print("Best Test Accuracy:", best_score)
print("\nAll models saved successfully!")
