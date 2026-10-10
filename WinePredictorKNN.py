import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler


# Project folder path
ProjectPath = Path(__file__).resolve().parent


def MarvellousClassifier(DataPath):

    Border = "-" * 40

    # Step 1 : Load the dataset from CSV file

    print(Border)
    print("Step 1 : Load the dataset from CSV file")
    print(Border)

    DatasetPath = Path(DataPath)

    if not DatasetPath.is_absolute():
        DatasetPath = ProjectPath / DatasetPath

    df = pd.read_csv(DatasetPath)

    print(Border)
    print("Some entries from dataset")
    print(df.head())
    print(Border)

    # Step 2 : Clean the dataset by removing empty rows

    print(Border)
    print("Step 2 : Clean the dataset by removing empty rows")
    print(Border)

    df.dropna(inplace=True)

    print("Total records : ", df.shape[0])
    print("Total columns : ", df.shape[1])
    print(Border)

    # Step 3 : Separate independent & dependent variables

    print(Border)
    print("Step 3 : Separate independent & dependent variables")
    print(Border)

    X = df.drop(columns=["Class"])
    Y = df["Class"]

    print("Shape of X : ", X.shape)
    print("Shape of Y : ", Y.shape)

    print(Border)
    print("Input columns : ", X.columns.to_list())
    print("Output column : Class")

    # Step 4 : Split the dataset for training & testing

    print(Border)
    print("Step 4 : Split the dataset for training & testing")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    print(Border)
    print("Information of training and testing data")

    print("X_train shape : ", X_train.shape)
    print("X_test shape : ", X_test.shape)
    print("Y_train shape : ", Y_train.shape)
    print("Y_test shape : ", Y_test.shape)

    print(Border)

    # Step 5 : Create Pipeline for Feature Scaling and KNN

    print(Border)
    print("Step 5 : Create Pipeline")
    print(Border)

    print("Pipeline contains StandardScaler and KNeighborsClassifier")

    # Step 6 : Explore the multiple values of K
    # Hyperparameter tuning using Cross Validation

    print(Border)
    print("Step 6 : Explore the multiple values of K")
    print(Border)

    accuracy_scores = []

    K_values = range(1, min(21, len(X_train) + 1))

    for k in K_values:

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])

        scores = cross_val_score(
            model,
            X_train,
            Y_train,
            cv=5,
            scoring="accuracy"
        )

        accuracy_scores.append(scores.mean())

    print(Border)
    print("Cross-validation accuracy report of K values")

    for k, score in zip(K_values, accuracy_scores):
        print(f"K = {k}, Accuracy = {score:.4f}")

    print(Border)

    # Step 7 : Plot graph of K vs Accuracy

    print(Border)
    print("Step 7 : Plot graph of K vs Accuracy")
    print(Border)

    plt.figure(figsize=(8, 5))
    plt.plot(K_values, accuracy_scores, marker="o")

    plt.title("K Values vs Cross-Validation Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.grid(True)
    plt.xticks(list(K_values))

    # Save graph inside Wine_Prediction folder
    # Save graph inside project folder
    GraphPath = ProjectPath / "K_vs_Accuracy.png"

    plt.savefig(GraphPath, dpi=300, bbox_inches="tight")
    print("Accuracy graph saved at :", GraphPath)

    plt.show()
    plt.close()

    # Step 8 : Find best value of K

    print(Border)
    print("Step 8 : Find best value of K")
    print(Border)

    best_k = list(K_values)[accuracy_scores.index(max(accuracy_scores))]

    print("Best value of K is : ", best_k)

    # Step 9 : Build final model using best value of K

    print(Border)
    print("Step 9 : Build final model using best value of K")
    print(Border)

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k))
    ])

    final_model.fit(X_train, Y_train)

    Y_pred = final_model.predict(X_test)

    print("Final Pipeline training completed")

    # Step 10 : Calculate final accuracy

    print(Border)
    print("Step 10 : Calculate final accuracy")
    print(Border)

    accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy of model is : ", accuracy * 100, "%")

    # Step 11 : Display confusion matrix

    print(Border)
    print("Step 11 : Display confusion matrix")
    print(Border)

    cm = confusion_matrix(Y_test, Y_pred)

    print(cm)

    ConfusionMatrixDisplay.from_predictions(
        Y_test,
        Y_pred,
        cmap="Blues"
    )

    plt.title("Wine Classification - Confusion Matrix")

    # Save confusion matrix image inside Wine_Prediction folder
    # Save confusion matrix image inside project folder
    MatrixPath = ProjectPath / "Confusion_Matrix.png"

    plt.savefig(MatrixPath, dpi=300, bbox_inches="tight")
    print("Confusion matrix image saved at :", MatrixPath)

    plt.show()
    plt.close()

    # Step 12 : Print classification report

    print(Border)
    print("Step 12 : Print classification report")
    print(Border)

    print(classification_report(Y_test, Y_pred, zero_division=0))


def main():

    Border = "-" * 40

    print(Border)
    print("Wine Classifier using KNN Pipeline")
    print(Border)

    # CSV file is located in the same folder as this Python file
    MarvellousClassifier("WinePredictor.csv")


if __name__ == "__main__":
    main()