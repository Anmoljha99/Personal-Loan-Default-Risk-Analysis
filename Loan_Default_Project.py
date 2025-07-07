# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#Load Dataset
df = pd.read_csv("loan_default_cleaned_final.csv")

# Preview Data
print("First 5 Rows:")
print(df.head())
print("\nData Summary:")
print(df.describe())

#1.Loan Default Distribution
sns.countplot(data=df, x="Defaulted")
plt.title("Loan Default Distribution (0 = No, 1 = Yes)")
plt.show()

# 2.Applicant Income vs Default
sns.boxplot(data=df, x="Defaulted", y="ApplicantIncome")
plt.title("Applicant Income vs Loan Default")
plt.show()

# 3.Credit History vs Default
sns.barplot(data=df, x="Credit_History", y="Defaulted")
plt.title("Default Rate by Credit History")
plt.xlabel("Credit History (0 = Bad, 1 = Good)")
plt.ylabel("Proportion Defaulted")
plt.show()

# 4.Loan Amount vs Term (Colored by Default)
sns.scatterplot(data=df, x="Loan_Amount_Term", y="LoanAmount", hue="Defaulted")
plt.title("Loan Amount vs Term (by Default Status)")
plt.xlabel("Loan Term (Months)")
plt.ylabel("Loan Amount (INR)")
plt.show()

# 5.Education vs Default
sns.barplot(data=df, x="Education", y="Defaulted")
plt.title("Default Rate by Education")
plt.xticks(rotation=15)
plt.ylabel("Proportion Defaulted")
plt.show()

# 6.Property Area vs Default
sns.barplot(data=df, x="Property_Area", y="Defaulted")
plt.title("Default Rate by Property Area")
plt.ylabel("Proportion Defaulted")
plt.show()

#Logistic Regression Model

# Features and Target
X = df[['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
y = df['Defaulted']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

#Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
