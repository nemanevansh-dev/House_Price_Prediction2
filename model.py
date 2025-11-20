import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def create_sample_loan_data():
    """Create realistic sample loan data for training"""
    np.random.seed(65)
    n_samples = 1000

    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 65, n_samples),
        'Income': np.random.randint(20000, 120000, n_samples),
        'LoanAmount': np.random.randint(5000, 50000, n_samples),
        'LoanTerm': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'CreditScore': np.random.randint(300, 850, n_samples),
        'EmploymentStatus': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples, p=[0.7, 0.2, 0.1]),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.5, 0.1]),
        'PreviousDefaults': np.random.randint(0, 3, n_samples),
        'Defaulted': np.zeros(n_samples)
    }

    df = pd.DataFrame(data)

    # Create realistic default patterns
    default_prob = (
            (df['Income'] < 30000) * 0.3 +
            (df['LoanAmount'] > 30000) * 0.2 +
            (df['CreditScore'] < 600) * 0.4 +
            (df['EmploymentStatus'] == 'Unemployed') * 0.3 +
            (df['PreviousDefaults'] > 0) * 0.5 +
            np.random.random(n_samples) * 0.1
    )

    df['Defaulted'] = (default_prob > 0.5).astype(int)

    return df


# Load or create dataset
try:
    df = pd.read_csv('loan_default.csv')
    print("✅ loan_default.csv file loaded successfully!")
except FileNotFoundError:
    print("📊 Creating sample loan default dataset...")
    df = create_sample_loan_data()
    df.to_csv('loan_default.csv', index=False)
    print("✅ Sample dataset created and saved as 'loan_default.csv'")

# Data preprocessing
le = LabelEncoder()
categorical_columns = ['Gender', 'EmploymentStatus', 'MaritalStatus']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Feature and target selection
X = df[['Gender', 'Age', 'Income', 'LoanAmount', 'LoanTerm',
        'CreditScore', 'EmploymentStatus', 'MaritalStatus', 'PreviousDefaults']]
y = df['Defaulted']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65, stratify=y)

# Initialize and train model
dtc_model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=65
)

dtc_model.fit(X_train, y_train)

# Save the model
with open('dtc_trained_model.pkl', 'wb') as model_file:
    pickle.dump(dtc_model, model_file)

print("🎉 Model trained and saved successfully!")
print(f"Training Accuracy: {dtc_model.score(X_train, y_train):.4f}")
print(f"Testing Accuracy: {dtc_model.score(X_test, y_test):.4f}")