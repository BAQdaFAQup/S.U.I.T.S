import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


file_path = '/content/Gold_Scaled _The_Rainforest_Chronicles.csv'
data = pd.read_csv(file_path)


selected_columns = [
    'Year Last Documented', 'NY Listing Status', 'Federal Listing Status',
    'State Conservation Rank', 'Global Conservation Rank', 'Distribution Status'
]
df = data[selected_columns]

df = df.dropna()

df = df.sample(n=7000, random_state=42) 

label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


df['Year_Status_Interaction'] = df['Year Last Documented'] * df['NY Listing Status']
df['Federal_State_Rank_Sum'] = df['Federal Listing Status'] + df['State Conservation Rank']


X = df.drop('Global Conservation Rank', axis=1)
y = df['Global Conservation Rank']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = GradientBoostingClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

unique_classes = sorted(set(y_test) | set(y_pred))
target_names = label_encoders['Global Conservation Rank'].inverse_transform(unique_classes)
report = classification_report(y_test, y_pred, labels=unique_classes, target_names=target_names, zero_division=0)
print("Classification Report:\n")
print(report)

example_data_point = pd.DataFrame([[20, 2, 0, 3, 1, 20*2, 0+3]], columns=X.columns)
predicted_risk = model.predict(example_data_point)
predicted_risk_label = label_encoders['Global Conservation Rank'].inverse_transform(predicted_risk)
print(f"The predicted conservation status is: {predicted_risk_label[0]}")
