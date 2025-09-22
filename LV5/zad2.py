import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv('./occupancy_processed.csv')
X = data[['S3_Temp', 'S5_CO2']]
y = data['Room_Occupancy_Count']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print("Matrica zabune:\n", confusion_matrix(y_test, y_pred))
print("Točnost klasifikacije:", accuracy_score(y_test, y_pred))
print("Detaljna evaluacija:\n", classification_report(y_test, y_pred))
