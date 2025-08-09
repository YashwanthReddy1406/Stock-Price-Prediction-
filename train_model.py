import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load and preprocess data
df = pd.read_csv("AAPL.csv")
df['LogVolume'] = np.log1p(df['Volume'])
X = df[['Open', 'High', 'Low', 'LogVolume']]
y = df['Close']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Train MLP model
model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'mlp_model.pkl')
joblib.dump(scaler, 'mlp_scaler.pkl')

print("✅ MLP model and scaler saved.")
