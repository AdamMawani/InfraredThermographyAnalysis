from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from flask import Flask, request, jsonify

# fetch dataset
infrared_thermography_temperature = fetch_ucirepo(id=925)

# data (as pandas dataframes)
X = infrared_thermography_temperature.data.features
y = infrared_thermography_temperature.data.targets

# metadata
print(infrared_thermography_temperature.metadata)

# variable information
print(infrared_thermography_temperature.variables)

# Step 1: Data Preprocessing
# Convert data to pandas DataFrame
df_X = pd.DataFrame(X, columns=infrared_thermography_temperature.variables[3:]["name"])
df_y = pd.DataFrame(y, columns=["aveOralF"])  # Target variable

# Step 2: Exploratory Data Analysis (EDA)
# Example EDA - Pairplot for some variables
sample_variables = ['Age', 'Humidity', 'T_atm', 'Distance', 'T_offset1']
sample_df = df_X[sample_variables].copy()
sample_df['aveOralF'] = df_y['aveOralF']
plt.figure(figsize=(10, 8))
pd.plotting.scatter_matrix(sample_df, figsize=(12, 10))
plt.show()

# Step 3: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Building the Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Training the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Step 7: Model Evaluation
train_loss = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Train Loss: {train_loss}, Test Loss: {test_loss}")

# Step 8: Flask Demo
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = pd.DataFrame(data, index=[0])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'predicted_temperature': prediction[0][0]})

if __name__ == '__main__':
    app.run(debug=True)