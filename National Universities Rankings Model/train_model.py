import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
import joblib

# Load and clean your dataset
df = pd.read_csv("National Universities Rankings.csv")
df.columns = df.columns.str.strip()
df["Tuition and fees"] = df["Tuition and fees"].replace('[\$,]', '', regex=True).astype(float)
df["Undergrad Enrollment"] = df["Undergrad Enrollment"].replace(',', '', regex=True).astype(float)

# Prepare features and labels
X = df[["Tuition and fees", "Undergrad Enrollment"]]
y = df["Rank"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(16, input_dim=2, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Custom callback to stop on val_mae increase or no improvement
class StopOnValMAEIncrease(Callback):
    def __init__(self, delta=0.01):
        super().__init__()
        self.prev_val_mae = None
        self.val_mae_history = []
        self.train_mae_history = []
        self.delta = delta  # minimum improvement

    def on_epoch_end(self, epoch, logs=None):
        current_val_mae = logs.get("val_mae")
        current_train_mae = logs.get("mae")
        self.val_mae_history.append(current_val_mae)
        self.train_mae_history.append(current_train_mae)

        # Stop if val_mae increases or doesn't improve by at least 'delta' value
        if self.prev_val_mae is not None and current_val_mae >= self.prev_val_mae + self.delta:
            print(f"\nEarly stopping at epoch {epoch+1}: val_mae increased or didn't improve enough ({self.prev_val_mae:.4f} → {current_val_mae:.4f})")
            self.model.stop_training = True

        self.prev_val_mae = current_val_mae

# Initialize the callback
stop_callback = StopOnValMAEIncrease(delta=0.01)

# Train the model
model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=8,
    verbose=1,
    callbacks=[stop_callback]
)

# Save model and scaler
model.save("university_rank_model.h5")
joblib.dump(scaler, "university_scaler.pkl")
print("Model and scaler saved!")

# Plot training and validation MAE
plt.plot(stop_callback.train_mae_history, label='Train MAE')
plt.plot(stop_callback.val_mae_history, label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.title('Training vs Validation MAE')
plt.grid(True)
plt.show()
