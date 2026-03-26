import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tkinter as tk
from tkinter import ttk

# load dataset
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train
history = model.fit(X_train, y_train, epochs=20, batch_size=10, verbose=0)

# accuracy
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# GUI
root = tk.Tk()
root.title("❤️ Heart Disease Predictor")
root.geometry("520x780")
root.configure(bg="#e8f0fe")

# title
tk.Label(root, text="❤️ Heart Disease Predictor", font=("Arial", 18, "bold"), bg="#e8f0fe").pack(pady=10)
tk.Label(root, text="Enter Patient Details", font=("Arial", 10), bg="#e8f0fe").pack()

# accuracy
tk.Label(root, text=f"Model Accuracy: {round(accuracy*100,2)}%", bg="#e8f0fe", fg="blue").pack(pady=5)

frame = tk.Frame(root, bg="#e8f0fe")
frame.pack(pady=10)

entries = {}

labels_map = {
    "age": "Age",
    "sex": "Gender",
    "cp": "Chest Pain",
    "trestbps": "Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Sugar Level",
    "restecg": "ECG",
    "thalach": "Heart Rate",
    "exang": "Exercise Pain",
    "oldpeak": "ST Depression",
    "slope": "Slope",
    "ca": "Vessels",
    "thal": "Thal"
}

dropdowns = {
    "sex": ["Male (1)", "Female (0)"],
    "cp": ["0", "1", "2", "3"],
    "fbs": ["0", "1"],
    "restecg": ["0", "1", "2"],
    "exang": ["0", "1"],
    "slope": ["0", "1", "2"],
    "ca": ["0", "1", "2", "3"],
    "thal": ["0", "1", "2", "3"]
}

# inputs
for col in df.columns[:-1]:
    row = tk.Frame(frame, bg="#e8f0fe")
    row.pack(pady=4)

    tk.Label(row, text=labels_map[col], width=18, anchor='w', bg="#e8f0fe").pack(side=tk.LEFT)

    if col in dropdowns:
        combo = ttk.Combobox(row, values=dropdowns[col], width=18)
        combo.current(0)
        combo.pack(side=tk.RIGHT)
        entries[col] = combo
    else:
        entry = tk.Entry(row, width=20)
        entry.insert(0, "120" if col == "trestbps" else "200" if col == "chol" else "50")
        entry.pack(side=tk.RIGHT)
        entries[col] = entry

# prediction
def predict():
    try:
        values = []
        for col in df.columns[:-1]:
            val = entries[col].get()

            if col in dropdowns:
                val = val.split("(")[-1].replace(")", "")

            values.append(float(val))

        values = np.array(values).reshape(1, -1)
        scaled_values = scaler.transform(values)

        prediction = model.predict(scaled_values)
        risk = float(prediction[0][0]) * 100

        # result
        if risk > 50:
            result_label.config(text=f"⚠ High Risk ({round(risk,2)}%)", fg="red")
            summary = "Patient shows high risk due to factors like age, cholesterol or blood pressure."
        else:
            result_label.config(text=f"✅ Low Risk ({round(risk,2)}%)", fg="green")
            summary = "Patient appears to have low risk based on provided health parameters."

        summary_label.config(text=summary)

        # feature importance
        important = []

        if values[0][4] > 200:
            important.append("Cholesterol")

        if values[0][3] > 130:
            important.append("Blood Pressure")

        if values[0][0] > 50:
            important.append("Age")

        if important:
            importance_label.config(text="Important Factors: " + ", ".join(important))
        else:
            importance_label.config(text="Important Factors: None")

        # risk meter
        progress['value'] = risk

    except:
        result_label.config(text="⚠ Enter valid inputs", fg="orange")

# graph
def show_graph():
    plt.plot(history.history['accuracy'])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

# buttons
tk.Button(root, text="Predict", command=predict,
          font=("Arial", 12, "bold"),
          bg="#4CAF50", fg="white", width=15).pack(pady=10)

tk.Button(root, text="Show Accuracy Graph", command=show_graph,
          font=("Arial", 10),
          bg="#2196F3", fg="white").pack()

# result
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#e8f0fe")
result_label.pack(pady=10)

# summary
summary_label = tk.Label(root, text="", font=("Arial", 10), wraplength=400, bg="#e8f0fe")
summary_label.pack(pady=5)

# feature importance
importance_label = tk.Label(root, text="", font=("Arial", 10, "bold"), bg="#e8f0fe")
importance_label.pack(pady=5)

# progress bar (risk meter)
progress = ttk.Progressbar(root, orient="horizontal",
                           length=300, mode="determinate")
progress.pack(pady=10)

root.mainloop()