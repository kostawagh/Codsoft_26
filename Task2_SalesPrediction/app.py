# advertising_dashboard.py
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- Load Best Model --------------------
with open('best_ad_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# -------------------- Global Variables --------------------
df = None
X_train = X_test = y_train = y_test = y_pred = None
canvas_widget = None

# -------------------- Load CSV automatically --------------------
def load_csv():
    global df
    try:
        df = pd.read_csv("advertising.csv")
        data_text.set(df.head().to_string())
        status_label.config(text="CSV loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load CSV: {e}")

# -------------------- Evaluate Best Model --------------------
def evaluate_model():
    global X_train, X_test, y_train, y_test, y_pred

    if df is None:
        messagebox.showerror("Error", "Load the CSV first!")
        return

    X = df[feature_cols]
    y = df['Sales']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics_text.set(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f}")
    status_label.config(text="Model evaluated successfully!")

# -------------------- Plot Helper --------------------
def clear_plot():
    global canvas_widget
    for widget in plot_inner_frame.winfo_children():
        widget.destroy()
    canvas_widget = None

def draw_plot(fig):
    global canvas_widget
    clear_plot()
    canvas_widget = FigureCanvasTkAgg(fig, master=plot_inner_frame)
    canvas_widget.draw()
    canvas_widget.get_tk_widget().pack()

# -------------------- Plots --------------------
def plot_actual_vs_pred():
    if y_test is None or y_pred is None:
        messagebox.showerror("Error", "Evaluate model first!")
        return
    fig, ax = plt.subplots(figsize=(4,2.5))
    ax.scatter(y_test, y_pred, color='teal', s=15)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    draw_plot(fig)

def plot_residuals():
    if y_test is None or y_pred is None:
        messagebox.showerror("Error", "Evaluate model first!")
        return
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(4,2.5))
    ax.hist(residuals, bins=15, color='coral', edgecolor='black')
    ax.set_title("Residuals")
    draw_plot(fig)

def plot_feature_importance():
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fig, ax = plt.subplots(figsize=(4,2.5))
        ax.bar(feature_cols, importances, color='skyblue')
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance")
        draw_plot(fig)
    else:
        status_label.config(text="Feature importance not available for this model!")

# -------------------- Predict Sales from Inputs --------------------
def predict_sales():
    try:
        tv = float(tv_var.get())
        radio = float(radio_var.get())
        newspaper = float(newspaper_var.get())
        X_input = pd.DataFrame([[tv, radio, newspaper]], columns=feature_cols)
        prediction = model.predict(X_input)[0]
        pred_result.set(f"Predicted Sales: {prediction:.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Enter valid numbers!")

# -------------------- Tkinter GUI --------------------
window = tk.Tk()
window.title("Advertising Sales Dashboard")
window.geometry("600x750")  # Increased height
window.resizable(False, False)

# ---------- Section 1: Load & Evaluate CSV ----------
header1 = tk.Label(window, text="Section 1: Load & Evaluate CSV", font=("Arial", 12, "bold"))
header1.grid(row=0, column=0, columnspan=3, pady=5)

load_btn = tk.Button(window, text="Load CSV", command=load_csv)
load_btn.grid(row=1, column=0, pady=5)

data_text = tk.StringVar()
data_label = tk.Label(window, textvariable=data_text, justify="left")
data_label.grid(row=2, column=0, columnspan=3, padx=5)

eval_btn = tk.Button(window, text="Evaluate Model", command=evaluate_model)
eval_btn.grid(row=3, column=0, pady=5)

metrics_text = tk.StringVar()
metrics_label = tk.Label(window, textvariable=metrics_text, font=("Arial", 10))
metrics_label.grid(row=3, column=1, columnspan=2)

# Plot buttons
plot_btn1 = tk.Button(window, text="Actual vs Predicted", command=plot_actual_vs_pred)
plot_btn1.grid(row=4, column=0, pady=5)
plot_btn2 = tk.Button(window, text="Residuals", command=plot_residuals)
plot_btn2.grid(row=4, column=1)
plot_btn3 = tk.Button(window, text="Feature Importance", command=plot_feature_importance)
plot_btn3.grid(row=4, column=2)

status_label = tk.Label(window, text="Load CSV to start.", fg="blue")
status_label.grid(row=5, column=0, columnspan=3, pady=5)

# ---------- Section 2: Predict Sales ----------
header2 = tk.Label(window, text="Section 2: Predict Sales", font=("Arial", 12, "bold"))
header2.grid(row=6, column=0, columnspan=3, pady=10)

tv_var = tk.StringVar()
radio_var = tk.StringVar()
newspaper_var = tk.StringVar()
pred_result = tk.StringVar()

ttk.Label(window, text="TV Ad Spend ($)").grid(row=7, column=0, sticky="w", padx=5)
ttk.Entry(window, textvariable=tv_var, width=10).grid(row=7, column=1, padx=5)
ttk.Label(window, text="Radio Ad Spend ($)").grid(row=8, column=0, sticky="w", padx=5)
ttk.Entry(window, textvariable=radio_var, width=10).grid(row=8, column=1, padx=5)
ttk.Label(window, text="Newspaper Ad Spend ($)").grid(row=9, column=0, sticky="w", padx=5)
ttk.Entry(window, textvariable=newspaper_var, width=10).grid(row=9, column=1, padx=5)

ttk.Button(window, text="Predict Sales", command=predict_sales).grid(row=10, column=0, pady=10)
tk.Label(window, textvariable=pred_result, font=("Arial", 12, "bold"), fg="green").grid(row=10, column=1, columnspan=2)

# ---------- Scrollable Plot Frame ----------
plot_frame = tk.Frame(window, width=500, height=250, relief="sunken", bd=1)
plot_frame.grid(row=11, column=0, columnspan=3, pady=10)
plot_frame.grid_propagate(False)

canvas = tk.Canvas(plot_frame, width=500, height=250)
scrollbar = tk.Scrollbar(plot_frame, orient="vertical", command=canvas.yview)
plot_inner_frame = tk.Frame(canvas)

plot_inner_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)
canvas.create_window((0, 0), window=plot_inner_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

window.mainloop()