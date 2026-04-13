# -------------------- Tkinter Dashboard with Transaction Simulator --------------------
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# -------------------- Load Model & Test Data --------------------
with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# -------------------- Predictions & Metrics --------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

cm = confusion_matrix(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# -------------------- Tkinter GUI --------------------
root = tk.Tk()
root.title("Credit Card Fraud Detection Dashboard")
root.geometry("1200x700")  # smaller window
root.minsize(width=1000, height=600)
root.configure(bg="#f5f5f5")

# -------------------- Frames --------------------
header_frame = tk.Frame(root, bg="#2c3e50", height=50)
header_frame.pack(fill='x')

header_label = tk.Label(header_frame, text="Credit Card Fraud Detection Dashboard",
                        bg="#2c3e50", fg="white", font=("Arial", 16, "bold"))
header_label.pack(pady=8)

main_frame = tk.Frame(root, bg="#f5f5f5")
main_frame.pack(fill='both', expand=True, padx=10, pady=10)

# Left panel for buttons + simulator
left_panel = tk.Frame(main_frame, bg="#ecf0f1", width=280)
left_panel.pack(side='left', fill='y', padx=(0,10), pady=5)

# Buttons frame at top
button_frame = tk.Frame(left_panel, bg="#ecf0f1")
button_frame.pack(fill='x', pady=(0,10))

# Simulator frame (scrollable)
sim_frame_container = tk.Frame(left_panel, bg="#ecf0f1")
sim_frame_container.pack(fill='both', expand=True)

canvas_sim = tk.Canvas(sim_frame_container, bg="#ecf0f1")
scrollbar_sim = ttk.Scrollbar(sim_frame_container, orient="vertical", command=canvas_sim.yview)
scrollable_frame = tk.Frame(canvas_sim, bg="#ecf0f1")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas_sim.configure(scrollregion=canvas_sim.bbox("all"))
)

canvas_sim.create_window((0,0), window=scrollable_frame, anchor="nw")
canvas_sim.configure(yscrollcommand=scrollbar_sim.set)
canvas_sim.pack(side="left", fill="both", expand=True)
scrollbar_sim.pack(side="right", fill="y")

# Right panel for plots/text
display_frame = tk.Frame(main_frame, bg="white")
display_frame.pack(side='right', fill='both', expand=True)
plot_frame = tk.Frame(display_frame, bg="white")
plot_frame.pack(fill='both', expand=True, padx=5, pady=5)
canvas = None  # placeholder for matplotlib

# -------------------- Functions --------------------
def clear_display():
    global canvas
    for widget in plot_frame.winfo_children():
        widget.destroy()
    canvas = None

def show_confusion_matrix():
    clear_display()
    fig, ax = plt.subplots(figsize=(3.5,2.5))  # smaller plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    global canvas
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def show_pr_curve():
    clear_display()
    fig, ax = plt.subplots(figsize=(3.5,2.5))  # smaller plot
    ax.plot(recall, precision, marker='.', label='Random Forest')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve (PR-AUC={pr_auc:.4f})')
    ax.legend()
    global canvas
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def show_feature_importance():
    clear_display()
    fig, ax = plt.subplots(figsize=(4,3))  # smaller plot
    sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax)
    ax.set_title('Feature Importance')
    global canvas
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

def show_classification_report_table():
    clear_display()
    metrics_label = tk.Label(plot_frame, text=f"ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}",
                             font=("Arial", 12, "bold"), bg="white")
    metrics_label.pack(pady=5)
    
    columns = ("Class", "Precision", "Recall", "F1-Score", "Support")
    tree = ttk.Treeview(plot_frame, columns=columns, show="headings", height=6)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=90)
    
    report_data = classification_report(y_test, y_pred, output_dict=True)
    for cls, metrics in report_data.items():
        if cls in ['0','1']:
            tree.insert("", tk.END, values=(
                cls,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{int(metrics['support'])}"
            ))
    scrollbar_tree = ttk.Scrollbar(plot_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar_tree.set)
    scrollbar_tree.pack(side="right", fill="y")
    tree.pack(fill="both", expand=True)

# -------------------- Buttons --------------------
btn_params = [
    ("Confusion Matrix", show_confusion_matrix),
    ("Precision-Recall Curve", show_pr_curve),
    ("Feature Importance", show_feature_importance),
    ("Classification Report & Scores", show_classification_report_table)
]

for text, cmd in btn_params:
    b = ttk.Button(button_frame, text=text, command=cmd)
    b.pack(fill='x', padx=10, pady=5)

# -------------------- Transaction Simulator --------------------
entries = {}
for i, feat in enumerate(X_test.columns):
    lbl = tk.Label(scrollable_frame, text=feat, bg="#ecf0f1", font=("Arial", 8))
    lbl.grid(row=i, column=0, sticky='w', padx=5, pady=2)
    ent = tk.Entry(scrollable_frame, width=8)
    ent.grid(row=i, column=1, padx=5, pady=2)
    entries[feat] = ent

def predict_transaction():
    try:
        data = {feat: [float(entries[feat].get())] for feat in X_test.columns}
        df = pd.DataFrame(data)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0,1]
        result = "FRAUD" if pred==1 else "Genuine"
        messagebox.showinfo("Prediction Result", f"Transaction: {result}\nProbability of Fraud: {prob:.4f}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input or missing values.\n{str(e)}")

predict_btn = tk.Button(scrollable_frame, text="Predict Transaction", command=predict_transaction,
                        bg="#3498db", fg="white")
predict_btn.grid(row=len(X_test.columns), column=0, columnspan=2, pady=10)

# -------------------- Add Entry from Dataset Button --------------------
import random
from tkinter import filedialog  # if not already imported at the top

def add_entry_from_dataset():
    try:
        # Ask user to select a CSV file
        file_path = filedialog.askopenfilename(title="Select CSV Dataset",
                                               filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return  # user cancelled
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Pick a random row
        row = df.sample(n=1).iloc[0]
        
        # Fill the simulator entries
        for feat in X_test.columns:
            if feat in row:
                entries[feat].delete(0, tk.END)
                entries[feat].insert(0, str(row[feat]))
        
        # Predict
        data = {feat: [float(entries[feat].get())] for feat in X_test.columns}
        df_pred = pd.DataFrame(data)
        pred = model.predict(df_pred)[0]
        prob = model.predict_proba(df_pred)[0,1]
        result = "FRAUD" if pred==1 else "Genuine"
        
        # Show actual (if target column exists) and predicted
        actual = row.get('Class', 'Unknown')  # replace 'Class' with your dataset target column name
        messagebox.showinfo("Random Entry Prediction",
                            f"Actual: {actual}\nPredicted: {result}\nProbability of Fraud: {prob:.4f}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{str(e)}")

# Add button at the end of scrollable frame
add_random_btn = tk.Button(scrollable_frame, text="Add Entry from Dataset", command=add_entry_from_dataset,
                           bg="#27ae60", fg="white")
add_random_btn.grid(row=len(X_test.columns)+1, column=0, columnspan=2, pady=10)

# -------------------- Start GUI --------------------
root.mainloop()