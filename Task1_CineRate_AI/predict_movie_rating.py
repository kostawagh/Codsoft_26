import tkinter as tk
from tkinter import ttk
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageTk  # For cover image

# --- Load saved model and artifacts ---
with open('models/rf_movie_model.pkl', 'rb') as f: best_model = pickle.load(f)
with open('models/imputer.pkl', 'rb') as f: imputer = pickle.load(f)
with open('models/feature_cols.pkl', 'rb') as f: feature_cols = pickle.load(f)
with open('models/global_mean.pkl', 'rb') as f: global_mean = pickle.load(f)
with open('models/director_avg.pkl', 'rb') as f: director_smooth = pickle.load(f)
with open('models/actor1_avg.pkl', 'rb') as f: actor1_smooth = pickle.load(f)
with open('models/actor2_avg.pkl', 'rb') as f: actor2_smooth = pickle.load(f)
with open('models/actor3_avg.pkl', 'rb') as f: actor3_smooth = pickle.load(f)
with open('models/top_genres.pkl', 'rb') as f: top_genres = pickle.load(f)
with open('models/directors_list.pkl', 'rb') as f: directors_list = pickle.load(f)
with open('models/actors_list.pkl', 'rb') as f: actors_list = pickle.load(f)
with open('models/genres_list.pkl', 'rb') as f: genres_list = pickle.load(f)

# --- Prediction function ---
def predict_rating():
    try:
        name = name_var.get() or "Unknown Movie"
        year = int(year_var.get()) if year_var.get().isdigit() else 2020
        duration = int(duration_var.get()) if duration_var.get().isdigit() else 120

        if year < 1900 or year > 2025:
            raise ValueError("Enter a valid year (1900–2025)")

        if duration < 30 or duration > 300:
            raise ValueError("Duration must be between 30–300 mins")
        
        genre = genre_var.get() or "Unknown"
        votes = 0  # Default since input removed
        director = director_var.get()
        if director not in director_smooth:
            director = "Unknown"

        actor1 = actor1_var.get()
        if actor1 not in actor1_smooth:
            actor1 = "Unknown"

        actor2 = actor2_var.get() or "Unknown"
        if actor2 not in actor2_smooth:
            actor2 = "Unknown"

        actor3 = actor3_var.get() or "Unknown"
        if actor3 not in actor3_smooth:
            actor3 = "Unknown"

        log_votes = np.log1p(votes)
        
        from datetime import datetime
        current_year = datetime.now().year
        movie_age = current_year - year

        num_genres = len(genre.split(','))

        dir_rating = director_smooth.get(director, global_mean)
        act1_rating = actor1_smooth.get(actor1, global_mean)
        act2_rating = actor2_smooth.get(actor2, global_mean)
        act3_rating = actor3_smooth.get(actor3, global_mean)
        actors_avg = (act1_rating + act2_rating + act3_rating) / 3

        user_genres = [g.strip().title() for g in genre.split(',')]
        genre_flags = [int(g in user_genres) for g in top_genres]

        row = [duration, log_votes, movie_age, num_genres, dir_rating, actors_avg] + genre_flags
        row_df = pd.DataFrame([row], columns=feature_cols)
        row_imputed = imputer.transform(row_df)

        predicted = best_model.predict(row_imputed)[0]
        predicted = np.clip(predicted, 1, 10)

        result_label.config(text=f"{name} ({year})\nPredicted Rating: {predicted:.2f} / 10",
                            fg="white", bg="green")
    except Exception as e:
        result_label.config(text=f"Error: {e}", fg="white", bg="red")

# --- Tkinter UI ---
root = tk.Tk()
root.title("🎬 CineRate AI")
root.geometry("450x700")
root.minsize(450, 700)
root.configure(bg="#f0f4f8")

# --- Cover Image ---
cover_image = Image.open("data/cover.png")
cover_image = cover_image.resize((430, 180))
cover_photo = ImageTk.PhotoImage(cover_image)
cover_label = tk.Label(root, image=cover_photo, bg="#f0f4f8")
cover_label.pack(pady=5)

# Header
header = tk.Label(root, text="CineRate AI", 
                  font=("Segoe UI", 20, "bold"), bg="#f0f4f8")
header.pack(pady=10)

frame = tk.Frame(root, bg="#f0f4f8")
frame.pack(pady=10, padx=10, fill="both", expand=True)

# Input variables

name_var = tk.StringVar()
year_var = tk.StringVar()
duration_var = tk.StringVar()
genre_var = tk.StringVar()
director_var = tk.StringVar()
actor1_var = tk.StringVar()
actor2_var = tk.StringVar()
actor3_var = tk.StringVar()

fields = [
    ("Movie Name", name_var),
    ("Year", year_var),
    ("Duration (min)", duration_var),
    ("Genre (comma-separated)", genre_var),
    ("Director", director_var),
    ("Actor 1", actor1_var),
    ("Actor 2", actor2_var),
    ("Actor 3", actor3_var),
]

for i, (label, var) in enumerate(fields):
    tk.Label(frame, text=label, anchor="w",
             font=("Segoe UI", 10), bg="#f0f4f8").grid(row=i, column=0, pady=5, sticky="w")

    if label in ["Director", "Actor 1", "Actor 2", "Actor 3"]:
        ttk.Combobox(frame, textvariable=var, values=actors_list if "Actor" in label else directors_list, width=28)\
            .grid(row=i, column=1, pady=5, padx=5)

    elif label == "Genre (comma-separated)":
        ttk.Combobox(frame, textvariable=var, values=genres_list, width=28)\
            .grid(row=i, column=1, pady=5, padx=5)

    else:
        tk.Entry(frame, textvariable=var, width=30,
                 font=("Segoe UI", 10), bg="#e0f7fa")\
            .grid(row=i, column=1, pady=5, padx=5)
        
tk.Button(root, text="Predict Rating", command=predict_rating,
          bg="#0288d1", fg="white",
          font=("Segoe UI", 12, "bold"),
          activebackground="#0277bd",
          relief="flat", padx=10, pady=5).pack(pady=20)

# Result label
result_label = tk.Label(root, text="", font=("Segoe UI", 14, "bold"), width=40, height=3, bg="#f0f4f8")
result_label.pack(pady=10)

root.mainloop()