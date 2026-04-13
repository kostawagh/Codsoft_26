import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("data/IMDb_Movies_India.csv", encoding='latin1')

# Clean columns (important)
df['Director'] = df['Director'].fillna("Unknown")
df['Actor 1'] = df['Actor 1'].fillna("Unknown")
df['Actor 2'] = df['Actor 2'].fillna("Unknown")
df['Actor 3'] = df['Actor 3'].fillna("Unknown")
df['Genre'] = df['Genre'].fillna("Unknown")

# --- Extract unique values ---
directors = sorted(df['Director'].unique())
actors = sorted(set(df['Actor 1']) | set(df['Actor 2']) | set(df['Actor 3']))

# Genres need splitting
genre_set = set()
for g in df['Genre']:
    for item in str(g).split(','):
        genre_set.add(item.strip())

genres = sorted(genre_set)

# --- Save them ---
with open('models/directors_list.pkl', 'wb') as f:
    pickle.dump(directors, f)

with open('models/actors_list.pkl', 'wb') as f:
    pickle.dump(actors, f)

with open('models/genres_list.pkl', 'wb') as f:
    pickle.dump(genres, f)

print("Metadata lists saved!")