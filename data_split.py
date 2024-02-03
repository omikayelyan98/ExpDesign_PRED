import os
import pickle
import sklearn
from sklearn.model_selection import train_test_split

# Load sample data
with open("sample_data/ids.pkl", "rb") as file:
    ids = pickle.load(file)
with open("sample_data/labels.pkl", "rb") as file:
    labels = pickle.load(file)
with open("sample_data/one_hot_labels.pkl", "rb") as file:
    one_hot_labels = pickle.load(file)
with open("sample_data/texts.pkl", "rb") as file:
    texts = pickle.load(file)

# Shuffle the data
ids, labels, one_hot_labels, texts = sklearn.utils.shuffle(ids, labels, one_hot_labels, texts, random_state=42)

# Split data into train, validation, and test sets
train_ids, val_test_ids, train_labels, val_test_labels, train_one_hot_labels, val_test_one_hot_labels, train_texts, val_test_texts = train_test_split(
    ids, labels, one_hot_labels, texts, test_size=0.2, random_state=42)
val_ids, test_ids, val_labels, test_labels, val_one_hot_labels, test_one_hot_labels, val_texts, test_texts = train_test_split(
    val_test_ids, val_test_labels, val_test_one_hot_labels, val_test_texts, test_size=0.5, random_state=42)

# Create directories if they don't exist
for set_name in ['train', 'val', 'test']:
    os.makedirs(f"sample_data/{set_name}", exist_ok=True)

# Save data into directories
for set_name, set_ids, set_labels, set_one_hot_labels, set_texts in [
    ('train', train_ids, train_labels, train_one_hot_labels, train_texts),
    ('val', val_ids, val_labels, val_one_hot_labels, val_texts),
    ('test', test_ids, test_labels, test_one_hot_labels, test_texts)
]:
    with open(f"sample_data/{set_name}/ids.pkl", "wb") as file:
        pickle.dump(set_ids, file)
    with open(f"sample_data/{set_name}/labels.pkl", "wb") as file:
        pickle.dump(set_labels, file)
    with open(f"sample_data/{set_name}/one_hot_labels.pkl", "wb") as file:
        pickle.dump(set_one_hot_labels, file)
    with open(f"sample_data/{set_name}/texts.pkl", "wb") as file:
        pickle.dump(set_texts, file)

print("Data split and saved successfully.")