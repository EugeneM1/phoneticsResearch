from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def prepare_data(words, get_word_embedding, get_phonetic_representation):
    X = []  # Phonetic or Spelling
    y = []  # Semantic vectors

    for word in words:
        embedding = get_word_embedding(word)
        phonetic = get_phonetic_representation(word)
        if phonetic:
            phonetic_flat = [item for sublist in phonetic for item in sublist]
            X.append(phonetic_flat)
            y.append(embedding)
    
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test
