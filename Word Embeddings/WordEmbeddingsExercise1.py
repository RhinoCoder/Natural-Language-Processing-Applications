# Exercise: download pretrained GloVe vectors from
# https://nlp.stanford.edu/projects/glove/
# Implement your own find_analogies() and nearest_neighbors()
# Hint: you do NOT have to go hunting around on Stackoverflow
# you do NOT have to copy and paste/
# @authorRhinoCoder


import sys

import numpy as np
from numpy.linalg import norm
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLineEdit, QLabel, QFileDialog, QTextEdit
)


def LoadGloveModel(glove_text_file):
    glove_dict = {}
    with open(glove_text_file, "r", encoding="utf-8") as inp:
        for line in inp:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            glove_dict[word] = vector
    print("Successfully loaded the model..")
    return glove_dict


def CosineSimilarity(vec1, vec2):
    sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return sim


class NLPApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model = None

    def init_ui(self):
        self.setWindowTitle("Word Embeddings")
        self.setGeometry(200, 200, 500, 400)
        self.layout = QVBoxLayout()

        # Load Model Button
        self.load_button = QPushButton("Load Glove model", self)
        self.load_button.clicked.connect(self.load_model)
        self.layout.addWidget(self.load_button)

        # Word Analogy Inputs
        self.word1_input = QLineEdit(self)
        self.word1_input.setPlaceholderText("Word 1 (e.g., 'man')")
        self.layout.addWidget(self.word1_input)

        self.word2_input = QLineEdit(self)
        self.word2_input.setPlaceholderText("Word 2 (e.g., 'king')")
        self.layout.addWidget(self.word2_input)

        self.word3_input = QLineEdit(self)
        self.word3_input.setPlaceholderText("Word 3 (e.g., 'woman')")
        self.layout.addWidget(self.word3_input)

        self.analogy_button = QPushButton("Find Analogy", self)
        self.analogy_button.clicked.connect(self.find_analogies_wrapper)  # Connect to wrapper
        self.layout.addWidget(self.analogy_button)

        # Nearest Neighbors Input
        self.word_input = QLineEdit(self)
        self.word_input.setPlaceholderText("Word (e.g., 'queen')")
        self.layout.addWidget(self.word_input)

        self.neighbors_button = QPushButton("Find Nearest Neighbors", self)
        self.neighbors_button.clicked.connect(self.nearest_neighbors_wrapper)  # Connect to wrapper
        self.layout.addWidget(self.neighbors_button)

        # Output Box
        self.output_box = QTextEdit(self)
        self.output_box.setReadOnly(True)
        self.layout.addWidget(self.output_box)

        self.setLayout(self.layout)

    def load_model(self):
        options = QFileDialog.Option(0)

        filepath, _ = QFileDialog.getOpenFileName(self, "Open GloVe Model", "", "Text Files (*.txt)", options=options)
        if filepath:
            try:
                self.model = LoadGloveModel(filepath)
                self.output_box.setText("GloVe Model Loaded Successfully!")
                self.analogy_button.setEnabled(True)  # Enable buttons after loading
                self.neighbors_button.setEnabled(True)
            except Exception as e:
                self.output_box.setText(f"Error loading model: {e}")
                self.model = None
                self.analogy_button.setEnabled(False) # Disable if loading fails
                self.neighbors_button.setEnabled(False)


    def find_analogies_wrapper(self):
        if self.model is None:
            self.output_box.setText("Please load a GloVe model first.")
            return

        word1 = self.word1_input.text()
        word2 = self.word2_input.text()
        word3 = self.word3_input.text()

        if not all([word1, word2, word3]):
            self.output_box.setText("Please enter all three words for analogy.")
            return

        try:
            best_match = self.find_analogies(self.model, word1, word2, word3)
            if best_match:
                self.output_box.setText(f"'{word1}' is to '{word2}' as '{word3}' is to '{best_match}'")
            else:
                self.output_box.setText("One or more words not found in the model.")
        except Exception as e:
            self.output_box.setText(f"An error occurred: {e}")

    def find_analogies(self, model, word1, word2, word3):
        if word1 not in model or word2 not in model or word3 not in model:
            return None

        vec1 = model[word1]
        vec2 = model[word2]
        vec3 = model[word3]
        resulting_vector = vec2 - vec1 + vec3

        best_match = None
        best_score = -float("inf")

        for word, vec in model.items():
            if word in {word1, word2, word3}:
                continue
            similarity = CosineSimilarity(resulting_vector, vec)
            if similarity > best_score:
                best_score = similarity
                best_match = word

        return best_match


    def nearest_neighbors_wrapper(self):
        if self.model is None:
            self.output_box.setText("Please load a GloVe model first.")
            return

        word = self.word_input.text()
        if not word:
            self.output_box.setText("Please enter a word.")
            return

        try:
            neighbors = self.nearest_neighbors(self.model, word)
            if neighbors:
                output_text = "\n".join([f"{neighbor[0]}: {neighbor[1]:.4f}" for neighbor in neighbors])
                self.output_box.setText(output_text)
            else:
                self.output_box.setText("Word not found in the model.")
        except Exception as e:
            self.output_box.setText(f"An error occurred: {e}")



    def nearest_neighbors(self, model, word, top_n=5):
        if word not in model:
            return None

        vector_word = model[word]
        similarities = []

        for other_word, vec in model.items():
            if other_word == word:
                continue
            similarity = CosineSimilarity(vector_word, vec)
            similarities.append((other_word, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NLPApp()
    window.show()
    sys.exit(app.exec())


    '''
    #FOR MANUAL USAGE, UNCOMMENT THIS AND COMMENT APP PART ABOVE.
    #Provide your model path to the program.
    glove_text_file_path = "../Datasets/glove.6B.300d.txt"
    # Load the given embeddings.
    glove_embeddings = LoadGloveModel(glove_text_file_path)
    print("Finding analogies of given words....")
    FindAnalogies(glove_embeddings, "man", "king", "woman")
    print("\n Nearest neighbor of 'queen", NearestNeighbors(glove_embeddings, "queen"))
    '''
