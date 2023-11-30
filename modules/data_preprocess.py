import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataGenerator:
    def __init__(self, train_russian_dir, train_english_dir, valid_russian_dir, valid_english_dir, padding_type='post', trunc_type='post'):
        self.train_russian_dir = train_russian_dir
        self.train_english_dir = train_english_dir
        self.valid_russian_dir = valid_russian_dir
        self.valid_english_dir = valid_english_dir
        self.padding_type = padding_type
        self.trunc_type = trunc_type
        self.tokenizer_russian = Tokenizer()
        self.tokenizer_english = Tokenizer()

    def load_data(self, russian_dir, english_dir):
        russian_data = []
        english_data = []
        if not os.listdir(russian_dir) or not os.listdir(english_dir):  # Check if the directories are not empty
            return russian_data, english_data
        for filename in os.listdir(russian_dir):
            if os.path.isfile(os.path.join(russian_dir, filename)):
                with open(os.path.join(russian_dir, filename), 'r') as f:
                    russian_data.append(f.read())
        for filename in os.listdir(english_dir):
            if os.path.isfile(os.path.join(english_dir, filename)):
                with open(os.path.join(english_dir, filename), 'r') as f:
                    english_data.append(f.read())
        return russian_data, english_data

    def prepare_data(self, russian_data, english_data):
        self.tokenizer_russian.fit_on_texts(russian_data)
        self.tokenizer_english.fit_on_texts(english_data)
        russian_sequences = self.tokenizer_russian.texts_to_sequences(russian_data)
        english_sequences = self.tokenizer_english.texts_to_sequences(english_data)
        russian_padded = pad_sequences(russian_sequences, padding=self.padding_type, truncating=self.trunc_type)
        english_padded = pad_sequences(english_sequences, padding=self.padding_type, truncating=self.trunc_type)
        russian_data = [tf.expand_dims(p, 0) for p in russian_padded]  # Add an extra dimension at the beginning to avoid ValueError
        english_data = [tf.expand_dims(p, 0) for p in english_padded]  # Add an extra dimension at the beginning to avoid ValueError
        return russian_data, english_data

    def generate(self):
        train_russian_data, train_english_data = self.load_data(self.train_russian_dir, self.train_english_dir)
        valid_russian_data, valid_english_data = self.load_data(self.valid_russian_dir, self.valid_english_dir)
        train_russian_data, train_english_data = self.prepare_data(train_russian_data, train_english_data)
        if valid_russian_data and valid_english_data:  # Check if the validation data is not empty
            valid_russian_data, valid_english_data = self.prepare_data(valid_russian_data, valid_english_data)

        print(f"Train Russian data info: {len(train_russian_data)} samples")
        print(f"Train English data info: {len(train_english_data)} samples")
        print(f"Valid Russian data info: {len(valid_russian_data)} samples")
        print(f"Valid English data info: {len(valid_english_data)} samples")

        return (train_russian_data, train_english_data, valid_russian_data, valid_english_data)

