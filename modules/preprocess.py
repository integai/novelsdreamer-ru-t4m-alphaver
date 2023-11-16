import tensorflow as tf
import os
import re

class Preprocess:
    def __init__(self, max_len=100, batch_size=32, train_dir=None, valid_dir=None, oov_token="<OOV>"):
        self.max_len = max_len
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.tokenizer_eng = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
        self.tokenizer_rus = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^а-яА-Яa-zA-Z0-9]", " ", text)
        return text

    def load_data(self, directory):
        texts_eng = []
        texts_rus = []
        try:
            for lang_dir in os.listdir(directory):
                lang_path = os.path.join(directory, lang_dir)
                if os.path.isdir(lang_path):
                    for filename in os.listdir(lang_path):
                        with open(os.path.join(lang_path, filename), 'r', encoding='utf-8') as file:
                            text = self.clean_text(file.read())
                            if lang_dir.lower() == 'english':
                                texts_eng.append(text)
                            elif lang_dir.lower() == 'russian':
                                texts_rus.append(text)
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return [], []
        return texts_eng, texts_rus

    def tokenize_texts(self, texts, tokenizer):
        tokenizer.fit_on_texts(texts)
        return tokenizer.texts_to_sequences(texts)

    def pad_sequences(self, sequences):
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_len)

    def prepare_dataset(self, directory):
        texts_eng, texts_rus = self.load_data(directory)
        if not texts_eng or not texts_rus:
            print("No data found for preparation.")
            return None
        sequences_eng = self.tokenize_texts(texts_eng, self.tokenizer_eng)
        sequences_rus = self.tokenize_texts(texts_rus, self.tokenizer_rus)
        padded_sequences_eng = self.pad_sequences(sequences_eng)
        padded_sequences_rus = self.pad_sequences(sequences_rus)
        # Ensure the alignment of English-Russian pairs
        min_len = min(len(padded_sequences_eng), len(padded_sequences_rus))
        dataset = tf.data.Dataset.from_tensor_slices((padded_sequences_eng[:min_len], padded_sequences_rus[:min_len]))
        dataset = dataset.batch(self.batch_size)
        return dataset