import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataGenerator:
    def __init__(self, train_dir, valid_dir, padding_type='post', trunc_type='post'):
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.padding_type = padding_type
        self.trunc_type = trunc_type
        self.tokenizer = Tokenizer()

    def load_data(self, dir_name):
        data = {}
        for class_name in os.listdir(dir_name):
            class_dir = os.path.join(dir_name, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            data[class_name] = []
            for filename in os.listdir(class_dir):
                if os.path.isfile(os.path.join(class_dir, filename)):
                    with open(os.path.join(class_dir, filename), 'r') as f:
                        data[class_name].append(f.read())
        return data

    def prepare_data(self, data):
        for class_name in data.keys():
            self.tokenizer.fit_on_texts(data[class_name])
            sequences = self.tokenizer.texts_to_sequences(data[class_name])
            if sequences:
                padded = pad_sequences(sequences, padding=self.padding_type, truncating=self.trunc_type)
                data[class_name] = [tf.expand_dims(p, -1) for p in padded]  # Add an extra dimension at the end to avoid ValueError
        return data

    def generate(self):
        train_data = self.load_data(self.train_dir)
        valid_data = self.load_data(self.valid_dir)
        train_data = self.prepare_data(train_data)
        valid_data = self.prepare_data(valid_data)

        train_data = {k: v for k, v in train_data.items() if len(v) > 0}
        valid_data = {k: v for k, v in valid_data.items() if len(v) > 0}

        print(f"Train data info: {len(train_data.keys())} classes, {sum([len(v) for v in train_data.values()])} samples")
        print(f"Valid data info: {len(valid_data.keys())} classes, {sum([len(v) for v in valid_data.values()])} samples")

        return (train_data, valid_data)
