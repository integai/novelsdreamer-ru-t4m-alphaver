import tensorflow as tf
def translate(model, text, max_length=40):
        """
        Translate English text to Russian using the trained model.
        """
        # Tokenize the input text
        tokenized_text = [model.input_vocab_size.word_index.get(i, 0) for i in text.split()]
        tokenized_text = tf.expand_dims(tokenized_text, 0)

        # Initialize the output
        output = tf.expand_dims([model.target_vocab_size.word_index['<start>']], 0)

        for i in range(max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = model.create_masks(tokenized_text, output)

            # Predictions shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = model.call(tokenized_text, output, False, enc_padding_mask, combined_mask, dec_padding_mask)

            # Select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # Return the result if the predicted_id is equal to the end token
            if predicted_id == model.target_vocab_size.word_index['<end>']:
                return tf.squeeze(output, axis=0), attention_weights

            # Concatenate the predicted_id to the output which is given to the decoder as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights