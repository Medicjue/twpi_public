import pickle
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class MdlTrainer:
    def __init__(self):
        self.maxlen = 64
        # self.vocab_size = 52000
        self.vocab_size = 100000

        # Google Drive Path: https://drive.google.com/file/d/1HZd_DoLlxOfWIDVZARMABEhxpfrwwaqF/view?usp=sharing
        with open('bookingReview/X_train.pickle', 'rb') as f:
            self.x_train = pickle.load(f)
        # Google Drive Path: https://drive.google.com/file/d/17cPbvfU6fHPBtCNsxz4KGpkqlZkz81QK/view?usp=sharing
        with open('bookingReview/X_test.pickle', 'rb') as f:
            self.x_val = pickle.load(f)
        # Google Drive Path: https://drive.google.com/file/d/1RH_FFu2DTcfdYwAxyXg1bMe4klfalASO/view?usp=sharing
        with open('bookingReview/y_train.pickle', 'rb') as f:
            self.y_train = pickle.load(f)
        # Google Drive Path: https://drive.google.com/file/d/1sVjHjI0XsxxqNKkrND_6__VCXdItIvxa/view?usp=sharing
        with open('bookingReview/y_test.pickle', 'rb') as f:
            self.y_val = pickle.load(f)


    def create_mdl(self):
        self.embed_dim = 32  # Embedding size for each token
        self.num_heads = 2  # Number of attention heads
        self.ff_dim = 32  # Hidden layer size in feed forward network inside transformer

        inputs = layers.Input(shape=(self.maxlen,))
        embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size, self.embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(2, activation="softmax")(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        self.model.summary()

    def train_mdl(self):
        history = self.model.fit(
            self.x_train, self.y_train, batch_size=32, epochs=5, validation_data=(self.x_val, self.y_val)
        )

    def save_mdl(self, path):
        self.model.save(path)

if __name__ == '__main__':
    mdlTrainer = MdlTrainer()
    mdlTrainer.create_mdl()
    mdlTrainer.train_mdl()
    mdlTrainer.save_mdl(path='bookingReview/bookingReviewMdl_20220630')