import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model


#--------------------------------------------------------------------------------COMMENT OUT THE CODE IF DOES NOT WORK
#Main data from dataset
english_list = []  #Your dataset goes here
assamese_list = [] #and here

assamese_texts = assamese_list[:100]
english_texts = english_list[:100]

#YOU CAN INCREASE DECREASE THE SIZE OF ARRAY ABOVE AS PER YOUR SYSTEMS CAPACITY

# If system is unable to take load try the below given example data---------------------------------------------IMPORTANT-----------------
# System cannot take too much load of huge dataset

# Example data
# english_texts = ["Hello.", "How are you?", "I love NLP."]
# assamese_texts = ["হ্যালো।", "আপনি কেমন আছেন?", "মই NLP ভাল পৱন।"]




# Tokenization and padding
num_words = 10000  # Number of unique words in the vocabulary
tokenizer_eng = Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer_as = Tokenizer(num_words=num_words, oov_token="<OOV>")

tokenizer_eng.fit_on_texts(english_texts)
tokenizer_as.fit_on_texts(assamese_texts)

eng_sequences = tokenizer_eng.texts_to_sequences(english_texts)
as_sequences = tokenizer_as.texts_to_sequences(assamese_texts)

max_seq_length = max(max(len(seq) for seq in eng_sequences),
                     max(len(seq) for seq in as_sequences))

eng_padded = pad_sequences(eng_sequences, maxlen=max_seq_length, padding="post")
as_padded = pad_sequences(as_sequences, maxlen=max_seq_length, padding="post")

# Model architecture
latent_dim = 512  # Size of the LSTM layer

encoder_inputs = Input(shape=(max_seq_length,))
enc_emb = Embedding(num_words, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_seq_length,))
dec_emb_layer = Embedding(num_words, latent_dim, mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_words, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Training
decoder_targets = np.zeros((len(eng_padded), max_seq_length, num_words), dtype="float32")
for i, seq in enumerate(as_padded):
    for j, word_idx in enumerate(seq):
        decoder_targets[i, j, word_idx] = 1.0

model.fit([eng_padded, as_padded], decoder_targets, batch_size=2, epochs=50)

# Inference
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb_inf = dec_emb_layer(decoder_inputs)
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm(
    dec_emb_inf, initial_state=decoder_states_inputs
)
decoder_states = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs_inf] + decoder_states
)


#This is for output
def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_as.word_index["<OOV>"]  # Start with OOV token

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer_as.index_word[sampled_token_index]

        if sampled_word != "<OOV>" and sampled_word != "<end>":
            decoded_sentence += " " + sampled_word

        if sampled_word == "<end>" or len(decoded_sentence.split()) > max_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()


input_sentence = input("Enter your sentence\n")
input_sequence = tokenizer_eng.texts_to_sequences([input_sentence])
input_padded = pad_sequences(input_sequence, maxlen=max_seq_length, padding="post")

translated_sentence = translate_sentence(input_padded)
print("Input:", input_sentence)
print("Translated:", translated_sentence)
