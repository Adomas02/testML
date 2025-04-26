import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split



# 1. Load your dataset
data = pd.read_csv('your_dataset.csv')  # change the file name if needed

# 2. Prepare input (test code) and output (labels)
X = data['test_code'].astype(str).values
y = data[['eager_test', 'mystery_guest', 'resource_optimism', 'test_redundancy']].astype(int).values

# 3. Tokenize and pad sequences
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X)

sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(sequences, padding='post', maxlen=300)  # adjust maxlen as needed

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# 5. Build the LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=300),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(4, activation='sigmoid')  # 4 outputs for 4 smells, multilabel classification
])

# 6. Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 8. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 9. Save the model
model.save('lstm_code_smell_detector.h5')

# 10. Predicting on new data (example)
# new_tests = ["public void newTest() { ... }"]
# new_seq = tokenizer.texts_to_sequences(new_tests)
# new_pad = pad_sequences(new_seq, padding='post', maxlen=300)
# predictions = model.predict(new_pad)
# print(predictions)
