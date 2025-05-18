import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.utils.class_weight import compute_class_weight
import os

# 1. Load dataset
data = pd.read_csv(r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file.csv')

# Test code smells to loop through
smells = ['isEagerTestManual', 'isMysteryGuestManual', 'isResourceOptimismManual', 'isTestRedundancyManual']

# Hyperparameter ranges
epochs_list = [10, 15, 20]
dropout_rates = [0.2, 0.3, 0.4]

for smell in smells:
    print(f'Processing smell: {smell}')
    data[smell] = data[smell].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
    X = data['method_code'].astype(str).values
    y = data[smell].values

    # Tokenize and pad
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(sequences, padding='post', maxlen=300)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

    # Compute class weights
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}

    for dropout in dropout_rates:
        for epochs in epochs_list:
            print(f'Training model for {smell} with epochs={epochs} and dropout={dropout}')
            model = Sequential([
                Embedding(input_dim=10000, output_dim=128, input_length=300),
                Bidirectional(LSTM(128, return_sequences=True)),
                Dropout(dropout),
                Bidirectional(LSTM(64)),
                Dropout(dropout),
                Dense(1, activation='sigmoid')
            ])

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.1,
                class_weight=class_weight
            )

            # Create the directory structure
            out_dir = os.path.join(smell, str(dropout), str(epochs))
            os.makedirs(out_dir, exist_ok=True)

            # Evaluate
            loss, accuracy = model.evaluate(X_test, y_test)
            y_pred_probs = model.predict(X_test)
            y_pred = (y_pred_probs > 0.5).astype(int)
            f1 = f1_score(y_test, y_pred)
            print(
                f'{smell} - Epochs: {epochs}, Dropout: {dropout} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}')

            # Save results
            with open(os.path.join(out_dir, 'results.txt'), 'w') as f:
                f.write(f'Loss: {loss:.4f}\nAccuracy: {accuracy:.4f}\nF1-score: {f1:.4f}\n')

            # Plot Loss
            plt.figure(figsize=(8, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Loss Over Epochs ({smell} epochs={epochs}, dropout={dropout})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, 'loss.png'))
            plt.close()

            # Plot Accuracy
            plt.figure(figsize=(8, 6))
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Accuracy Over Epochs ({smell} epochs={epochs}, dropout={dropout})')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, 'accuracy.png'))
            plt.close()

            # Plot ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve ({smell} epochs={epochs}, dropout={dropout})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, 'roc_curve.png'))
            plt.close()

            model.save(os.path.join(out_dir, 'model.keras'))
