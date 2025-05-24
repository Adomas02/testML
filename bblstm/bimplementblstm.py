import pandas as pd
import numpy as np
import matplotlib
import tensorflow as tf

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

tf.random.set_seed(42)
np.random.seed(42)

# 1. Load dataset
data = pd.read_csv(r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file.csv')

smells = [
    'isEagerTestManual',
    'isMysteryGuestManual',
    'isResourceOptimismManual',
    'isTestRedundancyManual'
]

# Hyperparameter ranges (simplify, focus on higher dropout)
epochs_list = [10, 15, 20]
dropout_rates = [0.4, 0.5, 0.6]

for smell in smells:
    print(f'Processing smell: {smell}')

    data[smell] = data[smell].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})
    X = data['method_code'].astype(str).values
    y = data[smell].values

    # Tokenize and pad
    tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(sequences, padding='post', maxlen=150)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

    # Compute class weights
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}

    for dropout in dropout_rates:
        for epochs in epochs_list:
            print(f'Training model for {smell} with epochs={epochs} and dropout={dropout}')

            model = Sequential([
                Embedding(input_dim=20000, output_dim=128),
                Bidirectional(LSTM(64, kernel_regularizer=l2(0.01))),  # simpler, regularised
                Dropout(dropout),
                Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
            ])

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=6,
                restore_best_weights=True,
                verbose=1
            )

            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,  # bigger batch for regularisation
                validation_split=0.1,
                class_weight=class_weight,
                callbacks=[early_stop],
                verbose=2
            )

            out_dir = os.path.join(smell, f'dropout_{dropout}', f'epochs_{epochs}')
            os.makedirs(out_dir, exist_ok=True)

            # Evaluate
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            y_pred_probs = model.predict(X_test, verbose=0)
            y_pred = (y_pred_probs > 0.5).astype(int)
            f1 = f1_score(y_test, y_pred)

            print(
                f'{smell} - Epochs: {epochs}, Dropout: {dropout} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}')

            # Save results
            with open(os.path.join(out_dir, 'results.txt'), 'w') as f:
                f.write(f'Dropout: {dropout}\nLoss: {loss:.4f}\nAccuracy: {accuracy:.4f}\nF1-score: {f1:.4f}\n')

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

            cleaned_smell_name = smell.replace("is", "").replace("Manual", "")

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FALSE', 'TRUE'])
            plt.figure(figsize=(6, 6))
            disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
            plt.title(f'Confusion Matrix {cleaned_smell_name}')
            plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
            plt.close()
