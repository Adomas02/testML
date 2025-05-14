import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import matplotlib.pyplot as plt

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN custom operations

# Suppress Hugging Face cache warning on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Set up logging level to reduce verbosity
tf.get_logger().setLevel('ERROR')

# 1. Load dataset
data = pd.read_csv(r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file.csv')

# 2. Convert labels properly (TRUE/FALSE → 1/0)
smell = 'isEagerTestManual'
data[smell] = data[smell].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})

# 3. Prepare input and output
X = data['method_code'].astype(str).values
y = data[smell].values

# 4. Tokenize using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        padding='max_length',
        max_length=300,
        truncation=True,
        return_tensors='np'
    )

tokenized_data = tokenize_texts(X)

# 5. Convert tensors to numpy for compatibility with train_test_split
X_input_ids = np.array(tokenized_data['input_ids'])
X_attention_mask = np.array(tokenized_data['attention_mask'])

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_input_ids, y, test_size=0.2, random_state=42
)
attention_train, attention_test = train_test_split(
    X_attention_mask, test_size=0.2, random_state=42
)

# 7. Compute class weights
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight = {0: weights[0], 1: weights[1]}

# 8. Build BERT model using TFBertForSequenceClassification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# 9. Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 10. Train model
history = model.fit(
    {'input_ids': X_train, 'attention_mask': attention_train},
    y_train,
    epochs=3,
    batch_size=16,
    validation_split=0.1,
    class_weight=class_weight
)

# 11. Evaluate model
loss, accuracy = model.evaluate(
    {'input_ids': X_test, 'attention_mask': attention_test},
    y_test
)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 12. Predict and calculate F1-score
y_pred_logits = model.predict({'input_ids': X_test, 'attention_mask': attention_test}).logits
y_pred_probs = tf.nn.sigmoid(y_pred_logits).numpy()
y_pred = (y_pred_probs > 0.5).astype(int)
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1:.4f}")

# # 12. Predict and calculate F1-score
# y_pred_probs = model.predict({'input_ids': X_test, 'attention_mask': tokenized_data['attention_mask'][len(X_train):]})
# y_pred = (y_pred_probs > 0.5).astype(int)
# f1 = f1_score(y_test, y_pred)
# print(f"F1-score for {smell} smell: {f1:.4f}")

# 13. Save model
model.save('bert_mystery_guest_detector.keras')

# 1. Plot Loss over Epochs
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('bert_loss_over_epochs.png')
plt.show()

# 2. Plot Accuracy over Epochs
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('bert_accuracy_over_epochs.png')
plt.show()

# 3. Plot ROC Curves
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'{smell} (AUC = {roc_auc:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for {smell} Classification')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('bert_roc_curve.png')
plt.show()
