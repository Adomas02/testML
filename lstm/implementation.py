import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')  # Force backend fix
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.utils.class_weight import compute_class_weight

# 1. Load dataset
data = pd.read_csv(r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file.csv')

# 2. Convert labels properly (TRUE/FALSE → 1/0)
data['isEagerTestManual'] = data['isEagerTestManual'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0})

# 3. Prepare input and output
X = data['method_code'].astype(str).values
y = data['isEagerTestManual'].values

# 4. Tokenize and pad
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(sequences, padding='post', maxlen=300)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# 6. Check imbalance
print("Training set positives:", np.sum(y_train))
print("Training set negatives:", len(y_train) - np.sum(y_train))

# 7. Compute class weights
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight = {0: weights[0], 1: weights[1]}
print(f"Class weights: {class_weight}")

# 8. Build model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=300),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# 9. Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 10. Train model with class weights
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, class_weight=class_weight)

# 11. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 12. Predict and calculate F1-score
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)
f1 = f1_score(y_test, y_pred)
print(f"F1-score for Mystery Guest smell: {f1:.4f}")

# 13. Save model
model.save('lstm_mystery_guest_detector.keras')

# --- 🎯 Performance Visualization ---

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
plt.savefig('loss_over_epochs.png')  # save the figure
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
plt.savefig('accuracy_over_epochs.png')  # save the figure
plt.show()

# 3. Plot ROC Curves
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Eager Test (AUC = {roc_auc:.2f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Eager Test Classification')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve_eager_test.png')  # Save the figure
plt.show()

# 10. Predicting on new data (example)
# new_tests = ["""@Test
#     public void testValidOnPolicyWithLimitAndRole() {
#         properties.setKeyPrefix("prefix");
#         Policy policy = getPolicy(1L, null);
#         policy.getType().add(new Policy.MatchType(RateLimitType.ROLE, "user"));
#         properties.getDefaultPolicyList().add(policy);
#         properties.getPolicyList().put("key", Lists.newArrayList(policy));
#         Set<ConstraintViolation<RateLimitProperties>> violations = validator.validate(properties);
#         assertThat(violations).isEmpty();
#     }"""]
# new_seq = tokenizer.texts_to_sequences(new_tests)
# new_pad = pad_sequences(new_seq, padding='post', maxlen=300)
# predictions = model.predict(new_pad)
# print(predictions)
