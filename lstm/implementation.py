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
from sklearn.metrics import roc_curve, auc


# 1. Load your dataset
data = pd.read_csv(r'C:\Users\kazen\PycharmProjects\testML\repo\merged_file.csv')

# 2. Prepare input and output
X = data['method_code'].astype(str).values
y = data[['isEagerTestManual', 'isMysteryGuestManual', 'isResourceOptimismManual', 'isTestRedundancyManual']].astype(int).values

# 3. Tokenize and pad
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(sequences, padding='post', maxlen=300)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# 5. Build model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=300),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(4, activation='sigmoid')  # multilabel
])

# 6. Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 7. Train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 8. Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 9. Save
model.save('lstm_code_smell_detector.h5')

# --- 🎯 Add Performance Diagrams Below ---

# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 🎯 ROC Curve for each label
y_pred = model.predict(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
smell_labels = ['Eager Test', 'Mystery Guest', 'Resource Optimism', 'Test Redundancy']

for i in range(4):  # for each output label
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(4):
    plt.plot(fpr[i], tpr[i], label=f'{smell_labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Code Smell')
plt.legend(loc="lower right")
plt.grid(True)
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
