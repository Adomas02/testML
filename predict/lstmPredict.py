from keras.src.utils import pad_sequences

from lstm.implementation import tokenizer
from tensorflow.keras.models import load_model


# 10. Predicting on new data (example)
new_tests = ["""@Test
    public void testValidOnPolicyWithLimitAndRole() {
        properties.setKeyPrefix("prefix");
        Policy policy = getPolicy(1L, null);
        policy.getType().add(new Policy.MatchType(RateLimitType.ROLE, "user"));
        properties.getDefaultPolicyList().add(policy);
        properties.getPolicyList().put("key", Lists.newArrayList(policy));
        Set<ConstraintViolation<RateLimitProperties>> violations = validator.validate(properties);
        assertThat(violations).isEmpty();
    }"""]
new_seq = tokenizer.texts_to_sequences(new_tests)
new_pad = pad_sequences(new_seq, padding='post', maxlen=300)
predictions = model.predict(new_pad)
print(predictions)