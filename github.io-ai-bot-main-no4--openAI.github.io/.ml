import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Define the training data
training_data = [
    ("휴먼로이드로봇로봇은 항상 인간을 보호, 보조 하여 협력하며 판단을 도와준다", 1),
    ("휴먼로이드로봇은 법을 준수한다", 2),
    ("인간의 인권을 존중한다", 3)
]

# Preprocessing the training data
X_train = [data[0] for data in training_data]
y_train = [data[1] for data in training_data]

# Build the model
input_layer = Input(shape=(1,), dtype=tf.string)
embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=16)(input_layer)
flatten = tf.keras.layers.Flatten()(embedding)
hidden_layer = Dense(32, activation='relu')(flatten)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile and fit the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

# Use the model for prediction
new_input = ["휴먼로이드로봇로봇은 인간을 존중합니다"]
prediction = model.predict(new_input)
print(prediction)
