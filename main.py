import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



# Read in the insurance dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# Check out the insurance dataset
print(insurance.head())

# Numerical encoding for sex, smoker, region using one hot encoding
insurance_one_hot = pd.get_dummies(insurance)
print("\nDataframe after hot encoding with get_dummies\n", insurance_one_hot.head())

# Create X & y values (features and label)
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build a neural network
tf.random.set_seed(42)
# 1. Create a model
"""insurance_model = tf.keras.Sequential([tf.keras.layers.Dense(20),
                                       tf.keras.layers.Dense(10),
                                      tf.keras.layers.Dense(1)])"""
insurance_model = tf.keras.models.Sequential()
insurance_model.add(tf.keras.layers.Dense(units=20, activation='relu'))
insurance_model.add(tf.keras.layers.Dense(units=10, activation='relu'))
insurance_model.add(tf.keras.layers.Dense(units=1))


# 2. Compile model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        metrics=["mae"])

# 3. Fit the model
history = insurance_model.fit(X_train, y_train, epochs=150)

# Check the results of the insurance model
print(insurance_model.evaluate(X_test, y_test))

pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()
