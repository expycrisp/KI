from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def create_neural_network(num_layers, input_dim, units_per_layer, activation, output_units=1, output_activation='sigmoid'):
    """
    Creates a neural network with the specified number of layers using Keras.

    Parameters:
        num_layers (int): Number of hidden layers in the network.
        input_dim (int): Number of input features.
        units_per_layer (int): Number of units in each hidden layer.
        activation (str): Activation function for the hidden layers.
        output_units (int): Number of output units.
        output_activation (str): Activation function for the output layer.

    Returns:
        model (Sequential): A Keras Sequential model.
    """
    model = Sequential()

    # Add the input layer
    model.add(Dense(units_per_layer, input_dim=input_dim, activation=activation))

    # Add the hidden layers
    for _ in range(num_layers - 1):
        model.add(Dense(units_per_layer, activation=activation))

    # Add the output layer
    model.add(Dense(output_units, activation=output_activation))

    return model


print("#########################################################")

layers = input("Anzahl der Layers: ")
print("#########################################################")

neurons = input("Anzahl der Neuronen: ")
print("#########################################################")

activation = input("Aktuvierungsfunktion: ") # relu, sigmoid, tanh, softmax, linear
print("#########################################################")
epochs = input("Epochen: ") 
epochs = int(epochs)

layers = int(layers)
neurons = int(neurons)

class CustomLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, n, factor=0.5, min_lr=1e-6):
        super().__init__()
        self.n = n
        self.factor = factor
        self.min_lr = min_lr
        self.prev_loss = None

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:
            current_loss = logs.get('val_loss')
            if self.prev_loss is not None and current_loss >= self.prev_loss:
                old_lr = float(keras.backend.get_value(self.model.optimizer.lr))
                new_lr = max(old_lr * self.factor, self.min_lr)
                keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f"\nEpoch {epoch + 1}: Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
            self.prev_loss = current_loss








# Daten laden
#df = pd.read_csv("student_depression_dataset.csv")
df_c = pd.read_csv("/home/jan/Programmieren/Ordentlich/KI/Datasets/cleaned_student_depression_dataset.csv")
df = pd.read_csv('/home/jan/Programmieren/Ordentlich/KI/Datasets/cleaned_student_depression_dataset_outcome.csv')

# Sämtliche Daten
X = df_c.to_numpy(dtype=float)
y = df.to_numpy(dtype=float)



# Gesamtdaten aufteilen:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Jetzt nochmal Trainingsdaten in Training + Validation aufteilen
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)  # ergibt 60/20/20


# Definiere Netz
model = create_neural_network(layers, X.shape[1], neurons, activation)


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=1e-6)




# Training (80% Training, 20% Validierung)
custom_lr_scheduler = CustomLearningRateScheduler(n=5, factor=0.5, min_lr=1e-6)



model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy']
              )



history = model.fit(X_train, y_train,
                    epochs=epochs, 
                    validation_data=(X_val, y_val),
                    )#callbacks=[custom_lr_scheduler])

# Modell speichern (minimaler Aufwand)
model.save(f"/home/jan/Programmieren/Ordentlich/KI/Gespeicherte_Netze/Student_Depression/layers={layers}_neurons={neurons}_epochs={epochs}_activation={activation}.keras")


# Trainings- und Validierungsfehler plotten
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'], label='Train Loss')
plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.xticks(range(1, len(history.history['loss']) + 1))  # Ganze Zahlen für die Epochen
plt.legend()
plt.show()























print("#########################################################")



## TESTEN
print("Ergebnis für Testdaten: ")
loss, acc = model.evaluate(X_test, y_test)
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")



