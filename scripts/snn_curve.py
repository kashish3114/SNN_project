import matplotlib.pyplot as plt

epochs = list(range(1, 16))
accuracy = [
    0.6926, 0.7146, 0.7086, 0.7126, 0.7138,
    0.7146, 0.7143, 0.7115, 0.7140, 0.7057,
    0.7109, 0.7097, 0.7192, 0.7003, 0.7166
]

plt.plot(epochs, accuracy, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("SNN Training Curve")
plt.grid()

plt.savefig("snn_training_curve.png")
plt.show()