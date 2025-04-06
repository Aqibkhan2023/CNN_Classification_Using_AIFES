# This script converts the MNIST data set to C header files. The data set is stored as a flattened
# one-dimensional array for each of training and test data, rather than a two-dimensional array.

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(float)/255.0
x_test = x_test.astype(float)/255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)

def generate_train_data():
    '''
    Generates two header files. The first one contains the training data as a one-dimensional array.
    Instead of [NUM_TRAINING_DATA][784], it is declared as [NUM_TRAINING_DATA*784].
    The second header file contains the training labels as a one-dimensional array ([NUM_TRAINING_DATA*10]).
    '''
    total_train_values = NUM_TRAINING_DATA * 784
    with open("MNIST_training_data.h", "w") as f:    
        f.write("float MNIST_training_data[" + str(total_train_values) + "] = {\n")
        values = []
        for i in range(NUM_TRAINING_DATA):
            x_train_flat = x_train[i].flatten()
            # Append each value formatted as float literal
            values.extend([f"{val}f" for val in x_train_flat])
        f.write(", ".join(values))
        f.write("\n};\n")
        
    total_train_labels = NUM_TRAINING_DATA * 10
    with open("MNIST_training_data_label.h", "w") as f:    
        f.write("float MNIST_training_data_label[" + str(total_train_labels) + "] = {\n")
        values = []
        for i in range(NUM_TRAINING_DATA):
            # For each sample, append one-hot label values formatted as float literal
            values.extend([f"{val}f" for val in y_train[i]])
        f.write(", ".join(values))
        f.write("\n};\n")

def generate_test_data():
    '''
    Generates two header files. The first one contains the test data as a one-dimensional array.
    It is declared as [NUM_TEST_DATA*784]. The second header file contains the test labels as a one-dimensional
    array ([NUM_TEST_DATA*10]).
    '''
    total_test_values = NUM_TEST_DATA * 784
    with open("MNIST_test_data.h", "w") as f:    
        f.write("float MNIST_test_data[" + str(total_test_values) + "] = {\n")
        values = []
        for i in range(NUM_TEST_DATA):
            x_test_flat = x_test[i].flatten()
            values.extend([f"{val}f" for val in x_test_flat])
        f.write(", ".join(values))
        f.write("\n};\n")
    
    total_test_labels = NUM_TEST_DATA * 10
    with open("MNIST_test_data_label.h", "w") as f:    
        f.write("float MNIST_test_data_label[" + str(total_test_labels) + "] = {\n")
        values = []
        for i in range(NUM_TEST_DATA):
            values.extend([f"{val}f" for val in y_test[i]])
        f.write(", ".join(values))
        f.write("\n};\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate MNIST data as C header files.")
    parser.add_argument("--generate-train", action="store_true", help="Generate training data header files.")
    parser.add_argument("--generate-test", action="store_true", help="Generate test data header files.")
    parser.add_argument("--num-training-data", type=int, default=30000, help="Number of training data samples to generate (max 60000).")
    parser.add_argument("--num-test-data", type=int, default=10000, help="Number of test data samples to generate (max 10000).")
    args = parser.parse_args()

    global NUM_TRAINING_DATA, NUM_TEST_DATA
    NUM_TRAINING_DATA = args.num_training_data
    NUM_TEST_DATA = args.num_test_data

    if args.generate_train:
        generate_train_data()
    if args.generate_test:
        generate_test_data()

if __name__ == "__main__":
    main()