# CNN Classification Using AIfES

This project demonstrates a Convolutional Neural Network (CNN) implementation using the AIfES framework. The model, detailed in `main.c`, classifies input images and prints out an accuracy score after execution.

## Project Structure

- **src/**  
  Contains the source code files:  
  - `main.c`: The entry point of the application where the CNN model is executed.
  - `mnist_to_c.py`: A script to convert the MNIST dataset into C header files. This script flattens the data arrays for training and testing.

- **AIfES_for_Arduino/**  
  Houses the AIfES core and CNN-related files used by the project.

## Prerequisites

- **Code::Blocks**  
  You need to have Code::Blocks installed to run this code. The project is located in the **CNN** folder. Open the project in Code::Blocks, build, and compile it. Running the compiled binary will display the accuracy of the CNN model using AIfES.

- **Python Environment**

  After cloning the repository, set up a Python virtual environment and, once activated, install the required packages using the command below:

  ```bash
  pip install -e .
  ```

  This will install the package in editable mode, ensuring that all the Python dependencies (including TensorFlow for the MNIST data conversion script) are correctly set up.

## Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Aqibkhan2023/CNN_Classification_Using_AIFES
   cd CNN_Classification_Using_AIFES
   ```

2. **Set Up the Python Environment**

   Create and activate a Python virtual environment (e.g., using `venv`):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Mac
   pip install -e .
   ```

3. **Convert the MNIST Dataset (Optional)**

   To generate C header files from the MNIST dataset, run the following command while being in the root directory of the project:

   ```bash
   mnist-c --generate-train --generate-test --num-training-data 1000 --num-test-data 200
   ```

4. **Build and Run in Code::Blocks**

   - Open Code::Blocks
   - Open the project located in the **CNN** folder.
   - Build and compile the project.
   - Run the executable. The program will display the accuracy of the CNN model during runtime.

## Troubleshooting

- Ensure that Code::Blocks is correctly installed and configured on your system.
- Verify that your Python virtual environment is active before installing the dependencies.
- If changes are made to the MNIST data conversion process, re-run the `mnist_to_c.py` script to update the header files used by the C code.

## Acknowledgments

- [AIfES](https://github.com/Fraunhofer-IMS/AIfES_for_Arduino) framework for inspiration.
- TensorFlow for the MNIST dataset utilities.

Happy Coding!