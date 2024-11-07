
**1. NN_model.py** provides a robust framework to classify samples in an omics dataset using a neural network.

Here's a summary of the code:

1. Data Loading and Preprocessing: The dataset is loaded and split into amino acid and protein features, then each subset is scaled independently using MinMax scaling to normalize the values between 0 and 1. Labels are encoded into binary format (e.g., 0 for "Control" and 1 for "CRC").
2. Splitting the Data: The data is divided into training and test sets, ensuring balanced class representation in both.
3. Neural Network Model Definition: A simple neural network model is created with three hidden layers, each followed by a dropout layer to reduce overfitting. The final layer has a sigmoid activation function, suited for binary classification, as it outputs a probability.
4. Model Training: The model is compiled with the binary cross-entropy loss function and an optimizer (adam). Early stopping is applied to prevent overfitting by stopping the training if validation loss doesn’t improve after 10 epochs. The model is trained using 80% of the training data, with 20% held out for validation during training.
5. Evaluation and Performance Metrics: The model's accuracy on the test set is calculated and printed. A ROC-AUC curve is plotted based on test set predictions to visualize the trade-off between sensitivity and specificity, with the AUC score indicating the model's performance.


**2. CNN-based integration.ipynb** This code implements a Convolutional Neural Network (CNN) to classify multi-omics data.

1. **Data Preparation**:
   - Loads the dataset, separates the target labels (`Label` column), and scales the features using `MinMaxScaler`.
   - Encodes labels into a binary format (`Control` as 0 and `CRC` as 1), then transforms them into a categorical format for compatibility with the CNN.

2. **Data Reshaping**:
   - Reshapes the data to a 2D grid structure (`20x26`), simulating image-like data with a single grayscale channel.

3. **Data Splitting**:
   - Splits the reshaped data into training and testing sets.

4. **CNN Model Definition**:
   - Builds a CNN with two convolutional layers followed by max-pooling and dropout layers to reduce overfitting.
   - A final dense layer with `softmax` activation is used for binary classification.

5. **Model Compilation and Training**:
   - Compiles the model using the Adam optimizer and binary cross-entropy loss.
   - Trains the model with early stopping based on validation loss to prevent overfitting.

6. **Evaluation and Visualization**:
   - Evaluates the model’s performance on the test set, printing the test accuracy.
   - Plots the training and validation accuracy over epochs to visualize the model’s learning progress. 

**3. Network-based_NetworkX.ipynb** constructing and visualizing within-omics networks for amino acid and protein data, enabling a network-based integration of omics data.

Data Loading and Preprocessing: The omics data is loaded, and the columns are split into amino acids and protein features.

Within-Omics Network Construction:
Two separate graphs (G_amino for amino acids and G_prot for proteins) are created using the NetworkX library.
For each feature pair in both data types, an edge is added between nodes (features) if their correlation exceeds a set threshold (e.g., 0.7). This step captures high-correlation relationships, indicating potential interactions between features.

Network Visualization:
One of the networks (e.g., the amino acid network) is visualized using Matplotlib and NetworkX.
Nodes represent features, and edges represent high-correlation relationships. Visualizing this network provides insights into the connectivity and potential functional relationships within the omics layer.





