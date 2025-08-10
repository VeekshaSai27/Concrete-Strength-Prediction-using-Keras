# 🏗 Concrete Strength Prediction using Keras

This project uses a simple **feed-forward neural network** built with **Keras** to predict the **compressive strength of concrete** based on its ingredients (cement, water, aggregates, etc.).

<img width="825" height="610" alt="image" src="https://github.com/user-attachments/assets/01b15639-01eb-4e9e-9388-884bb403a80c" />

##  Key Steps in the Workflow

- Load the dataset and split it into features (predictors) and target (strength).
- Normalize the input features for better training performance.
- Build a 3-layer neural network:
  - Input layer: accepts all feature columns.
  - Two hidden layers.
  - Output layer with no activation for regression output.
- Train the model using the **Adam** optimizer and **Mean Squared Error (MSE)** loss function.
- Evaluate by printing sample predictions.

<img width="1089" height="860" alt="image" src="https://github.com/user-attachments/assets/080f6fe1-05ea-4576-a9a2-3909dd021a71" />

...

<img width="913" height="636" alt="image" src="https://github.com/user-attachments/assets/8b9003ff-971f-4dbf-a367-9504f5d0731b" />

This is a beginner-friendly example of applying deep learning to a real-world regression problem.
