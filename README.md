# Gaussian Naive Bayes Model

This repository contains an implementation of a Gaussian Naive Bayes model for multiclass classification, along with training, prediction, and evaluation capabilities. The project is designed in a Jupyter Notebook (`main.ipynb`) and includes examples with the Iris dataset.

## Authors

Edouard Clocheret (edouard_clocheret@brown.edu) <br>
Shivam Hingorani (shivam_hingorani@brown.edu) <br>
Junhui Huang (junhui_huang@brown.edu) <br>
Eric Fan (guanghe_fan@brown.edu)

## Features
- **Gaussian Naive Bayes Classifier**
  - Train using maximum likelihood estimation.
  - Predict labels based on input features.
  - Calculate accuracy for evaluation.
- **Unit Tests**
  - Verify the implementation with various test cases.
- **Iris Dataset Example**
  - Demonstrates usage of the model with the Iris dataset.

## Dependencies
This project requires the following Python libraries:
- `numpy`
- `pandas`
- `pytest`
- `scikit-learn`

Install the dependencies using:
```bash
pip install numpy pandas pytest scikit-learn
```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
2. Run the cells to execute the code. The notebook includes:
   - Training the Gaussian Naive Bayes model.
   - Running unit tests.
   - Example usage with the Iris dataset.

## Example
### Training and Testing
The example uses the Iris dataset:
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GaussNaiveBayes()
model.train(X_train, y_train)

# Evaluate accuracy
accuracy = model.accuracy(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

### Unit Tests
Run the included unit tests:
```bash
pytest
```

## File Structure
- `src/final_project.ipynb`: Jupyter Notebook containing the Gaussian Naive Bayes implementation, unit tests, and example usage.

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

## Acknowledgments
- The Iris dataset is provided by the scikit-learn library.
- The Gaussian Naive Bayes implementation is inspired by foundational machine learning techniques.