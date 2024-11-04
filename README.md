# Logistic Regression with Gradient Descent

This project implements logistic regression using gradient descent, optimized for large and high-dimensional datasets. The implementation meets strict timing and accuracy constraints, designed for use in competitive environments.

## Project Parameters and Constraints

- **Input**: Given a matrix \( X \) and label \( Y \), perform gradient descent to train a logistic regression model.
- **Test Cases**: The program must handle **10 independent test cases**, with each case contributing equally to the overall score.
- **Compilation Time**:
  - A **failed compilation** occurs if it does not complete within **1 minute**.
- **Execution Time**:
  - A test case is marked **incorrect** if it does not finish within **2 minutes**.
- **Training Accuracy**: 
  - The logistic regression model must achieve a minimum training accuracy of **60%**.
- **Ranking**:
  - The **summation of execution times** across all 10 test cases is used to rank correct solutions, rewarding efficient implementations.

## Features

- **Matrix Generation**: Generates matrix \( X \) with given parameters, normalized for logistic regression input.
- **Logistic Regression Implementation**: Gradient descent-based training, optimized to meet both accuracy and time constraints.
- **Output**: Produces \( D \) logistic regression parameters in a structured output format.

## File Descriptions

- **`compile.sh`**: Script to compile the main logistic regression program.
- **`run.sh`**: Script to execute the logistic regression program, managing inputs and output.
- **`input_generator.cc`**: Generates matrix \( X \) based on initial parameters.
- **`lr.cc`**: Core implementation of logistic regression with gradient descent.
- **`lr_exec.exe`**: Compiled binary for logistic regression execution.

## Setup and Usage

**Compile**:
   ```bash
   ./compile_generator.sh
