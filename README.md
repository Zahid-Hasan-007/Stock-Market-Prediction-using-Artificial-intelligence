# Stock Market Prediction using Machine Learning

This project aims to predict stock market prices using Machine Learning techniques, with a focus on the Long Short-Term Memory (LSTM) algorithm, which is a type of recurrent neural network (RNN) that is well-suited for sequence prediction problems.

## Overview

- **Algorithm**: LSTM (Long Short-Term Memory) Neural Networks
- **Primary Language**: Python
- **Libraries Used**: TensorFlow, Keras, Pandas, NumPy, Matplotlib
- **Dataset**: Twitter stock market data

## Project Structure

- `data/`: Contains the dataset used for training and testing the model.
- `models/`: Includes saved models and model checkpoints.
- `src/`: Contains the source code for data preprocessing, model training, and evaluation.
- `README.md`: You are here!

## Setup

1. **Install Dependencies**: Install the necessary libraries using `pip install -r requirements.txt`.
2. **Dataset**: Obtain historical stock market data for the desired stocks.
3. **Training**: Use the `train.py` script to train the LSTM model on the dataset.
4. **Evaluation**: Evaluate the model using the `evaluate.py` script.
5. **Prediction**: Use the trained model to make predictions using the `predict.py` script.

## Usage

1. **Training**: Run the `train.py` script to train the LSTM model on the dataset.
2. **Evaluation**: Use the `evaluate.py` script to evaluate the model's performance.
3. **Prediction**: Use the `predict.py` script to make predictions for future stock prices.

## References

- [Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
