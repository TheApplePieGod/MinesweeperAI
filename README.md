# Minesweeper AI

Various different [TensorFlow](https://www.tensorflow.org/) models and algorithms that play minesweeper.

# About
- Model 1
    - An implementation of reinforcement learning that takes the board as an input and attempts to predict the
    next best move on the board. This model and algorithm was not optimal.
- Model 2
    - An implementation of binary classification that attempts to classify a tile as a mine or not based on the
    surrounding tiles. This model was significantly more successful than the first.

# Run
- Install the required modules: `pip install pygame numpy tensorflow`
- Use Python 3 to run `main.py`. This will open a window and have the pre-trained AI play the game. Additionally,
you can run the model files directly to train the corresponding model.