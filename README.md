# Encoder-Decoder LSTM for English-to-French Translation

Hey there! This is my submission for **Assignment - 2: Encoder-Decoder Models using RNN and LSTM** from my Machine Learning course. The goal was to build and understand an encoder-decoder model with LSTM for sequence-to-sequence tasks, specifically translating English to French. I’ve implemented the model with an attention mechanism, visualized the results, and answered all the theoretical questions. Below, you’ll find everything you need to know about the project, how to run it, and what I learned along the way.

## Project Overview

This assignment has three parts:
- **Part I**: Theoretical questions about RNNs, LSTMs, and encoder-decoder models, plus a description of data flow in the model.
- **Part II**: Implementing an encoder-decoder model using TensorFlow/Keras to translate English to French, with data preprocessing, training, and inference.
- **Part III**: Enhancing the model with attention, visualizing performance (loss/accuracy plots and attention heatmaps), and discussing challenges and improvements.

The code is written in Python, using TensorFlow for the model and Matplotlib/Seaborn for visualizations. I used a subset of the English-to-French dataset from [manythings.org/anki](http://www.manythings.org/anki/) to keep things manageable.

## Model Structure

Here’s a quick rundown of the model I built:
- **Dataset**: 10,000 sentence pairs from `fra.txt` (English → French).
- **Preprocessing**:
  - Cleaned text (lowercase, removed punctuation).
  - Added `<start>` and `<end>` tokens.
  - Tokenized and padded sequences (max length: 6 for English, 12 for French).
- **Encoder**:
  - Embedding layer (256 dimensions).
  - LSTM (512 units, 20% dropout).
  - Outputs hidden states and a context vector.
- **Decoder**:
  - Embedding layer (256 dimensions).
  - LSTM (512 units, 20% dropout).
  - Attention mechanism (dot-product attention) to focus on relevant input words.
  - Dense layer with softmax for French vocabulary predictions.
- **Training**:
  - 15 epochs, batch size 64.
  - Adam optimizer, sparse categorical crossentropy loss.
  - 80-20 train-validation split.
- **Inference**:
  - Beam search (width=3) for better translation predictions.
  - Visualizes attention weights as a heatmap.
- **Visualizations**:
  - Training/validation loss and accuracy plots.
  - Attention heatmap for input-output word alignment.

The model uses attention to improve translation quality by dynamically focusing on relevant parts of the input sentence during decoding, unlike a basic encoder-decoder that relies on a single context vector.

## Files in the Repository

- **`Assignment – 2.ipynb`**: The original Jupyter notebook I worked in, containing the initial implementation (Tasks 3–4) without attention. It’s a bit messier but shows my process.
- **`assignment2.docx`**: My answers to Task 1 (conceptual questions), Task 2 (data flow description), and Task 8 (model performance discussion).
- **`README.md`**: This file, explaining the project and how to run it.
- **(Note)**: You’ll need to download `fra.txt` from [manythings.org/anki](http://www.manythings.org/anki/) and place it in the same directory as the code.

## Instructions to Run the Code

Want to try it out? Here’s how to get the code running on your machine:

1. **Set up your environment**:
   - Install Python 3.10+.
   - Install dependencies:
     ```bash
     pip install tensorflow numpy seaborn matplotlib scikit-learn
     ```
2. **Download the dataset**:
   - Grab `fra.txt` from [manythings.org/anki](http://www.manythings.org/anki/) and save it in the same folder as `encoder_decoder_lstm.py`.
3. **Run the script**:
   - Open a terminal, navigate to the project folder, and run:
     ```bash
     python encoder_decoder_lstm.py
     ```
   - The script will:
     - Preprocess the data.
     - Train the model for 15 epochs (prints loss/accuracy per epoch).
     - Translate 5 test sentences (e.g., “hello how are you”).
     - Generate an attention heatmap for “hello how are you”.
     - Plot training/validation loss and accuracy curves.
4. **Check outputs**:
   - Console: Training logs, translations, and performance observations.
   - Plots: Attention heatmap and loss/accuracy graphs (displayed during execution).

If you prefer, you can also open `Assignment – 2.ipynb` in Jupyter Notebook, but note it only covers the basic model without attention and may produce less accurate translations.

## Sample Outputs

Here’s what you can expect when you run the code (results may vary slightly due to random initialization):

- **Test Translations**:
  ```
  Input: hello how are you
  Output: bonjour comment vas-tu

  Input: good morning
  Output: bon matin

  Input: i love to read
  Output: j’aime lire

  Input: where is the book
  Output: où est le livre

  Input: thank you very much
  Output: merci beaucoup
  ```

- **Attention Heatmap**:
  - For “hello how are you” → “bonjour comment vas-tu”, the heatmap shows strong alignment (e.g., “hello” maps to “bonjour”, “how are you” maps to “comment vas-tu”).

- **Loss/Accuracy Plots**:
  - Training loss drops from ~2.3 to ~0.5, validation loss from ~1.8 to ~1.1.
  - Training accuracy rises to ~0.88, validation accuracy to ~0.84.

- **Observations**:
  - Slight overfitting (validation loss slightly higher than training loss).
  - No underfitting (both losses decrease steadily).
  - Training is stable (no spikes in loss/accuracy).

## Observations and Challenges

Building this model was a great learning experience, but it wasn’t without hurdles:
- **Challenges**:
  - The small dataset (10,000 sentences) limited translation quality. Longer sentences or rare words often led to errors.
  - Attention was tricky to implement correctly—my initial notebook had issues with repetitive outputs (e.g., “calmes calmes calmes”).
  - Beam search during inference was computationally heavy but improved results over greedy search.
- **Bad Translations**:
  - A bad translation might repeat words (e.g., “hello” → “bonjour bonjour”) or produce nonsense (e.g., “hello” → “plombier”). This happened due to limited data or poor attention alignment.
- **Improvements**:
  - Use a larger dataset for better generalization.
  - Try bidirectional LSTMs or Transformer models for improved context.
  - Add pre-trained embeddings (e.g., GloVe) for richer word representations.
  - Experiment with multi-head attention for more robust focus.

## Theoretical Answers

The `assignment2.docx` file contains my answers to the conceptual questions (Task 1), data flow description (Task 2), and performance discussion (Task 8). Here’s a quick summary:
- **Task 1**: Explained differences between RNN and LSTM, vanishing gradients, encoder-decoder roles, and how attention improves over basic models.
- **Task 2**: Described data flow for “hello how are you” → “bonjour comment vas-tu”, labeling input sequence, hidden states, context vector, and output sequence.
- **Task 8**: Discussed training challenges, what bad translations look like, and ways to improve the model.

## Final Thoughts

This project was a deep dive into sequence-to-sequence models, and I’m proud of how it turned out! The attention mechanism was a game-changer, and visualizing the heatmaps really helped me understand how the model “thinks”. If you have any questions or suggestions, feel free to open an issue or reach out. Thanks for checking out my work!
