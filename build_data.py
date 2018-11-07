import numpy as np
import argparse
from sklearn.utils import resample

from dataset import load_vocab, Dataset
from constants import DATA, UNK
from utils import Timer


def get_embedding_vocab_text(we_file):
    vocab = set()
    with open(we_file, "r", encoding="utf8") as glove:
        lines = glove.readlines()
        for line in lines[1:]:
            word, vector = tuple(line.split(" ", 1))
            vocab.add(word)
    return vocab


def export_trimmed_embeddings_text(we_file, vocab, trimmed_filename, dim):
    embeddings = np.zeros([len(vocab) + 1, dim])
    with open(we_file, "r", encoding="utf8") as glove:
        lines = glove.readlines()
        for line in lines[1:]:
            word, vector = tuple(line.split(" ", 1))
            vector = np.fromstring(vector, sep=" ")

            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = vector
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    vocab = list(vocab)
    vocab.sort()
    with open(filename, "w", encoding='utf8') as f:
        for i, word in enumerate(vocab):
            f.write("{}\n".format(word))
        f.write(UNK)
    print("- done. {} tokens".format(len(vocab)))


def build_data(dataset, word_embedding):
    d = Dataset(dataset)
    data_vocab_words, data_vocab_char = d.get_vocabs()
    embed_vocab_words = get_embedding_vocab_text(word_embedding)
    vocab_words = data_vocab_words & embed_vocab_words

    oov_words = list(data_vocab_words - vocab_words)
    print("OOV words: {}".format(len(oov_words)))
    print("Examples: {}".format("\t".join(resample(oov_words, n_samples=20, replace=False))))

    write_vocab(vocab_words, DATA + "all_words.txt")
    write_vocab(data_vocab_char, DATA + "all_chars.txt")

    vocab_words = load_vocab(DATA + "all_words.txt")
    export_trimmed_embeddings_text(we_file=word_embedding,
                                   vocab=vocab_words,
                                   trimmed_filename=DATA + "embedding_data",
                                   dim=300)


def main(dataset, word_embedding):
    t = Timer()
    t.start("Prepare the data")
    build_data(dataset, word_embedding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build necessary data for model training and evaluating.')
    parser.add_argument('dataset', help="the name of the dataset that the model will be trained on, e.g: original")
    parser.add_argument('word_embedding', help="path to the word embedding pre-trained model, i.e: "
                                               "data/w2v_model/wiki.en.vec")
    args = parser.parse_args()
    main(args.dataset, args.word_embedding)
