import pickle
import numpy as np
import argparse
from sklearn.utils import shuffle, resample
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import Timer
from constants import DATA, UNK, CONVERTED_DATA


def get_trimmed_embeddings(filename):
    with np.load(filename) as data:
        return data["embeddings"]


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename, encoding='utf8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx + 1  # preserve idx 0 for pad_tok
    return d


def pad_sequences(sequences, pad_tok, nlevels=1, fixed_word_len=None, fixed_sent_len=None):
    def _pad_sequences(seqs, tok, max_len):
        seq_padded, seq_len = [], []

        for seq in seqs:
            seq = list(seq)
            seq_ = seq[:max_len] + [tok] * max(max_len - len(seq), 0)
            seq_padded += [seq_]
            seq_len += [min(len(seq), max_len)]

        return seq_padded, seq_len

    if nlevels == 1:
        if fixed_sent_len:
            max_length = fixed_sent_len
        else:
            max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    else:
        if fixed_word_len:
            max_length_word = fixed_word_len
        else:
            max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    return sequence_padded, sequence_length


class TfIdfCompressor:
    def __init__(self, size, train_data, dev_data):
        self.size = size
        self.train_data = train_data
        self.dev_data = dev_data

        self.tf_idf = TfidfVectorizer(norm='l2',
                                      smooth_idf=False,
                                      tokenizer=lambda x: x.split(" "),
                                      analyzer='word',
                                      ngram_range=(1, 1))
        self.tfidf_vocabs = None
        self.reversed_tfidf_vocabs = None
        self.tfidf_docs = None

    def _get_all_contexts(self):
        contexts = []
        indexes = list(self.train_data.keys())
        for idx in indexes:
            contexts.append(self.train_data[idx]['c'])

        indexes = list(self.dev_data.keys())
        for idx in indexes:
            contexts.append(self.dev_data[idx]['c'])

        return contexts

    def train(self):
        t = Timer()
        t.start("Training TF-IDF compressor")

        all_context = self._get_all_contexts()
        self.tfidf_docs = self.tf_idf.fit_transform(all_context)
        self.tfidf_vocabs = self.tf_idf.vocabulary_
        self.reversed_tfidf_vocabs = {}
        for k, v in self.tfidf_vocabs.items():
            self.reversed_tfidf_vocabs[v] = k
        t.stop()

    def compress(self, context):
        r = self.tf_idf.transform([context])
        len_r = r.nnz
        r = r.toarray()[0]
        top_idx = r.argsort()[::-1][:min(self.size, len_r)]
        context_word_set = [self.reversed_tfidf_vocabs[idx] for idx in top_idx]
        context_words = context.split(" ")
        compressed_context = " ".join([w for w in context_words if w in context_word_set])
        return compressed_context


class Dataset:
    def __init__(self, dataset, num_ques_sample=3):
        self.dataset = dataset
        self.num_ques_sample = num_ques_sample

        self.train_examples = None
        self.dev_examples = None

        self._load_vocabs()
        self.max_word_length = 0
        self.max_sent_length = 0
        self.max_negative_set = 0

        self.train_data, self.dev_data = self._load_data()
        self.tfidf_compressor = TfIdfCompressor(30, self.train_data, self.dev_data)

    def _load_data(self):
        with open(DATA + self.dataset + "/" + CONVERTED_DATA, 'rb') as f:
            train_data = pickle.load(f)
            dev_data = pickle.load(f)
        return train_data, dev_data

    def create_pairwise(self):
        self.tfidf_compressor.train()

        t = Timer()
        t.start("Creating pair-wise dataset")

        train_c_word_set, train_c = self.get_all_c_word_set(self.train_data)
        dev_c_word_set, dev_c = self.get_all_c_word_set(self.dev_data)

        train_examples = self.get_examples(self.train_data, train_c_word_set, train_c)
        dev_examples = self.get_examples(self.dev_data, dev_c_word_set, dev_c)

        t.stop()
        return train_examples, dev_examples

    def save_pairwise(self, file_name, data):
        t = Timer()
        t.start("Saving pair-wise dataset", verbal=True)
        print("Max sentence length: {}\nMax word length: {}".format(self.max_sent_length, self.max_word_length))
        print("Max negative set size: {}".format(self.max_negative_set))
        print("Number of training examples: {}\nNumber of developing examples: {}".format(len(data[0]), len(data[1])))
        with open(file_name, 'wb') as f:
            pickle.dump((self.max_word_length, self.max_sent_length), f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(data[0], f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(data[1], f, pickle.HIGHEST_PROTOCOL)
        t.stop()

    def load_data_pairwise(self, file_name):
        with open(file_name, 'rb') as f:
            self.max_word_length, self.max_sent_length = pickle.load(f)
            self.train_examples = pickle.load(f)
            self.dev_examples = pickle.load(f)

    def create_train_examples(self, model):
        train_examples = []
        for e in self.train_examples:
            if self.num_ques_sample >= len(e[0]):
                ques_samples = e[0]
            else:
                ques_samples = resample(e[0], replace=False, n_samples=self.num_ques_sample)

            if model.k_neg >= len(e[2]):
                neg_samples = e[2]
            else:
                neg_samples = resample(e[2], replace=False, n_samples=model.k_neg)

            for q in ques_samples:
                for cn in neg_samples:
                    train_examples.append([q, e[1], cn])
        return shuffle(train_examples)

    def process_sent(self, text, update_max=True):
        words = text.split()

        char_ids = []
        word_ids = []
        for w in words:
            c_ids = []
            # get chars of word
            for char in w:
                # ignore chars out of vocabulary
                if char in self.vocab_chars:
                    c_ids.append(self.vocab_chars[char])

            char_ids.append(c_ids)
            if update_max and len(c_ids) > self.max_word_length:
                self.max_word_length = len(char_ids)

            if w in self.vocab_words:
                word_ids.append(self.vocab_words[w])
            else:
                word_ids.append(self.vocab_words[UNK])

        if update_max and len(word_ids) > self.max_sent_length:
            self.max_sent_length = len(word_ids)

        # return tuple char ids, word id
        return char_ids, word_ids

    def _load_vocabs(self):
        self.vocab_words = load_vocab(DATA + "all_words.txt")
        self.vocab_chars = load_vocab(DATA + "all_chars.txt")

    def get_examples(self, data, all_c_word_set, c):
        examples = []
        indexes = list(data.keys())

        for idx in indexes:
            compressed_c = self.tfidf_compressor.compress(data[idx]['c'])
            c_word_set = set(data[idx]['c'].split(" "))
            q_word_set = set()
            questions = []
            for q in data[idx]['qs']:
                q_word_set.update(q.split(" "))
                questions.append(self.process_sent(q))

            neg_examples = []
            for i in range(len(all_c_word_set)):
                if all_c_word_set[i] != c_word_set and len(q_word_set.intersection(all_c_word_set[i])) > 0:
                    compressed_cn = self.tfidf_compressor.compress(c[i])
                    neg_examples.append(self.process_sent(compressed_cn))

            examples.append([questions,
                             self.process_sent(compressed_c),
                             neg_examples])
            if len(neg_examples) > self.max_negative_set:
                self.max_negative_set = len(neg_examples)
        return shuffle(examples)

    def get_vocabs(self):
        vocab_word = set()
        vocab_char = set()

        for data in [self.train_data, self.dev_data]:
            for _, v in data.items():
                context_tokens = v['c'].split()
                question_tokens = []
                for q in v['qs']:
                    question_tokens += q.split()

                vocab_word.update(context_tokens)
                vocab_word.update(question_tokens)
                for t in context_tokens:
                    vocab_char.update(t)
                for t in question_tokens:
                    vocab_char.update(t)

        print("- Done. {} tokens".format(len(vocab_word)))
        return vocab_word, vocab_char

    @staticmethod
    def get_all_c_word_set(data):
        # get all contexts
        all_c_word_set = []
        all_c = []
        indexes = list(data.keys())
        indexes.sort()
        for idx in indexes:
            c_word_set = set(data[idx]['c'].split(" "))
            all_c_word_set.append(c_word_set)
            all_c.append(data[idx]['c'])
        return np.array(all_c_word_set), np.array(all_c)

    @staticmethod
    def get_num_chars():
        path = DATA + "all_chars.txt"
        with open(path, 'r', encoding='utf8') as f:
            d = f.readlines()
        return len(d)


def main(dataset, data_file):
    d = Dataset(dataset)
    pairwise_data = d.create_pairwise()
    d.save_pairwise(DATA + "{}/{}.pickle".format(dataset, data_file), pairwise_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare training data for model training and evaluating.')
    parser.add_argument('dataset', help="the name of the dataset that the model will be trained on, e.g: extend")
    parser.add_argument('data_file', help="the name of the dataset saved file, e.g: train_dev_pairwise_compressed")

    args = parser.parse_args()
    main(args.dataset, args.data_file)
