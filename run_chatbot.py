import argparse
import numpy as np
import tensorflow as tf

from faq_model import NtuModel
from pre_process import PreProcess
from dataset import Dataset, get_trimmed_embeddings, pad_sequences
from constants import DATA, TRAINED_MODELS

seed = 13
np.random.seed(seed)


class Inference:
    def __init__(self, model_name, dataset):
        self.model_name = TRAINED_MODELS + model_name + "/"
        self.dataset = dataset

        self.data = Dataset(self.dataset)
        self.data.tfidf_compressor.train()

        self.model = self._load_model()
        self.pre_process = PreProcess()

        idx = list(self.data.train_data.keys())
        idx.sort()
        self.train_c_word_set, self.train_c = self.data.get_all_c_word_set(self.data.train_data)
        self.all_train_contexts = np.array([self.data.train_data[i]['context'] for i in idx])
        self.related_questions = np.array([self.data.train_data[i]['qs'] for i in idx])

    def _load_model(self):
        # load model
        num_chars = self.data.get_num_chars()

        embeddings = get_trimmed_embeddings(DATA + "embedding_data.npz")

        model = NtuModel(model_name=self.model_name, embeddings=embeddings, num_chars=num_chars,
                         batch_size=32, early_stopping=False, k_neg=0)
        model.build()
        saver = tf.train.Saver()
        saver.restore(model.sess, tf.train.latest_checkpoint(self.model_name))

        return model

    def get_answer(self, question):
        question_example = self.pre_process.process(question, remove_stop_words=False)
        q_word_set = set(question_example)
        question_example = self.data.process_sent(" ".join(question_example))

        filtered_idx = []
        for i in range(len(self.train_c_word_set)):
            if len(q_word_set.intersection(self.train_c_word_set[i])) > 0:
                filtered_idx.append(i)

        context_examples = [self.data.process_sent(
            self.data.tfidf_compressor.compress(c)) for c in self.train_c[filtered_idx]]

        scores = self.model.get_scores(question_example, context_examples)
        c_max = scores.argsort()[::-1][:10]
        if len(c_max) == 0:
            return "There is no answer for that.", ["None"]

        top_related_questions = self.related_questions[filtered_idx][c_max]
        top_original_context = self.all_train_contexts[filtered_idx][c_max]

        # process top related questions
        related_question_examples = [self.data.process_sent(i[0]) for i in top_related_questions]

        q_closet = self._arg_closest_related_questions(question_example, related_question_examples)
        return top_original_context[q_closet], top_related_questions[q_closet]

    def _arg_closest_related_questions(self, question, related_questions):
        all_question = [question] + related_questions
        q_char_ids, q_word_ids = zip(*[zip(*zip(*x)) for x in all_question])

        padded_q_word_ids, q_sequence_lengths = pad_sequences(q_word_ids, pad_tok=0)
        padded_q_char_ids, q_word_lengths = pad_sequences(q_char_ids, pad_tok=0, nlevels=2)

        feed_dict = {self.model.q_word_ids: padded_q_word_ids,
                     self.model.q_char_ids: padded_q_char_ids,
                     self.model.q_sequence_lengths: q_sequence_lengths,
                     self.model.q_word_lengths: q_word_lengths,
                     self.model.keep_op: 1.0,
                     self.model.is_training: False}
        question_embeddings = self.model.sess.run(self.model.q_dense, feed_dict=feed_dict)
        q = question_embeddings[0]  # 1, 300
        rq = question_embeddings[1:]
        scores = np.sum(np.square(rq - q), axis=-1)

        q_min = scores.argsort()[0]
        return q_min


def main(model_name, dataset):
    inference = Inference(model_name, dataset)

    while True:
        q = input("\nQuestion: ")
        if q == "x":
            break
        a, qs = inference.get_answer(q)
        print(a)
        print("\nRelated question:")
        for i in qs:
            print("- {}".format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the NTU FAQs Chatbot with a pre-trained model and FAQs dataset')
    parser.add_argument('model', help="the name of the model")
    parser.add_argument('dataset', help="the name of the FAQs dataset (must be placed in data/ folder), e.g: original")

    args = parser.parse_args()

    main(args.model, args.dataset)
