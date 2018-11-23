import argparse
import tensorflow as tf
from sklearn.utils import resample

from models.pairwise_model import FixedAttendNeuralScoreRanker, NoAttendNeuralScoreRanker, RelationRanker
from utils import Timer
from dataset import Dataset, get_trimmed_embeddings
from constants import DATA


class NtuModel(RelationRanker):
    def __init__(self, model_name, embeddings, num_chars, batch_size=64, early_stopping=False, k_neg=50):
        super().__init__(model_name, embeddings, num_chars, batch_size, early_stopping, k_neg)

    def _add_model_op(self):
        super()._add_model_op()

    def _dev_acc(self, top_k=1, num_ques_sample=3):
        count_true = 0
        count_total = 0
        for i in self.dev_examples:
            if num_ques_sample >= len(i[0]):
                dev_question_examples = i[0]
            else:
                dev_question_examples = resample(i[0], replace=False, n_samples=num_ques_sample)

            dev_context_examples = i[2][:]  # negative contexts
            dev_context_examples.append(i[1])  # positive context
            for q in dev_question_examples:
                count_total += 1
                scores = self.get_scores(q, dev_context_examples)
                c_max = scores.argsort()[::-1][:top_k]
                if (len(scores) - 1) in c_max:
                    count_true += 1
        return count_true / count_total

    def evaluate(self, examples, top_k=(1, 5, 10)):
        timer = Timer()
        timer.start("Evaluating on a given dataset")

        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(self.model_name))

        count_true = [0] * len(top_k)
        count_total = 0

        for i in examples:
            context_examples = i[2][:]
            context_examples.append(i[1])
            for q in i[0]:
                count_total += 1
                scores = self.get_scores(q, context_examples)
                for j in range(len(top_k)):
                    c_max = scores.argsort()[::-1][:top_k[j]]
                    if (len(scores) - 1) in c_max:
                        count_true[j] += 1

        for i in range(len(top_k)):
            print("Top {}:".format(top_k[i]))
            print("- Accuracy: {}".format(count_true[i] / count_total))
            print("- Total: {} - Correct: {}".format(count_total, count_true[i]))
        self.sess.close()
        timer.stop()


def main(model_name, dataset, train_set, is_evaluate, early_stopping, epoch, k_neg, is_continue):
    train = Dataset(dataset, num_ques_sample=5)

    train.load_data_pairwise(DATA + "{}/{}".format(dataset, train_set))
    num_chars = train.get_num_chars()

    embeddings = get_trimmed_embeddings(DATA + "embedding_data.npz")

    model = NtuModel(model_name=model_name, embeddings=embeddings, num_chars=num_chars,
                     batch_size=32, early_stopping=early_stopping, k_neg=k_neg)
    model.load_data(train)
    model.build()

    if is_evaluate:
        model.evaluate(train.dev_examples)
    elif early_stopping:
        model.run_train(epoch, is_continue, patience=5)
    else:
        model.run_train(epoch, is_continue)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a new model or evaluating a pre-trained one.')
    parser.add_argument('model', help="the name of the model")
    parser.add_argument('dataset', help="the name of the dataset that the model is trained on, e.g: original")
    parser.add_argument('train_set', help="the name of the training data file,"
                                          "e.g: train_dev_pairwise_compressed.pickle")

    # optional
    parser.add_argument("-eval", "--evaluate", help="evaluate existing model", action="store_true")
    parser.add_argument("-es", "--early_stopping", help="use early stopping", action="store_true")
    parser.add_argument("-e", "--epoch", help="number of epochs to train or maximum epoch when early stopping",
                        type=int, default=10)
    parser.add_argument("-k", "--k_neg", help="number of negative examples to be sampled", type=int, default=50)
    parser.add_argument("-c", "--is_continue", help="continue to train or not", action="store_true")
    args = parser.parse_args()

    main(args.model, args.dataset, args.train_set, args.evaluate,
         args.early_stopping, args.epoch, args.k_neg, args.is_continue)
