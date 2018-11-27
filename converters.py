import json
import argparse
import re
import sys
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from pre_process import PreProcess
from constants import DATA, CONVERTED_DATA
from utils import Timer


class FaqConverter:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_file = "questions_and_answers_ntu.json"
        self._get_train_dev()

    def convert_data(self):
        train_data = self._convert_data(self.train_data)
        dev_data = self._convert_data(self.dev_data)
        with open(DATA + self.dataset + "/" + CONVERTED_DATA, 'wb') as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(dev_data, f, pickle.HIGHEST_PROTOCOL)

    def _get_train_dev(self):
        with open(DATA + self.data_file, encoding='utf-8') as f:
            text = f.read()
        obj = json.loads(text, encoding='utf-8')

        data = [obj[i] for i in obj]
        data = shuffle(data)

        self.train_data, self.dev_data = train_test_split(data, test_size=0.1, random_state=42)

    @staticmethod
    def _convert_data(data_obj):
        pre_process = PreProcess()

        data = {}
        idx = 0
        for d in data_obj:
            # custom pre-process
            d['answer'] = d['answer'].strip("Answer:")
            d['answer'] = re.sub("&nbsp;", " ", d['answer'])

            context = " ".join(pre_process.process(d['answer'], url_norm=True))
            question = " ".join(pre_process.process(d['question'], remove_stop_words=False))
            if not (d['answer'] and context and question):
                continue
            data[idx] = {
                'context': d['answer'],
                'c': context,
                'qs': [question]
            }
            idx += 1
        return data


class FaqExtendedConverter:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_file = "questions_and_answers_ntu.json"
        self.extend_question_dict = self._get_extended_questions()
        self.data = self._get_data()

    def convert_data(self):
        train_data, dev_data = self._convert_data(self.data)

        with open(DATA + self.dataset + "/" + CONVERTED_DATA, 'wb') as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(dev_data, f, pickle.HIGHEST_PROTOCOL)

    def _get_data(self):
        with open(DATA + self.data_file, encoding='utf-8') as f:
            text = f.read()
        obj = json.loads(text, encoding='utf-8')

        data = [obj[i] for i in obj]
        data = shuffle(data)
        return data

    def _get_extended_questions(self):
        with open(DATA + self.dataset + "/extra_questions.txt", 'r', encoding='utf8') as f:
            raw = f.read().strip()

        question_frames = raw.split(
            "====================================================================================================")
        question_frames = [qf.strip() for qf in question_frames[:-1]]

        def process(question_frame):
            # return original question and its permutations
            lines = question_frame.split('\n')
            lines = [l.strip() for l in lines]
            if lines[0][:2] == "No":
                return None

            original = lines[0].strip("Permutations of '")[:-2]
            permutations = [l for l in lines[1:] if l]
            return original, permutations

        pre_process = PreProcess()

        question_dict = {}
        for qf in question_frames:
            tmp = process(qf)
            if tmp:
                o, p = process(qf)
                k = " ".join(pre_process.process(o, remove_stop_words=False))
                question_dict[k] = [" ".join(pre_process.process(i, remove_stop_words=False)) for i in p]

        return question_dict

    def _convert_data(self, data_obj):
        pre_process = PreProcess()

        train_data = {}
        dev_data = {}
        idx = 0
        for d in data_obj:
            # custom pre-process
            d['answer'] = d['answer'].strip("Answer:")

            context = " ".join(pre_process.process(d['answer'], url_norm=True))
            if not context:
                continue

            original_question = " ".join(pre_process.process(d['question'], remove_stop_words=False))
            extended_questions = self.extend_question_dict.get(original_question, [])

            if extended_questions:
                # split train and dev by questions
                train_questions, dev_questions = train_test_split(extended_questions, test_size=0.1, random_state=42)

                train_data[idx] = {
                    'context': d['answer'],
                    'c': context,
                    'qs': [original_question] + train_questions
                }
                dev_data[idx] = {
                    'context': d['answer'],
                    'c': context,
                    'qs': dev_questions
                }
            else:
                train_data[idx] = {
                    'context': d['answer'],
                    'c': context,
                    'qs': [original_question]
                }
            idx += 1
        return train_data, dev_data


class TfIdfVectorizer:
    def __init__(self):
        self.tf_idf = TfidfVectorizer(norm='l2',
                                      smooth_idf=False,
                                      tokenizer=lambda x: x.split(" "),
                                      analyzer='word',
                                      ngram_range=(1, 1))

    def train(self, ques):
        self.tf_idf.fit_transform(ques)

    def distance(self, q1, ques):
        tf_idf_questions = self.tf_idf.transform([q1] + ques)
        cosine_similarities = linear_kernel(tf_idf_questions[0], tf_idf_questions[1:])[0]

        return cosine_similarities.argsort()


class FaqExtendedBigConverter:
    def __init__(self, dataset, top_k=100):
        self.dataset = dataset
        self.top_k = top_k
        self.tf_idf = TfIdfVectorizer()
        self.data_file = "questions_and_answers_ntu.json"
        self.extend_question_dict = self._get_extended_questions()
        self.data = self._get_data()

    def convert_data(self):
        train_data, dev_data = self._convert_data(self.data)

        with open(DATA + self.dataset + "/" + CONVERTED_DATA, 'wb') as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(dev_data, f, pickle.HIGHEST_PROTOCOL)

    def _get_data(self):
        with open(DATA + self.data_file, encoding='utf-8') as f:
            text = f.read()
        obj = json.loads(text, encoding='utf-8')

        data = [obj[i] for i in obj]
        data = shuffle(data)
        return data

    def _get_extended_questions(self):
        with open(DATA + self.dataset + "/extra_questions.txt", 'r', encoding='utf8') as f:
            raw = f.read().strip()

        question_frames = raw.split(
            "====================================================================================================")
        question_frames = [qf.strip() for qf in question_frames[:-1]]

        def process(question_frame):
            # return original question and its permutations
            lines = question_frame.split('\n')
            lines = [l.strip() for l in lines]
            if lines[0][:2] == "No":
                return None

            original = lines[0].strip("Permutations of '")[:-2]
            permutations = [l for l in lines[1:] if l]
            return original, permutations

        pre_process = PreProcess()

        question_dict = {}
        t = Timer()
        for qf in question_frames:
            tmp = process(qf)
            if tmp:
                t.start("", verbal=False)
                o, p = process(qf)
                k = " ".join(pre_process.process(o, remove_stop_words=False))
                question_dict[k] = [" ".join(pre_process.process(i, remove_stop_words=False)) for i in p]

                # select the most diverse question set
                self.tf_idf.train([k] + question_dict[k])
                del_num = len(question_dict[k]) // self.top_k
                if del_num == 0:
                    t.remaining_time(t.stop(verbal=False), len(question_frames))
                    continue

                selected = []
                while question_dict[k]:
                    indices = self.tf_idf.distance(k, question_dict[k])
                    q = question_dict[k].pop(indices[0])
                    selected.append(q)
                    if not question_dict[k]:
                        break
                    close_q = self.tf_idf.distance(q, question_dict[k])[::-1][:del_num]
                    question_dict[k] = [question_dict[k][i] for i in range(len(question_dict[k])) if i not in close_q]
                question_dict[k] = selected
                t.remaining_time(t.stop(verbal=False), len(question_frames))

        return question_dict

    def _convert_data(self, data_obj):
        pre_process = PreProcess()

        train_data = {}
        dev_data = {}
        idx = 0
        for d in data_obj:
            # custom pre-process
            d['answer'] = d['answer'].strip("Answer:")

            context = " ".join(pre_process.process(d['answer'], url_norm=True))
            if not context:
                continue

            original_question = " ".join(pre_process.process(d['question'], remove_stop_words=False))
            extended_questions = self.extend_question_dict.get(original_question, [])

            if extended_questions:
                # split train and dev by questions
                train_questions, dev_questions = train_test_split(extended_questions, test_size=0.1, random_state=42)

                train_data[idx] = {
                    'context': d['answer'],
                    'c': context,
                    'qs': [original_question] + train_questions
                }
                dev_data[idx] = {
                    'context': d['answer'],
                    'c': context,
                    'qs': dev_questions
                }
            else:
                train_data[idx] = {
                    'context': d['answer'],
                    'c': context,
                    'qs': [original_question]
                }
            idx += 1
        return train_data, dev_data


def main(converter, dataset):
    c = getattr(sys.modules[__name__], converter)(dataset)
    t = Timer()
    t.start("Converting data", verbal=True)

    c.convert_data()
    t.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert data from a particular format into uniform input.')
    parser.add_argument('converter', help="the name of the converter to be used, e.g: FaqExtendedConverter")
    parser.add_argument('dataset', help="the name of the raw dataset to be converted, e.g: extend")

    args = parser.parse_args()

    main(args.converter, args.dataset)
