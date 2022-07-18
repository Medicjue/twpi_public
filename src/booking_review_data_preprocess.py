import glob
import pickle
import numpy as np

from sklearn.model_selection import train_test_split

def run():
    positive_review_files = glob.glob('./bookingReview/positiveReviews/*.txt')
    positive_reviews = []
    for positive_review_file in positive_review_files:
        with open(positive_review_file, 'r') as f:
            positive_review = ''.join(f.readlines())
            positive_reviews.append(positive_review)
    with open('./bookingReview/positiveReviews.pickle', 'wb') as f:
        pickle.dump(positive_reviews, f)

    negative_review_files = glob.glob('./bookingReview/negativeReviews/*.txt')
    negative_reviews = []
    for negative_review_file in negative_review_files:
        with open(negative_review_file, 'r') as f:
            negative_review = ''.join(f.readlines())
            negative_reviews.append(negative_review)
    with open('./bookingReview/negativeReviews.pickle', 'wb') as f:
        pickle.dump(negative_reviews, f)


    with open('./bookingReview/positiveReviews.pickle', 'rb') as f:
        positive_reviews = pickle.load(f)
    with open('./bookingReview/negativeReviews.pickle', 'rb') as f:
        negative_reviews = pickle.load(f)

    tokenizer = ZhTwTokenizer()

    words_set = set()

    positive_word_sentence_list = tokenizer.tokenize(positive_reviews)
    for positive_word_sentence in positive_word_sentence_list:
        for word in positive_word_sentence:
            words_set.add(word)
    print(len(words_set))

    negative_word_sentence_list = tokenizer.tokenize(negative_reviews)
    for negative_word_sentence in negative_word_sentence_list:
        for word in negative_word_sentence:
            words_set.add(word)
    print(len(words_set))

    words_map = {id+1:val for id, val in enumerate(words_set)}
    words_map[0] = None
    words_index = {v:k for k, v in words_map.items()}

    with open('./bookingReview/words_index.pickle', 'wb') as f:
        pickle.dump(words_index, f)
    with open('./bookingReview/words_map.pickle', 'wb') as f:
        pickle.dump(words_map, f)

    arr_size = 64

    positive_word_sentence_np_list = []
    non_exist_id = len(words_index) + 1
    for positive_word_sentence in positive_word_sentence_list:
        sentence_idx = []
        for word in positive_word_sentence:
            idx = words_index.get(word, non_exist_id)
            sentence_idx.append(idx)
        sentence_idx_arr = np.asarray(sentence_idx)
        sentence_idx_arr.resize(arr_size)
        sentence_idx_arr = np.flip(sentence_idx_arr)
        positive_word_sentence_np_list.append(sentence_idx_arr)

    with open('./bookingReview/positive_word_sentence_np_list.pickle', 'wb') as f:
        pickle.dump(positive_word_sentence_np_list, f)

    negative_word_sentence_np_list = []
    non_exist_id = len(words_index) + 1
    for negative_word_sentence in negative_word_sentence_list:
        sentence_idx = []
        for word in negative_word_sentence:
            idx = words_index.get(word, non_exist_id)
            sentence_idx.append(idx)
        sentence_idx_arr = np.asarray(sentence_idx)
        sentence_idx_arr.resize(arr_size)
        sentence_idx_arr = np.flip(sentence_idx_arr)
        negative_word_sentence_np_list.append(sentence_idx_arr)

    with open('./bookingReview/negative_word_sentence_np_list.pickle', 'wb') as f:
        pickle.dump(negative_word_sentence_np_list, f)

    positive_y = np.ones(len(positive_word_sentence_np_list))
    negative_y = np.zeros(len(negative_word_sentence_np_list))

    data_x = np.concatenate([positive_word_sentence_np_list, negative_word_sentence_np_list])
    data_y = np.concatenate([positive_y, negative_y])

    with open('./bookingReview/data_x.pickle', 'wb') as f:
        pickle.dump(data_x, f)
    with open('./bookingReview/data_y.pickle', 'wb') as f:
        pickle.dump(data_y, f)

    with open('./bookingReview/data_x.pickle', 'rb') as f:
        data_x = pickle.load(f)
    with open('./bookingReview/data_y.pickle', 'rb') as f:
        data_y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.15, random_state=42)

    with open('./bookingReview/X_train.pickle', 'wb') as f:
        pickle.dump(X_train, f)
    with open('./bookingReview/X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)
    with open('./bookingReview/y_train.pickle', 'wb') as f:
        pickle.dump(y_train, f)
    with open('./bookingReview/y_test.pickle', 'wb') as f:
        pickle.dump(y_test, f)
    

class ZhTwTokenizer:
    from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
    __ws = WS("./data")
    __pos = POS("./data")
    __ner = NER("./data")

    def tokenize(self, sentenses:list):
        word_sentence_list = self.__ws(sentenses)
        return word_sentence_list

    def pos(self, word_sentence_list:list):
        pos_sentence_list = self.__pos(word_sentence_list)
        return pos_sentence_list

        

if __name__ == '__main__':
    run()