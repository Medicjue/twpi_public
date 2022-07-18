


class Judger:
    __arr_size = 64

    def __init__(self, model_path, words_index_path):
        import pickle
        import tensorflow as tf
        from booking_review_data_preprocess import ZhTwTokenizer
        self.mdl = tf.keras.models.load_model(model_path)
        with open(words_index_path, 'rb') as f:
            self.words_index = pickle.load(f)
        self.tokenizer = ZhTwTokenizer()

    def judge(self, sentences:list):
        import numpy as np
        tokenized_sentences = self.tokenizer.tokenize(sentences)
        sentence_np_list = []
        non_exist_id = len(self.words_index) + 1
        for tokenized_sentence in tokenized_sentences:
            sentence_idx = []
            for word in tokenized_sentence:
                idx = self.words_index.get(word, non_exist_id)
                if idx >= 50000:
                    idx = 0
                sentence_idx.append(idx)
            sentence_idx_arr = np.asarray(sentence_idx)
            sentence_idx_arr.resize(self.__arr_size)
            sentence_idx_arr = np.flip(sentence_idx_arr)
            sentence_np_list.append(sentence_idx_arr)
        sentence_np_list = np.asarray(sentence_np_list)
        return self.mdl.predict(sentence_np_list.reshape(-1, self.__arr_size))

if __name__ == '__main__':
    judger = Judger(model_path='bookingReview/bookingReviewMdl/', words_index_path='bookingReview/words_index.pickle')
    sentiment_scores = judger.judge(['床很舒適，採光很好', '馬桶不通，房間有霉味'])
    print(sentiment_scores)
    sentiment_scores = judger.judge(['政府亂搞防疫政策，蔡英文下台', '國民黨只有有分數就及格\n民進黨只要不是100分就死當\n這三十年來一直如此'])
    print(sentiment_scores)
    print(type(sentiment_scores))
