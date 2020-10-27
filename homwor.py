from collections import Counter


class CountVectorizer:
    '''
    Count vectorizer class
    '''
    def __init__(self):
        self.vocab = set()

    def fit_transform(self, texts: list) -> list:
        '''
        :param texts: list of strings
        :return: list of list
        '''
        for i in range(len(texts)):
            texts[i] = texts[i].lower()
            self.vocab = self.vocab.union(set(texts[i].split(' ')))
        vecs = []
        for text in texts:
            vec_map = {token: 0 for token in self.vocab}
            counter = dict(Counter(text.split(' ')))
            vec_map.update(counter)
            vecs.append(list(vec_map.values()))
        return vecs

    def get_names(self) -> list:
        '''
        :return: list of strings
        '''
        return list(self.vocab)


if __name__ == '__main__':
    cv = CountVectorizer()
    corpus = ['Crock Pot Pasta Never boil pasta again',
              'Pasta Pomodoro Fresh ingredients Parmesan to taste']
    count_matrix = cv.fit_transform(corpus)
    print(count_matrix)
    print(cv.get_names())
