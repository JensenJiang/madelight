import numpy as np


class WordDict:
    def __init__(self, exclude=[], special=['_PAD', '_GO', '_EOS'],  max_size=20000):
        self._exclude = exclude
        self.max_size = max_size
        self._dict = {}
        self._rev = []
        for w in special:
            self.add_word(w)

    def add_word(self, w):
        if self._dict.get(w) is None:
            _len = len(self._dict)
            self._dict[w] = dict(wid=_len)
            self._rev.append(dict(word=w))
        return self._dict[w]['wid']

    def word2id(self, w):
        temp = self._dict.get(w)
        if temp is None:
            return None
        else:
            return temp['wid']

    def id2word(self, id):
        return self._rev[id]['word']


class WordVecDict(WordDict):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def add_word_vec(self, w, vec):
        wid = self.add_word(w)
        self._dict[w]['vec'] = vec
        self._rev[wid]['vec'] = vec
        return wid

    def get_vecs_mat(self):
        return np.array([i['vec'] for i in self._rev], dtype=np.float32)

    @classmethod
    def from_vec_file(cls, fin, *args, **kargs):
        _, vec_size = fin.readline().split()
        vec_size = int(vec_size)
        wd = cls(*args, **kargs)
        wd.add_word_vec('_PAD', np.zeros(shape=(vec_size,)))
        for line in fin:
            if len(wd._dict) >= wd.max_size:
                break
            split = line.split()
            w, vec = split[0], np.array(list(map(float, split[1:])), dtype=np.float32)
            if w in wd._exclude:
                continue
            wd.add_word_vec(w, vec)
        return wd
