import numpy as np
import logging
import logging
import os
from gensim.models.word2vec import Text8Corpus, Word2Vec

import matplotlib.pyplot as plt
from sklearn.manifold import t_sne
from MulticoreTSNE import MulticoreTSNE as TSNE

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

SIZE = 200
SAVEPATH = "text8.model"
FILEPATH = os.path.abspath("text8")
SAVE_MODELS = os.path.abspath("gensim_models")

if os.path.exists(SAVE_MODELS) is False:
    os.makedirs(SAVE_MODELS)


def text8model(filepath, save_models, savepath):
    sentences = Text8Corpus(filepath)
    model_path = os.path.abspath(save_models+"/"+savepath)
    if not os.path.exists(model_path):
        print("CREATING MODEL\n")
        model = Word2Vec(sentences, size=SIZE, workers=8)
        model.save(model_path)
    else:
        print("LOADING MODEL\n")
        model = Word2Vec.load(model_path)

    #test
    print(model.most_similar(positive=['woman', 'king'], negative=['man']))
    print(model.most_similar(['man']))

    return model, sentences


def max_length(sentences):
    max_len = 0
    for sent in sentences:
        if max_len < len(sent):
            max_len = len(sent)

    return max_len


def padding(sentences, max_length,pad="pad"):
    overall = []
    for sent in sentences:
        temp = []
        temp = sent
        if len(sent) < max_length:
            for _ in range(max_length-len(sent)):
                temp.append(pad)
        overall.append(temp)

    return overall


def to_vect(sentences):
    data_all = []
    for sent in sentences:
        temp = []
        for word in sent:
            if word == 'pad':
                temp.append(list(np.zeros(size), dtype=np.float32))
            else:
                try:
                    temp.append(list(model[word]))
                except KeyError:
                    temp.append(list(np.ones(size)))
        data_all.append(temp)

    return np.array(data_all)


def simple_tsne(model, is_multi=False):

    assert isinstance(model, Word2Vec), "Error, IT IS NOT A WORD2VEC MODEL"
    X = model[model.wv.vocab].astype(np.float64)

    if is_multi:
        tsne = TSNE(n_components=2, n_jobs=8, verbose=3)
    else:
        tsne = t_sne.TSNE(n_components=2, verbose=3)
    X_tsne = tsne.fit_transform(X)

    plt.scatter(X_tsne[:,0], X_tsne[:,1])
    plt.show()