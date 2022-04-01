import gensim.downloader as api

w2v = api.load('word2vec-google-news-300')
for w, p in w2v.most_similar('neural'):
    print(f"{w} -> {p}")

w2v.word_vec('play')[:20]
w2v.most_similar(positive=['king', 'woman'], negative=['man'])[0]
