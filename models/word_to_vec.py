from gensim.models import Word2Vec


def word_vec(tokens):
    print(tokens)
    model = Word2Vec(tokens, sg=0, vector_size=300, window=5, min_count=1, epochs=7, negative=10)
    model.save('./saved_models/word2vec_model')

    return model