import torch
import torch.nn as nn
from gensim.models import Word2Vec
from text_processing import get_nlp_pipeline, word_tokenization


# Gensim Word2vec CBOW
def get_tokenized_text(text_list, nlp_pipeline):
    selected_nlp_pipeline = get_nlp_pipeline(nlp_pipeline)

    input_tokens = []
    for _text in text_list:
        input_tokens.append(word_tokenization(_text, selected_nlp_pipeline, nlp_pipeline))
        
    return input_tokens

def train_gensim_cbow_model(entire_text, dim, window):
    skipgram_word2vec_model = Word2Vec(
        sentences=entire_text,
        vector_size=dim,
        alpha=0.025,
        window=window,
        min_count=1,
        sg=0,
        workers=4)
    
    skipgram_word2vec_model.build_vocab(entire_text, progress_per=10)
    skipgram_word2vec_model.train(entire_text,
                              total_examples=skipgram_word2vec_model.corpus_count, epochs=10)
    
    return skipgram_word2vec_model
  
  
def get_word_embeddings(model, cur_text, selected_nlp_pipeline, nlp_pipeline):
    tks = word_tokenization(cur_text, selected_nlp_pipeline, nlp_pipeline)

    return [torch.tensor(model.wv[x]) for x in tks]
