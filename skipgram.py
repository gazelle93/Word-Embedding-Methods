import torch
import torch.nn as nn
from gensim.models import Word2Vec

def get_tokenized_text(text_list, nlp_pipeline):
    selected_nlp_pipeline = get_nlp_pipeline(nlp_pipeline)

    input_tokens = []
    for _text in text_list:
        input_tokens.append(text_preprocessing(_text, selected_nlp_pipeline, nlp_pipeline))
        
    return input_tokens

def train_skipgram_model(entire_text, dim, window):
    skipgram_word2vec_model = Word2Vec(
        sentences=_entire_tokenized_text,
        vector_size=dim,
        alpha=0.025,
        window=window,
        min_count=1,
        sg=1,
        workers=4)
    
    skipgram_word2vec_model.build_vocab(_entire_tokenized_text, progress_per=10)
    skipgram_word2vec_model.train(_entire_tokenized_text, 
                              total_examples=skipgram_word2vec_model.corpus_count, epochs=10)
    
    return skipgram_word2vec_model
  
  
def get_word_embeddings(model, cur_text, selected_nlp_pipeline, nlp_pipeline):
    tks = text_preprocessing(cur_text, selected_nlp_pipeline, nlp_pipeline)
    tensor_list = []
    for tk in tks:
        tensor_list.append(torch.from_numpy(model.wv[tk]))
    
    return tensor_list
