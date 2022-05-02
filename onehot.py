import torch

def get_tokens(text, _pipeline):
    nlp_pipeline = "spacy"
    selected_nlp_pipeline = get_nlp_pipeline(_pipeline)

    return text_preprocessing(text, selected_nlp_pipeline, _pipeline)

def init_token2idx(text_list, _pipeline):
    selected_nlp_pipeline = get_nlp_pipeline(_pipeline)
    tk_lists = []
    for text in text_list:
        tk_lists.append(text_preprocessing(text, selected_nlp_pipeline, nlp_pipeline))
        
    whole_tokens = [x for tk_list in tk_lists for x in tk_list]
    set_of_tokens = list(set(whole_tokens)) + ["UNK"]
    
    set_of_tokens.sort()
    
    token2idx_dict = {}
    idx2token_dict = {}
    for idx, t in enumerate(set_of_tokens):
        token2idx_dict[t] = idx
        idx2token_dict[idx] = t
        
    return token2idx_dict, idx2token_dict

def tk2idx(_text, _pipeline, token2idx_dict, unk_ignore):
    selected_nlp_pipeline = get_nlp_pipeline(_pipeline)
    tk_lists = text_preprocessing(_text, selected_nlp_pipeline, _pipeline)
        
    
    idx_list = []
    if unk_ignore == True:
        for tk in tk_lists:
            if tk in token2idx_dict:
                idx_list.append(token2idx_dict[tk])
            else:
                idx_list.append(token2idx_dict["UNK"])
    else:
        for tk in tk_lists:
            idx_list.append(token2idx_dict[tk])
    return idx_list

def custom_one_hot_encoding(_idx_list, dim):
    tensor_list = []
    
    for idx in _idx_list:
        temp = torch.zeros(dim)
        temp[idx] = 1
        tensor_list.append(temp)
        
    return tensor_list

def tensor2token(_tensor):
    idx = (_tensor == 1).nonzero(as_tuple=True)[0].item()
    return idx2token_dict[idx]

  
def onehot_encoding(_encoder, _tks):
    tk_list = [[x] for x in _tks[0]]
    return [torch.tensor(x) for x in _encoder.transform(tk_list).toarray()]
