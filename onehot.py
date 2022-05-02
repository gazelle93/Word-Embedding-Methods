import torch

def get_tokens(_text, _pipeline):
    tk_lists = []
    for txt in _text:
        temp_tk_list, _ = text_preprocessing(txt, _pipeline)
        tk_lists.append(temp_tk_list)
    return tk_lists

def init_token2idx(_text, _pipeline):
    tk_lists = get_tokens(_text, _pipeline)
        
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
    tk_lists = get_tokens(_text, _pipeline)[0]
    
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

def custom_onehot_encoding(_idx_list, dim):
    tensor_list = []
    
    for idx in _idx_list:
        temp = torch.zeros(dim)
        temp[idx] = 1
        tensor_list.append(temp)
        
    return tensor_list

def tensor2token(_tensor):
    idx = (_tensor == 1).nonzero(as_tuple=True)[0].item()
    return idx2token_dict[idx]
  

  
def one_hot_encoding(_encoder, _tks):
    tk_list = [[x] for x in _tks[0]]
    return [torch.tensor(x) for x in _encoder.transform(tk_list).toarray()]
