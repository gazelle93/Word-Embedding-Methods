import stanza
import spacy
import nltk
import gensim

def word_tokenization(_input_text, _nlp_pipeline=None, _lower=False):
    if _lower == True:
        _input_text = _input_text.lower()
    
    input_tk_list = []
    
    if _nlp_pipeline == None:
        return _input_text.split()
    
    elif _nlp_pipeline == "stanza":
        nlp = stanza.Pipeline('en')
        text = nlp(_input_text)

        for sen in text.sentences:
            for tk in sen.tokens:
                tk_infor_dict = tk.to_dict()[0]
                cur_tk = tk_infor_dict["text"]
                input_tk_list.append(cur_tk)
        return input_tk_list

    elif _nlp_pipeline == "spacy":
        nlp = spacy.load("en_core_web_sm")
        text = nlp(_input_text)

        for tk_idx, tk in enumerate(text):
            cur_tk = tk.text
            input_tk_list.append(cur_tk)
            
        return input_tk_list

    elif _nlp_pipeline == "nltk":
        return nltk.tokenize.word_tokenize(_input_text)
    
    elif _nlp_pipeline == "gensim":
        return list(gensim.utils.tokenize(_input_text))
