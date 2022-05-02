import argparse

from text_processing import get_nlp_pipeline, word_tokenization
import my_onehot
import my_skipgram

def main(args):
    text_list = ["We are about to study the idea of a computational process.", 
             "Computational processes are abstract beings that inhabit computers.",
            "As they evolve, processes manipulate other abstract things called data.",
            "The evolution of a process is directed by a pattern of rules called a program.",
            "People create programs to direct processes.",
            "In effect, we conjure the spirits of the computer with our spells."]
    
    cur_text = "People create a computational process."
    
    if args.encoding != "bert":
        selected_nlp_pipeline = get_nlp_pipeline(args.nlp_pipeline)
    
    # One-hot Encoding
    if args.encoding == "onehot":
        if args.custom_enncoding == True:
            token2idx_dict, idx2token_dict = my_onehot.init_token2idx(text_list, args.nlp_pipeline)
            dim = len(idx2token_dict)
            cur_tk2idx = my_onehot.tk2idx(cur_text, _pipeline, token2idx_dict, unk_ignore=args.unk_ignore)
            
            embeddings = my_onehot.custom_one_hot_encoding(cur_tk2idx, dim)
            print("Customized One-hot Encoding Result")
            print(embeddings)
            
        else:
            if args.unk_ignore == True:
                sklearn_onehotencoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
            else:
                sklearn_onehotencoder = preprocessing.OneHotEncoder()
                
            token2idx_dict, _ = my_onehot.init_token2idx(text_list, arg.nlp_pipeline)
            sklearn_onehotencoder.fit([[t] for t in token2idx_dict])
            tks = my_onehot.get_tokens(cur_text, args.nlp_pipeline)
            
            embeddings = my_onehot.onehot_encoding(sklearn_onehotencoder, tks)
            print("Sklearn One-hot Encoding Result")
            print(embeddings)
            
            
    if args.encoding == "skipgram":
        if args.custom_enncoding == True:
            print("1")
        else:
            training_input_of_model = my_skipgram.get_tokenized_text(text_list, args.nlp_pipeline)
            skipgram_model = my_skipgram.train_skipgram_model(training_input_of_model, args.dim, arg.swindow)
            embeddings = my_skipgram.get_word_embeddings(skipgram_model, cur_text, selected_nlp_pipeline, args.nlp_pipeline)
            print("Skip-gram Embedding Result")
            print(embeddings)
            
          
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--encoding", default="onehot", type=str, help="The selection of encoding method.")
    parser.add_argument("--custom_enncoding", default=False, type=bool, help="Utilizing custom encoding.")
    parser.add_argument("--dim", default=100, type=int, help="The size of word embedding.")
    parser.add_argument("--unk_ignore", default=True, type=bool, help="Ignore unknown tokens.")
    parser.add_argument("--window", default=2, type=int, help="The selected window size.")
    
    #parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    #parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    #parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    
    args = parser.parse_args()

    main(args)
