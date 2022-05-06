import argparse
from text_processing import get_nlp_pipeline, word_tokenization
import my_onehot, my_skipgram, my_cbow, my_elmo, my_bert

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
        if args.custom == True:
            token2idx_dict, idx2token_dict = my_onehot.init_token2idx(text_list, args.nlp_pipeline)
            dim = len(idx2token_dict)
            cur_tk2idx = my_onehot.tk2idx(cur_text, args.nlp_pipeline, token2idx_dict, unk_ignore=args.unk_ignore)
            
            embeddings = my_onehot.custom_one_hot_encoding(cur_tk2idx, dim)
            print("Customized One-hot Encoding Result")
            print(embeddings)
            
        else:
            sklearn_onehotencoder = my_onehot.build_onehot_encoding_model(args.unk_ignore)
            token2idx_dict, _ = my_onehot.init_token2idx(text_list, args.nlp_pipeline)
            sklearn_onehotencoder.fit([[t] for t in token2idx_dict])
            tks = my_onehot.get_tokens(cur_text, args.nlp_pipeline)
            
            embeddings = my_onehot.onehot_encoding(sklearn_onehotencoder, tks)
            print("Sklearn One-hot Encoding Result")
            print(embeddings)
            

    # Skip-gram Embedding
    if args.encoding == "skipgram":
        if args.custom == True:
            vocab, input_tokens = my_skipgram.build_vocab(text_list, selected_nlp_pipeline, args.nlp_pipeline)
            word_to_ix = my_skipgram.get_word2idx(set(vocab))
            skipgram_model = my_skipgram.custom_skipgram(len(set(vocab))+1, args.hidden_dim, args.emb_dim, args.window)
            trained_model = my_skipgram.train_custom_skipgram_model(skipgram_model, input_tokens, args.window, word_to_ix)
            embeddings = my_skipgram.get_custom_word_embeddings(trained_model, cur_text, selected_nlp_pipeline, args.nlp_pipeline, word_to_ix)
            print("Customized Skip-gram Embedding Result")
            print(embeddings)

        else:
            training_input_of_model = my_skipgram.get_tokenized_text(text_list, args.nlp_pipeline)
            skipgram_model = my_skipgram.train_gensim_skipgram_model(training_input_of_model, args.emb_dim, args.window)
            embeddings = my_skipgram.get_word_embeddings(skipgram_model, cur_text, selected_nlp_pipeline, args.nlp_pipeline)
            print("Skip-gram Embedding Result")
            print(embeddings)

    # CBOW Embedding
    if args.encoding == "cbow":
        if args.custom == True:
            vocab, input_tokens = my_cbow.build_vocab(text_list, selected_nlp_pipeline, args.nlp_pipeline)
            word_to_ix = my_cbow.get_word2idx(set(vocab))
            cbow_model = my_cbow.custom_cbow(len(set(vocab))+1, args.hidden_dim, args.emb_dim, args.window)
            trained_model = my_cbow.train_custom_skipgram_model(cbow_model, input_tokens, args.window, word_to_ix)
            embeddings = my_cbow.get_custom_word_embeddings(trained_model, cur_text, selected_nlp_pipeline, args.nlp_pipeline, word_to_ix)
            print("Customized CBOW Embedding Result")
            print(embeddings)

        else:
            training_input_of_model = my_cbow.get_tokenized_text(text_list, args.nlp_pipeline)
            cbow_model = my_cbow.train_gensim_cbow_model(training_input_of_model, args.emb_dim, args.window)
            embeddings = my_cbow.get_word_embeddings(cbow_model, cur_text, selected_nlp_pipeline, args.nlp_pipeline)
            print("CBOW Embedding Result")
            print(embeddings)

    # ELMo Embedding
    if args.encoding == "elmo":
        elmo_model = my_elmo.get_model()
        embeddings = my_elmo.get_word_embeddings(elmo_model, cur_text, selected_nlp_pipeline, args.nlp_pipeline)

        print("ELMo Embedding Result")
        print(embeddings)

    # BERT Embedding
    if args.encoding == "bert":
        bert_model = my_bert.get_model()
        embeddings = my_bert.get_word_embeddings(bert_model, cur_text)

        print("BERT Embedding Result")
        print(embeddings)

          
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--nlp_pipeline", default="spacy", type=str, help="NLP preprocessing pipeline.")
    parser.add_argument("--encoding", default="bert", type=str, help="The selection of encoding method.")
    parser.add_argument("--custom", default=True, help="Utilizing custom encoding.")
    parser.add_argument("--emb_dim", default=10, type=int, help="The size of word embedding.")
    parser.add_argument("--hidden_dim", default=128, type=int, help="The size of hidden dimension.")
    parser.add_argument("--unk_ignore", default=True, help="Ignore unknown tokens.")
    parser.add_argument("--window", default=2, type=int, help="The selected window size.")
    args = parser.parse_args()

    main(args)
