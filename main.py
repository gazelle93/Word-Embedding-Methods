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
        sklearn_onehotencoder = my_onehot.build_onehot_encoding_model(args.unk_ignore)
        token2idx_dict, _ = my_onehot.init_token2idx(text_list, args.nlp_pipeline)
        sklearn_onehotencoder.fit([[t] for t in token2idx_dict])
        tks = my_onehot.get_tokens(cur_text, args.nlp_pipeline)

        embeddings = my_onehot.onehot_encoding(sklearn_onehotencoder, tks)
        print("Sklearn One-hot Encoding Result")
        print(embeddings)
            

    # Skip-gram Embedding
    if args.encoding == "skipgram":
        training_input_of_model = my_skipgram.get_tokenized_text(text_list, args.nlp_pipeline)
        skipgram_model = my_skipgram.train_gensim_skipgram_model(training_input_of_model, args.emb_dim, args.window)
        embeddings = my_skipgram.get_word_embeddings(skipgram_model, cur_text, selected_nlp_pipeline, args.nlp_pipeline)
        print("Skip-gram Embedding Result")
        print(embeddings)

    # CBOW Embedding
    if args.encoding == "cbow":
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
    parser.add_argument("--emb_dim", default=10, type=int, help="The size of word embedding.")
    parser.add_argument("--hidden_dim", default=128, type=int, help="The size of hidden dimension.")
    parser.add_argument("--unk_ignore", default=True, help="Ignore unknown tokens.")
    parser.add_argument("--window", default=2, type=int, help="The selected window size.")
    args = parser.parse_args()

    main(args)
