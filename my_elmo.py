from text_processing import get_nlp_pipeline, word_tokenization
from allennlp.modules.elmo import Elmo, batch_to_ids

def get_model():

    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    elmo = Elmo(options_file=options_file,
                weight_file=weight_file,
                num_output_representations=2,
                dropout=0)

    return elmo

def get_word_embeddings(model, cur_text, selected_nlp_pipeline, nlp_pipeline):
    tokens = word_tokenization(cur_text, selected_nlp_pipeline, nlp_pipeline)
    elmo_input = batch_to_ids([tokens])
    return model(elmo_input)['elmo_representations'][0]
