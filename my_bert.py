from transformers import BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast
pretrained_model = "bert-base-uncased"

def get_model():
    myconfig = BertConfig.from_pretrained(
                    pretrained_model
                )

    model = BertModel.from_pretrained(pretrained_model, config=myconfig)

    return model

def get_word_embeddings(model, cur_text):

    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
    bert_input = tokenizer(cur_text, return_tensors='pt')
    sequence_output, pooled_output = model(**bert_input)

    return sequence_output[0]
