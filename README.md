# Overview
- The most general usage of neural networks in NLP is to predict the label that satisfies the purpose of the task from word embeddings. To let the machine understand the human language, the tokenized text has to be represented in the numeric space and this can be done by transforming words into matrices which are called word embeddings. Word embedding is a representation of a word or group of words that are converted into a group of numbers in form of vectors or matrices. The most basic manner of word representation is One-hot encoding where the word is represented as 0 and 1. Word2Vec is the fundamental method of the distributed representation that vectorizes the word representation in multi-dimensional space. Word2Vec proposed two different methods: Continuous Bag of Words (CBoW) and Skip-Gram. CBoW is a method of predicting the target word from words around the target word called surrounding words. ELMo (Embeddings from Language Models) is a character-based word representation that contains bidirectionality. Unlike the traditional word embeddings like Word2Vec and Glove, ELMo solves the Out-Of-Vocabulary issue by applying the character-based technique. BERT is a subword-based representation with 12 attention layers that leverage attention scores to compute the importance of the tokens by comparing them with other tokens in a given context simultaneously. This project aims to implement differnent encoding methods.

# Brief description
- text_processing.py
> Output format
> - output: Tokenized result of a given text. (list)
- my_onehot.py; my_skipgram.py; my_cbow.py; my_elmo.py; my_bert.py
> Output format
> - output: List of tensor of input tokens.

# Prerequisites
- argparse
- stanza
- spacy
- nltk
- gensim
- torch
- allennlp
- transformers

# Parameters
- nlp_pipeline(str, defaults to "spacy"): Tokenization method (spacy, stanza, nltk, gensim).
- encoding(str, defaults to "bert"): Encoding method (onehot, skipgram, cbow, elmo, bert).
- custom(bool, defaults to "True"): Using custom method for word embeddings or not (onthot, skipgram, cbow).
- emb_dim(int, defaults to 10): The size of word embedding.
- hidden_dim(int, defaults to 128): The hidden size of skipgram and cbow.
- unk_ignore(bool, defaults to True): Ignore unseen word or not.
- window(int, defaults to 2): The size of window of skipgram and cbow.

# References
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- NLTK: Bird, Steven, Edward Loper and Ewan Klein (2009). Natural Language Processing with Python.  O'Reilly Media Inc.
- Gensim: Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. In In Proceedings of the LREC 2010 workshop on new challenges for NLP frameworks.
- Word2vec: Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.
- ELMo: Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. arXiv preprint arXiv:1802.05365.
- BERT: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
