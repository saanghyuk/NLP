# Studying NLP

> Start Date : 2021/11/18

- **Materials**
  - [Coursera NLP](https://www.coursera.org/specializations/natural-language-processing?)
  - [HuggingFace](https://huggingface.co/)
  - FastCampus NLP

####	1. Review Neural Network and Pytorch
| date           | conents                                                      |                                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2021/11/18 Thu | MNIST with Pytorch                                           | [Code](https://github.com/saanghyuk/NLP/tree/main/DL_review/01_mnist) |
| 2021/11/19 Fri | Basic Mathematics, Auto Encoder, Manifold Hypothesis         | [Code1](https://github.com/saanghyuk/NLP/tree/main/DL_review/02_representation_learning), [Code2](https://github.com/saanghyuk/NLP/tree/main/DL_review/03-geometric_perspective) |
| 2021/11/20 Sat | Data Loader, Pytorch Ignite, CNN, VCG, ResNet, Transfer Learning | [Code1](https://github.com/saanghyuk/NLP/tree/main/DL_review/04-advanced_pytorch_tutorials/04-01-custom_dataset), [Code2](https://github.com/saanghyuk/NLP/tree/main/DL_review/04-advanced_pytorch_tutorials/04-02-pytorch_ignite), [Code3](https://github.com/saanghyuk/NLP/tree/main/DL_review/05-cnn), [Code4](https://github.com/saanghyuk/NLP/tree/main/DL_review/06-transfer_learning) |
| 2021/11/21 Sun | RNN, BBTT, LSTM, Gradient Clipping                           | [Code](https://github.com/saanghyuk/NLP/tree/main/DL_review/07-rnn) |



#### 2. NLP Basic Concepts

| date           | contents                                                     |                                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2021/11/22 Mon | Basic Concept, Regex                                         | [Code](https://github.com/saanghyuk/NLP/tree/main/NLP_basic/01-preprocessing/0-regex) |
| 2021/11/23 Tue | Corpus Labeling, Tokenization, Subword Segmentation(BPE), Detokenization, TorchText, WordSense, WordNet, TF-IDF | [Code1](https://github.com/saanghyuk/NLP/tree/main/NLP_basic/01-preprocessing/0-regex), [Code2](https://github.com/saanghyuk/NLP/tree/main/NLP_basic/02-tokenization), [Code3](https://github.com/saanghyuk/NLP/tree/main/NLP_basic/03-subword_segmentation), [Code4](https://github.com/saanghyuk/NLP/tree/main/NLP_basic/04-detokenization), [Code5](https://github.com/saanghyuk/NLP/tree/main/NLP_basic/05-torchtext) |
| 2021/11/24 Wed | Word2Vec(Skip Gram), GloVe, FastText                         | [Code](https://github.com/saanghyuk/NLP/tree/main/NLP_basic/08-word_embedding) |
| 2021/11/25 Thu | Text Classification with RNN                                 | [Code](https://github.com/saanghyuk/NLP/blob/main/NLP_basic/09-text_classification/simple_ntc/simple_ntc/models/rnn.py) |
| 2021/11/26 Fri | Text Classification with CNN                                 | [Code](https://github.com/saanghyuk/NLP/blob/main/NLP_basic/09-text_classification/simple_ntc/simple_ntc/models/cnn.py) |
| 2021/11/27 Sat | Text Classification with FastText                            | [Code](https://github.com/saanghyuk/NLP/tree/main/NLP_basic/09-text_classification/FastText_classification) |





#### 3. Natural Language Generation

| date           | contents                                                     |                                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2021/11/27 Sat | Language Modeling(N-gram), Smoothing and Discounting, Interpolation and Backoff, Perplexity, RNN Language Modeling, Teacher Forcing |                                                              |
| 2021/11/28 Sun | Data Preparation([AI Hub 'Kor-Eng Paired Data' for building machine translator](https://aihub.or.kr/aidata/87/download)) | [Code](https://github.com/saanghyuk/NLP/tree/main/NLG/4-nlg/nlp_preprocessing) |
| 2021/11/29 Mon | Sequence to Sequence, Attention, Masking, Input Feeding, [Pack Sequence](https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch), Gradient Accumulation | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/Packed_Sequence.ipynb),  [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/models/seq2seq.py) |
| 2021/11/30 Tue | [AMP](https://pytorch.org/docs/stable/amp.html), Seq2Seq Implementation | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/models/seq2seq.py) |
| 2021/12/1 Wed  | Seq2Seq Implementation                                       | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/models/seq2seq.py) |
| 2021/12/2 Thu  | Trainer.py, Train.py, Data Loader, continue_train.py         | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/models/seq2seq.py), [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/trainer.py) |
| 2021/12/3 Fri  | Inference, Greedy Search, Penalty, Search Function, Translate.py | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/translate.py) |
| 2021/12/4 Sat  | Evaluations(BLEU), Beam Search(Continue)                     | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/search.py) |
| 2021/12/5 Sun  | Beam Search, [Transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), [Rectified Adam](https://github.com/LiyuanLucasLiu/RAdam), [On Layer Normalization(Xiong et al., 2020)](http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf) | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/search.py) |
| 2021/12/6 Mon  | Transformer Implementation/Code Review                       | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/models/transformer.py) |
| 2021/12/7 Tue  | Transformer Implementation(Search & Beam Search)/Code Review | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/models/transformer.py) |
| 2021/12/8 Wed  | Intro to Advanced topics on NLG(Model Ensemble, Back/Copied Translation) |                                                              |
| 2021/12/9 Thu  | REINFORCE algorithm(with Baseline), [Minimum Risk Training for Neural Machine Translation](https://arxiv.org/pdf/1512.02433.pdf) |                                                              |
| 2021/12/10 Fri | [Minimum Risk Training](https://arxiv.org/abs/1512.02433), Reinforcement Learning Trainer | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/rl_trainer.py) |
| 2021/12/11 Sat | Reinforcement Learning Code Implementation                   | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/rl_trainer.py), [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/train.py) |
| 2021/12/12 Sun | [Dual Learning](https://arxiv.org/pdf/1707.00415.pdf)        |                                                              |
| 2021/12/13 Mon | Dual Supervised Learning Implementation                      | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/lm_trainer.py) |
| 2021/12/14 Tue | Dual Supervised Learning Implementation                      | [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/lm_trainer.py), [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/dual_train.py), [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/translate.py), [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/lm_train.py), [Code](https://github.com/saanghyuk/NLP/blob/main/NLG/simple-nmt/simple_nmt/models/rnnlm.py) |
| 2021/12/15 Wed | [Dual Learning for Machine Translation(Cycle GAN idea)](https://arxiv.org/pdf/1611.00179.pdf) |                                                              |



#### 4.PLM, BERT, GPT, HuggingFace**🤗**

| date           | contents                                                     |                                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2021/12/16 Thu | Introduction to PLM                                          |                                                              |
| 2021/12/17 Fri | Transfer Learning, Self-Supervised Learning, Embedding, [ELMo](https://arxiv.org/pdf/1802.05365.pdf) |                                                              |
| 2021/12/20 Mon | Transformer                                                  |                                                              |
| 2021/12/31 Fri | Seq2Seq, Transformer                                         |                                                              |
| 2021/01/02 Sun | Pre-trained Language Model(GPT, BERT, BART)                  |                                                              |
| 2021/01/04 Tue | Huggingface Transformers, BERT Trainer, Pytorch Ignite       | [Code](https://github.com/saanghyuk/NLP/blob/main/BERT_GPT/simple-ntc/simple_ntc/bert_trainer.py), [Code](https://github.com/saanghyuk/NLP/blob/main/BERT_GPT/simple-ntc/simple_ntc/bert_dataset.py), [Code](https://github.com/saanghyuk/NLP/blob/main/BERT_GPT/simple-ntc/finetune_plm_native.py), [Code](https://github.com/saanghyuk/NLP/blob/main/BERT_GPT/simple-ntc/classify_plm.py) |
| 2021/01/05 Wed | Huggingface Trainer, ALBERT, Knowledge Distillation, Compressed PLMs via Knowledge Distillation, Tiny BERT | [Code](https://github.com/saanghyuk/NLP/blob/main/BERT_GPT/simple-ntc/finetune_plm_hftrainer.py) |
| 2021/01/06 Thu | Text to Text framework(Google T5), [GPT-3(Language Models are Few-Shot Learners)](https://arxiv.org/pdf/2005.14165.pdf), [GPT crush](https://gptcrush.com/) |                                                              |





## Coursera [NLP](https://www.coursera.org/specializations/natural-language-processing)

#### 1. Natural Language Processing with Classification and Vector Spaces

| date           | contents                                                     |                                                              |
| -------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| 2021/01/07 Fri | Sentiment Analysis with Logistic Regression                  | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W1_Logistic%20Regression/C1_W1_lecture_nb_01_preprocessing.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W1_Logistic%20Regression/C1_W1_lecture_nb_02_word%20frequencies.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W1_Logistic%20Regression/C1_W1_lecture_nb_03_logistic_regression_model.ipynb) |
| 2021/01/08 Sat | Week 1 Assignment, Sentiment Analysis with Naïve Bayes, Week 2 Assignment | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W1_Logistic%20Regression/Assignment/C1_W1_Assignment_complete.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W2_Na%C3%AFve%20Bayes/C1_W2_lecture_nb_01_visualizing_naive_bayes.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W2_Na%C3%AFve%20Bayes/Assignment/C1_W2_Assignment_complete.ipynb) |
| 2021/01/09 Sun | Vector Space Models, Week 3 Assignment                       | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W3_Vector%20Space/C1_W3_lecture_nb_01_linear_algebra.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W3_Vector%20Space/C1_W3_lecture_nb_02_manipulating_word_embeddings.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W3_Vector%20Space/C1_W3_lecture_nb_03_pca.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W3_Vector%20Space/Assignment/C1_W3_Assignment_complete.ipynb) |
| 2021/01/10 Mon | Machine Translation, Document Search, Week 4 Assignment      | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W4_Machine%20Translation/C1_W4_lecture_nb_01_vector_manipulation.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W4_Machine%20Translation/C1_W4_lecture_nb_02_hash_functions_and_multiplanes.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C1/W4_Machine%20Translation/Assignment/C1_W4_Assignment_complete.ipynb) |

#### 2. Natural Language Processing with Probabilistic Models

| date           | contents                                                     |                                                              |
| :------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| 2021/01/11 Tue | Autocorrect and Minimum edit distance, Week 1 Assignment(Autocorrect) | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W1_Autocorrect/C2_W1_lecture_nb_01_building_the_vocabulary_model.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W1_Autocorrect/C2_W1_lecture_nb_02_candidates_from_edits.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W1_Autocorrect/Assignment/C2_W1_Assignment_complete.ipynb) |
| 2021/01/12 Wed | Part of Speech Tagging                                       | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W2_POS_tagging/C2_W2_lecture_nb_1_strings_tags.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W2_POS_tagging/C2_W2_lecture_nb_2_numpy.ipynb) |
| 2022/01/13 Thu | Part of Speech Tagging                                       |                                                              |
| 2022/01/14 Fri | Viterbi Algorithm, Week 2 Assignment                         | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W2_POS_tagging/Assignment/C2_W2_Assignment_complete.ipynb) |
| 2022/01/15 Sat | Autocomplete                                                 | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W3_Autocomplete/C2_W3_lecture_nb_01_corpus_preprocessing.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W3_Autocomplete/C2_W3_lecture_nb_02_building_the_language_model.ipynb) |
| 2022/01/16 Sun | Autocomplete, Perplexity, Week 3 Assignment                  | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W3_Autocomplete/C2_W3_lecture_nb_03_oov.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W3_Autocomplete/Assignment/C2_W3_Assignment_complete.ipynb) |
| 2022/01/17 Mon | Word Embeddings                                              |                                                              |
| 2022/01/18 Tue | Word Embeddings                                              | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W4_WordEmbeddings/C2_W4_lecture_nb_1_data_prep.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W4_WordEmbeddings/C2_W4_lecture_nb_2_intro_to_CBOW.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W4_WordEmbeddings/C2_W4_lecture_nb_3_training_the_CBOW.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W4_WordEmbeddings/C2_W4_lecture_nb_4_word_embeddings_hands_on.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W4_WordEmbeddings/C2_W4_lecture_nb_5_word_embeddings_step_by_step.ipynb) |
| 2022/01/19 Wed | Week 4 Assignment                                            | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C2/W4_WordEmbeddings/Assignment/C2_W4_Assignment_complete.ipynb) |

#### 3. Natural Language Processing with Sequence Models

| date           | contents                                                     |                                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2022/01/19 Wed | Neural Networks for Sentiment Analysis, [Trax](https://github.com/google/trax) | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W1_Neural_Networks_for_Sentiment%20Analysis/NLP_C3_W1_lecture_nb_01_trax_intro.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W1_Neural_Networks_for_Sentiment%20Analysis/NLP_C3_W1_lecture_nb_02_classes.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W1_Neural_Networks_for_Sentiment%20Analysis/NLP_C3_W1_lecture_nb_03_data_generators.ipynb) |
| 2022/01/21 Fri | Neural Networks for Sentiment Analysis                       | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W1_Neural_Networks_for_Sentiment%20Analysis/Assignment/C3_W1_Assignment.ipynb) |
| 2022/01/22 Sat | Week 1 Assignment, N-grams vs. Sequence Models               | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W1_Neural_Networks_for_Sentiment%20Analysis/Assignment/C3_W1_Assignment_complete.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W2_N-grams_vs_Sequence%20Models/C3_W2_lecture_nb_1_Hidden_State_Activation.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W2_N-grams_vs_Sequence%20Models/C3_W2_lecture_nb_2_RNNs.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W2_N-grams_vs_Sequence%20Models/C3_W2_lecture_nb_3_perplexity.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W2_N-grams_vs_Sequence%20Models/C3_W2_lecture_nb_4_GRU.ipynb) |
| 2022/01/23 Sun | Week 2 Assignment, LSTMs and Named Entity Recognition, [Gradient Descent Additional](https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/), Week 3 Assignment | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W2_N-grams_vs_Sequence%20Models/Assignment/C3_W2_Assignment_complete.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W3_%20LSTMs_and_Named_Entity_Recognition/Assignment/C3_W3_Assignment_complete.ipynb) |
| 2022/01/24 Mon | Siamese Networks                                             | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W4_Siamese_Networks/C3_W4_lecture_nb_1_siamese.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W4_Siamese_Networks/C3_W4_lecture_nb_2_Modified_Triplet_Loss.ipynb) |
| 2022/01/25 Tue | Siamese Networks, Week 4 Assignment                          | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C3/W4_Siamese_Networks/Assignment/C3_W4_Assignment_compelte.ipynb) |

#### 4. Natural Language Processing with Attention Models

| date           | contents                                                     |                                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2022/01/26 Wed | Neural Machine Translation                                   | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W1_Neural_Machine_Translation/C4_W1_Ungraded_Lab_1_Basic_Attention.ipynb) |
| 2022/01/27 Thu | Neural Machine Translation                                   |                                                              |
| 2022/01/28 Fri | Neural Machine Translation                                   |                                                              |
| 2022/01/30 Sun | Neural Machine Translation                                   | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W1_Neural_Machine_Translation/C4_W1_Ungraded_Lab_3_Bleu_Score.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W1_Neural_Machine_Translation/C4_W1_Ungraded_Lab_4_Stack_Semantics.ipynb) |
| 2022/01/31 Mon | Week 1 Assignment                                            |                                                              |
| 2022/02/01 Tue | Week 1 Assignment                                            | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W1_Neural_Machine_Translation/Assignment/C4_W1_Assignment.ipynb) |
| 2022/02/02 Wed | Text Summarization, [T5 Demo](https://t5-trivia.glitch.me/), [Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W2_Text_Summarization/C4_W2_Ungraded_Lab_1_Attention.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W2_Text_Summarization/C4_W2_Ungraded_Lab_2_Transformer_Decoder.ipynb) |
| 2022/02/03 Thu | Week 2 Assignment                                            | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W2_Text_Summarization/Assignment/C4_W2_Assignment_complete.ipynb) |
| 2022/02/04 Fri | Question Answering                                           |                                                              |
| 2022/02/16 Wed | Question Answering, BPE                                      | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W3_Question_Answering/C4_W3_SentencePiece_and_BPE.ipynb) |
| 2022/02/24 Thu | HuggingFace, Week 3 Assignment                               | [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W3_Question_Answering/Assignment/C4_W3_Assignment.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W3_Question_Answering/C4_W3_1_Question_Answering_with_BERT_and_HuggingFace_Pytorch_tydiqa.ipynb), [Code](https://github.com/saanghyuk/NLP/blob/main/Coursera_NLP/C4/W3_Question_Answering/C4_W3_2_Question_Answering_with_BERT_and_HuggingFace_Pytorch_tydiqa.ipynb) |
| 2022/02/25 Fri | [DSBA review for Reformer](https://www.youtube.com/watch?v=6ognBL6DEYM) |                                                              |



