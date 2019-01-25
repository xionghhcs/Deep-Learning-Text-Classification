# Deep Learning Text Classification

Text classification is a fundamental task in Natural Language Processing. The goal of text classification is to assign labels to text. Traditional approaches of text classification attempt to map text into a fixed vector, such as tfidf, and then classify it to one class or more than one.

In this repository, I focus on deep learning methods in text classification field. Deep learning method, such as CNN, LSTM, MemNN, etc, have been broadly used in text representatation learning. We can finally classify text according to the representation. 


## Environment

- tensorflow 1.12.0
- python 3.5
- pandas
- numpy
- nltk

## Introduction

Text classification technology can be roughly divided into two categories, traditional text classification and deep learning text classification. Traditional text classification mainly focus on feature engineering. The text will be represented as a sparse vector like bag-of-word or n-gram and then feed into machine learning classifiers such as lr, svm and so on. For deep learning methods, the models will be train in an end-to-end way. The text feature will be extract by the neural network automatically. Researchers have designed many neural networks for text classification task like TextCNN, HAN, memory network, TMN, etc. Sometimes, we can use the seq2seq framework's encoder part to encode the text. This repository reimplement some deep learning text classification models based on my own understanding.

## Models

### TextCNN

![text cnn](./assert/text_cnn_model.png)

#### Implement reference

[1] https://github.com/cmasch/cnn-text-classification/blob/master/cnn_model.py

### TextRNN


### Fasttext

![](./assert/fasttext.png)

### LSTM_GRNN


![lstm_grnn](./assert/lstm_grnn_model.png)


[1] Duyu Tang .et, "Document Modeling with Gated Recurrent Neural Network for Sentiment Classification." ACL'2015

### HAN

![han](./assert/han_model.png)

Related paper:

[1] Zichao Yang .etc, "Hierarchical Attention Networks for Document Classification." 


### RCNN

![rcnn](./assert/rcnn_model.png)

Implement reference：

[1] https://github.com/roomylee/rcnn-text-classification

### Dynamic Memory Network

![dynamic memory network](./assert/dynamic_memory_network.png)


### Char-Level Convolutional Network

![](./assert/char-level-conv-net.png)

### VDCNN

![](./assert/vdcnn.png)

### Transformer's Encoder

Transformer is proposed by google in the paper of "Attention is all you need".In this text classification task, we only use the encoder of Transformer to learn a text representation.

![](./assert/transformer.png)

Implement reference:

[1] https://github.com/Lsdefine/attention-is-all-you-need-keras

### Google Universal Sentence Encoder


### TMN (Topic Memory Network)

![tmn](./assert/tmn.png)

### TextGCN

![](./assert/text_gcn.png)

### Bi-BloSAN

![](./assert/Bi-BloSAN.jpg)

## Paper list

[1] Convolutional Neural Networks for Sentence Classification

[2] Character-level Convolutional Networks for Text Classification

[3] Document Modeling with Gated Recurrent Neural Network for Sentiment Classificaton

[4] Hierarchical Attention Networks for Document Classification

[5] Ask Me Anything: Dynamic Memory Networks for Natural Language Processing

[6] Bag of Tricks for Efficient Text Classification

[7] Very Deep Convolutional Networks for Text Classification

[8] Attention is all your need

[9] Topic Memory Networks for Short Text Classification

[10] Graph Convolutional Neural Networks for Text Classification

[11] Bi-Directional Block Self-Attention for Fast and Memory-Efficient Sequence Modeling

[12] 



## Reference

[1] http://nlpprogress.com/english/sentiment_analysis.html

[2] https://www.jiqizhixin.com/articles/2018-10-23-6 

[3]  

[4] 

