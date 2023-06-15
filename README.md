# Arabic Sequence Labeling: Part Of Speech Tagging NLP task
This project is implemented on arabic part of speech tagging as part of the "Natural Language Processing" course of my master's degree. 
the project uses the [arabic PUD dataset](https://github.com/UniversalDependencies/UD_Arabic-PUD) from [universal dependencies](https://universaldependencies.org/) and implements 
1. Deep learning model (BiLSTM) for sequential labeling classification
2. Pre-deep learning model (KNN) for multi-class classification 

## Table of contents
- [Arabic PUD Dataset](#arabic-pud-dataset)
- [Arabic Word Embedding](#arabic-word-embedding)
- [Structure BiLSTM sequential labeling classification model](#structure-bilstm-sequential-labeling-classification-model)
- [Results](#results)
- [Requirements](#requirements)
- [References and Resources](#references-and-resources)

## Arabic PUD Dataset
During preprocessing steps the following processes are applied :
1. Remove tanween and tashkeel
2. Remove sentences that contains non-arabic words (i.e. english characters) </ol>
Such that the distribution of tags within the dataset is visualized as barchart where the majority of
words (5553 word) in the dataset is associated with `noun` tag while the least common tag with the dataset is `X`.
Each of the tags symbolizes part of the speech, refer to the image below for description of each tag.
<br>
<p float="left">
  <img src="https://github.com/shaimaaK/arabic-sequence-classification-POS/assets/54285485/b35a8283-d93a-4334-b537-a1191ff7c5e5" width="500"  height="400"/>
  <img src="https://github.com/shaimaaK/arabic-sequence-classification-POS/assets/54285485/548ecb19-20cb-459a-aa0b-6330e7bbda09" width="300" height="400"/> 
</p>

## Arabic Word Embedding
Word embedding provides a **dense** representation of words and their relative meanings.<br>
The word embedding technique used in this project is N-Gram Word2Vec -SkipGram model from [aravec project](https://github.com/bakrianoo/aravec) trained on twitter data with vector size 300.
## Structure BiLSTM sequential labeling classification model 
<p float="left">
  <img src="https://github.com/shaimaaK/arabic-sequence-classification-POS/assets/54285485/643607db-5442-497f-bd90-a9aeec647640" width="500"  height="400"/>
</p>


## Results
The dataset is split to 70% for training and 30% for testing
### BiLSTM sequential labeling classification model
<p float="left">
  <img src="https://github.com/shaimaaK/arabic-sequence-classification-POS/assets/54285485/987535c7-6ad9-472b-9828-ca906960ca6a" width="400" />
  <img src="https://github.com/shaimaaK/arabic-sequence-classification-POS/assets/54285485/1f3b8137-7ab0-40f7-8dc6-1e73744c880c" width="400" /> 
</p>

### KNN multi-class classification model
![image](https://github.com/shaimaaK/arabic-sequence-classification-POS/assets/54285485/f8929c78-2c80-4f83-b2a6-4bd404669c85)

## Requirements
**Preprocessing and visualization**
- conllu
- matplotlib.pyplot
- pandas
- re
- seaborn
- numpy
- tensorflow (Tokenizer,pad_sequences)
- sklearn (preprocessing.LabelEncoder,model_selection.train_test_split) </ul>
<strong>Word Embedding</strong>
<ul>
  <li>gensim </li>
</ul>
<strong>Classification model </strong>
<ul>
  <li>tensorflow</li>
  <li>keras.models.sequential</li>
  <li> keras.layers (Dense,Embedding,Bidirectional,LSTM,TimeDistributed,InputLayer)</li>
  <li>sklearn.neighbors.KNeighborsClassifier</li>
</ul>
<strong>Model Evaluation </strong>
<ul>
  <li>sklearn.metrics </li>
</ul>

<h2>References and Resources</h2>
<ul>
  <li>reading and parsing dataset: <a href="https://www.youtube.com/watch?v=lvJRFMvWtFI">link</a></li>
  <li>Processing input data:<a href="https://medium.com/@WaadTSS/how-to-use-arabic-word2vec-word-embedding-with-lstm-af93858b2ce">link</a></li>
  <li>Aravec for word embedding model :<a href="https://github.com/bakrianoo/aravec">link</a></li>
  <li>Keras Embedding layer : <a href="https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/">link1</a>,<a href="https://medium.com/analytics-vidhya/understanding-embedding-layer-in-keras-bbe3ff1327ce">link2</a>,<a href="https://www.kaggle.com/code/rajmehra03/a-detailed-explanation-of-keras-embedding-layer">link3</a></li>
</ul>
