# Image-captioning system
## Problem Statement
As price of cold storage have reduced drastically, cloud providers now provides image storage for free(debatable due to privacy issues). Examples are Facebook, Instagram, iCloud and Google photos where photos can be configured to automatically sync to the cloud in the background, so photos are not lost no matter what happens to the phone.

With an onslaught of photos on the web ( 350 Million photos daily on FB, 1.2 billion photos on Google Photos daily), it becomes a challenge to search across all these photos because most photos are not tagged with keywords. Without specific keywords being associated to the photos, the photos are not searchable.

## Data science problem
There are two main ways to solve this problem:

- Generate keywords through Multi label classification
- Generate captions

For this project, we built a model that automatically generates captions given photos as input. The generated captions can be stored in a search engine such as Elasticsearch / Solr and indexed so full text search can be performed across all the photos

## Training Set
For the training we used the well known Microsoft COCO dataset (version 2017) and the corresponding captions as the target.

## Modelling

An end to end Neural Network model which consist of:
A CNN encoder which translates the image into a fixed length vector representation that is passed in as the initial step for the RNN
A RNN ‘decoder’ which generates the target sentence, one word at a time. LSTM is used in this model.

Ref: A Neural Image Caption Generator

## Evaluation
Evaluation is done with the BLEU score.
The BLEU score is a string-matching algorithm that provides basic quality metrics for Machine Translation researchers and developers. Downsides includes:
Only measures direct word-to-word similarity and the extent to which word clusters in two sentences are identical
There is no consideration of paraphrases or synonyms 
"wander" doesn't get partial credit for "stroll," nor does "sofa" for "couch."
N-grams parameter can be set while evaluating BLEU score. Typically 1-4 grams are considered
