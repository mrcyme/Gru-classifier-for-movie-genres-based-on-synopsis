# Gru classifier for movie genres based on synopsis

This module provides a method to predict movie genres based on their synopsis.

It is a keras implementation of the multinomial GRU approach of this [paper](https://www.researchgate.net/publication/322517980_Predicting_Movie_Genres_Based_on_Plot_Summaries).

    @article{article,
    author = {Hoang, Quan},
    year = {2018},
    month = {01},
    pages = {},
    title = {Predicting Movie Genres Based on Plot Summaries}
    }

You can find test and train data on [kaggle](https://www.kaggle.com/c/radix-challenge/data).

The glove pretrain vectors can be downloaded [here](https://nlp.stanford.edu/projects/glove/).
Embedding to use for the code is the 100d embedding.

This code achieve a 68% Mean Average Precision at K score for the top 5 predicted genres.


