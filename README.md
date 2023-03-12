# Neural-Language-Modelling
A language model for English built using a LSTM network

### Model for Ulysses
https://drive.google.com/file/d/1q78aLOSzYQUex-7yfXou1xI1FeIwxljv/view?usp=share_link
notebook with model: https://colab.research.google.com/drive/1ncqmN0MofFgp6rjY9Zur23g0nvZ8xasN?usp=sharing

### Model for Pride and Prejudice
https://drive.google.com/file/d/1dlTOLI55q4HRoQ1AtC5WM65pMZ7uSIFE/view?usp=sharing
notebok with model: https://colab.research.google.com/drive/1JwvwPBS_7FX1mUS-mFRijDdKvaZ8Tc5h?usp=share_link

### How to run neural_language_model.py

``` py neural_language_model.py <path_to_model>```

Also need to set the saved variable to True to ensure that the model is used instead of a new model being generated or trained

### Things considered during tokenization 
- Punctuation marks are removed
- All words are converted to lower case
- All numbers are replaced with the token "NUM"
- Cases with can't, won't were handled separately
- Cases with 's, 're, 've were handled separately
- Cases with 'm were handled separately
- Cases with 'll were handled separately
- Cases with 'd were handled separately
- Cases with 't were handled separately
- Hyphenated words were considered as one word, this was done especially because of the presence of words like to-morrow in the corpus 
- Excess space and hyphens were also removed individually 

### Note 
Rest of the important information has either been provided as comments in the code itself or in the report
