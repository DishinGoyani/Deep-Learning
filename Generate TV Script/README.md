# TV Script Generation

In this project, you'll generate your own [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNNs.  You'll be using part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons.  The Neural Network you'll build will generate a new ,"fake" TV script, based on patterns it recognizes in this training data. This [project](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-tv-script-generation) is part of Udacity Deep Learning Nanodegree.

## Get the Data

The data is already provided for you in `./data/Seinfeld_Scripts.txt`

## Implement Pre-processing Functions
The first thing to do to any dataset is pre-processing.  Implement the following pre-processing functions below:
- Lookup Table
- Tokenize Punctuation

## Build the Neural Network
## Input
We'll use [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) to provide a known format to our dataset; in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), it will handle batching, shuffling, and other dataset iteration functions.

## Neural Network
Implement an RNN using PyTorch's [Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module). With an [long short-term memory (LSTM)](https://pytorch.org/docs/stable/nn.html?highlight=lstm#torch.nn.LSTM) RNN architecture.

## Model hyperparameters
- I had tried sequence length 10,12 with a dropout layer in FC. Later remove dropout and changed sequence length to 16 as a model was stop trained at 3.9 error after some epoch.
- Tried different learning rate 0.01, 0.001, 0.0001 tests for few batches of the dataset and select best fit 0.001
- embedding_dim and hidden_dim set 256 references from course material some best initial value is 50 or 200-300.
- n_layers set to 2 it usually outperform than 1 LSTM layer.
- Best Model saved with Loss: 3.251164775848389

### Generate a New Script
It's time to generate the text. Set `gen_length` to the length of TV script you want to generate and set `prime_word` to one of the following to start the prediction:
- "jerry"
- "elaine"
- "george"
- "kramer"

Ex.:
>jerry: honors, and the other friend, and i think it was a very good idea.
>
>george: i know, i was just sitting on my diaphragm.
>
>elaine: well, i was thinking, but i think he didn't be able to go.
>
>elaine: what?
>
>jerry: yeah.
>
>kramer: hey, hey, what about this?
>
>kramer:(to jerry) hey, i didn't even know what the odds is, and he was in the city.(she turns to the crowd)
>
>george:(entering monk's) well, you know what i think.
>
>jerry:(to george) i think you would.
>
>george: oh.(to jerry) you got the aids walk list?
>
>jerry: i don't know.
>
>george: i don't think so.
>
>jerry: you know, i was just thinking about the whole thing.
>
>jerry: you know what i do? you know, i just have a very special person. you know i think i can be a little more than a person.
>
>jerry: i know, i know.(to jerry) i can't.
>
>george: what?

You can set the prime word to _any word_ in our dictionary, but it's best to start with a name for generating a TV script. (You can also start with any other names you find in the original text file!)
### The TV Script is Not Perfect
You can see that there are multiple characters that say (somewhat) complete sentences, but it doesn't have to be perfect! It takes quite a while to get good results, and often, you'll have to use a smaller vocabulary (and discard uncommon words), or get more data.  The Seinfeld dataset is about 3.4 MB, which is big enough for our purposes; for script generation you'll want more than 1 MB of text, generally. 
