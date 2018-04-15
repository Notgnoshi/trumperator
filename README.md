# Trumperator

Generating Trump tweets with deep learning

---

## Dependencies

* Keras (with Tensorflow GPU backend)
* Numpy
* [trump_tweet_data_archive](http://www.trumptwitterarchive.com/)
* spaCy NLP toolkit

  Install by running `sudo -H pip install --upgrade spacy` and then `sudo -H python3 -m spacy download en` to download the English language model.

## Getting the [dataset](https://github.com/bpb27/trump_tweet_data_archive)

Note that this will drastically inflate your `.git/` folder (3.2ish gigabytes). Saving the tweets in zipped files in a git repository is a mistake because it can't diff binary files very well. Thus, for every change, it just saves a binary snapshot.

Run `git submodule init` and then `git submodule update` to download the the Trump tweet repository.

Note that scraping the tweets myself isn't quite as simple as I'd hoped. The official Twitter API limits the number of tweets you can directly download from a user to 3200ish tweets. This is nowhere near what I need for a training set. So, the collection of tweets needs to parse the HTML of the Twitter use page.

## Generating tweets

The model is defined in `experiments/model.py`. To train that model, run `experiments/train.py`. This will save the trained model using the filename and parameters defined in `experiments/model.py`. To use an already-trained model, edit the filename in `experiments/model.py` and run the `experiments/generate.py` script, specifying the number of seeds to use as an integer argument.

Here are some of my favorite tweets generated so far.

> should be fired the office with a real congress comments on his republican spirit and ratings with a terrible many photo of @mittromney is always a fair #bush

> the best president obama is a great people to be a great president i will be interviewed on @foxandfriends tonight at pm enjoy

## Models

I've tried the following models with mixed success. Most of the models take anywhere from 5-35 minutes per epoch to train. Part of the reason why is the sheer size of the training set (900k+ items, plus a 200k+ validation set)

* 64x6 unit with batch normalization. 50 epochs with RMSprop

  15-20 epochs. ~1.4 loss. Pretty intelligible.
* 128 unit with dropout. 20 epochs with RMSprop
  
  20 epochs. Validation almost identical to training loss.
* 128 unit with dropout. 30 epochs with RMSprop

  30 epochs. Validation still mostly identical to training loss.
* 128 unit with batch normalization, gradient norm clipping, and nesterov momentum. 20 epochs with SGD

  20 epochs. Validation loss jumpy, but tends downward. Unintelligible.
* 128 unit with batch normalization, gradient norm clipping, and nesterov momentum. 50 epochs with SGD

  50 epochs. Validation wtill jumpy, but does tend downward still. Would probably keep going, but slow to train. More intelligible, but still mostly nonsense.
* 128x2 unit with batch normalization. 20 epochs with RMSprop

  5 epochs. ~1.4 loss. Pretty intelligible.
* 128x2 unit. 20 epochs with RMSprop

  5 epochs. ~1.45 loss. Training loss spiked at the end.
* 128x3 unit with batch normalization. 20 epochs with RMSprop

  6 epochs. ~1.39 loss. Validation loss really flattened. Fairly intelligible.
* 256 unit with batch normalization and small minibatches. 20 epochs with RMSprop

  7 epochs. Training and validation loss spiked at the end. Not very intelligible.
* 256 unit with dropout. 20 epochs with RMSprop

  5 epochs. Nice curves. Validation loss really flattened. Pretty intelligible
* 256 unit with batch normalization. 20 epochs with RMSprop

  5 epochs. Nice curves. Validation loss started to slowly increase. Fairly intelligible, some generated snippets were empty
* 256 unit with batch normalization and gradient clipping. 20 epochs with RMSprop

  6 epochs. Nice curves. Validation loss started to increase. Intelligible, but has loops.
* 256 unit with batch normalization, gradient clipping, and nesterov momentum. 200 epochs with SGD

  200 epochs. Validation loss spikey, but tends downwards. Really loves @mentions
* 256 unit with batch normalization, gradiend clipping, and nexterov momentum. 100 epochs with SGD and high learning rate

  40 epochs. Validation loss not very spikey, nice curves. Frequent loops, sometimes intelligible.
* 256x2 unit with batch normalization. 50 epochs with RMSprop

  10 epochs. Large gap between validation and training losses. Validation loss increasing. Fairly intelligible.
* 512 unit with batch normalization and large minibatches. 20 epochs with RMSprop

  5 epochs, after which validation loss increases substantially. Somewhat intelligible.
* 512 unit with batch normalization. 20 epochs with RMSprop

  6 epochs, after which validation loss icnreases substantially. Pretty intelligible.
* 512 unit with dropout. 20 epochs with RMSprop

  3 epochs, after which validation loss flattened. Pretty intelligible.
* 512x2 unit with batch normalization. 20 epochs with RMSprop

  5 epochs. After which both loses grew spikey. Fairly intelligible, but chops off words.
* 1024 unit with batch normalization. 20 epochs with RMSprop

  5 epochs. After which both losses grew quite spikey. All it did was repeat words over and over.
* 2048 unit with batch normalization. 50 epochs with RMSprop

  Worthless. Training loss barely changed, but validation loss was quite spikey. Generates actual words, but they mean nothing.

## TODO

* Given a bunch of generated tweets, score them somehow (online poll?)
  * Grammar checker (NLTK and SpaCy have grammar parsers)
  * Second neural network
    * Dr. Pyeatt leans this way, and actually to focus on this. (Classify tweet as written by user X or not written by user X.) I would need to generate a dataset containing tweets from multiple users.
  * Online poll, filter 'good' generated tweets by hand.
* Figure out what exactly the diversity stuff is
* Use a smaller training set? (Last two years of tweets?)
* Filter the training data better?
* Write the paper
* Add `paper/paper.pdf` to repository on final commit. (Don't want to track a binary file that will change a lot)
* After training full models, look for elbows and train for less epochs
