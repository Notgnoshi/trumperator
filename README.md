# Trumperator

Generating Trump tweets with deep learning

---

## Dependencies

* Keras (with Tensorflow GPU backend)
* Numpy
* [trump_tweet_data_archive](http://www.trumptwitterarchive.com/)
* [Textgenrnn](https://github.com/minimaxir/textgenrnn)
* <strike>spaCy NLP toolkit</strike>

  Install by running `sudo -H pip install --upgrade spacy` and then `sudo -H python3 -m spacy download en` to download the English language model.

## Getting the [dataset](https://github.com/bpb27/trump_tweet_data_archive)

Note that this will drastically inflate your `.git/` folder (3.2ish gigabytes). Saving the tweets in zipped files in a git repository is a mistake because it can't diff binary files very well. Thus, for every change, it just saves a binary snapshot.

Run `git submodule init` and then `git submodule update` to download the the Trump tweet repository. This will also download the Textgenrnn submodule, which is also installable with Pip:

```shell
sudo -H pip install --upgrade textgenrnn
```

Note that scraping the tweets myself isn't quite as simple as I'd hoped. The official Twitter API limits the number of tweets you can directly download from a user to 3200ish tweets. This is nowhere near what I need for a training set. So, the collection of tweets needs to parse the HTML of the Twitter use page.

## Generating tweets

The model is defined in `experiments/model.py`. To train that model, run `experiments/train.py`. This will save the trained model using the filename and parameters defined in `experiments/model.py`. To use an already-trained model, edit the filename in `experiments/model.py` and run the `experiments/generate.py` script, specifying the number of seeds to use as an integer commandline argument.

Here are some of my favorite tweets generated so far.

> should be fired the office with a real congress comments on his republican spirit and ratings with a terrible many photo of @mittromney is always a fair #bush
>
> the best president obama is a great people to be a great president i will be interviewed on @foxandfriends tonight at pm enjoy

If you want to try to more sophisticated [Textgenrnn](https://github.com/minimaxir/textgenrnn) model, trained by me, run the following in the `experiments/` folder.

```python
from textgenrnn.textgenrnn import textgenrnn
# Files found in trumperator/experiments/
t = textgenrnn(config_path='trumperator_config.json',
               vocab_path='trumperator_vocab.json',
               weights_path='trumperator_weights.hdf5')
t.generate(10, temperature=0.2)
```

## The Paper

The paper is found in `paper/paper.pdf`, or can be made with the provided makefile.
