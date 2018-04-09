# Trumperator

Generating Trump tweets with deep learning

---

## Dependencies

* Keras (with Tensorflow GPU backend)
* Numpy
* Requests-HTML
* [trump_tweet_data_archive](http://www.trumptwitterarchive.com/)
* spaCy NLP toolkit

  Install by running `sudo -H pip install --upgrade spacy` and then `sudo -H python3 -m spacy download en` to download the English language model.

## Getting the [dataset](https://github.com/bpb27/trump_tweet_data_archive)

Note that this will drastically inflate your `.git/` folder (3.2ish gigabytes). Saving the tweets in zipped files in a git repository is a mistake because it can't diff binary files very well. Thus, for every change, it just saves a binary snapshot.

Run `git submodule init` and then `git submodule update` to download the the Trump tweet repository.

Note that scraping the tweets myself isn't quite as simple as I'd hoped. The official Twitter API limits the number of tweets you can directly download from a user to 3200ish tweets. This is nowhere near what I need for a training set. So, the collection of tweets needs to parse the HTML of the Twitter use page.

## Generating tweets

Here's an example from the `experiments/model.py` script's output.

> Generating with diversity: 0.2
>
> Generating with seed: "as great seeing and yesterday a smart negotiator would us"
>
> Generated:
>
> as great seeing and yesterday a smart negotiator would use the best thing they are doing a great job he is a great guy and the most important to be the next president of the united states  and the fake news media who can be the next president of the world

Or my personal favorite (so far)

> should be fired the office with a real congress comments on his republican spirit and ratings with a terrible many photo of @mittromney is always a fair #bush

> the best president obama is a great people to be a great president i will be interviewed on @foxandfriends tonight at pm enjoy

## TODO

* Given a bunch of generated tweets, score them somehow (online poll?)
  * Grammar checker (NLTK has a grammar parser)
  * Second neural network
    * Dr. Pyeatt leans this way, and actually to focus on this. (Classify tweet as written by user X or not written by user X.) I would need to generate a dataset containing tweets from multiple users.
  * Online poll, filter 'good' generated tweets by hand.
* Figure out what exactly the diversity stuff is
* Use a smaller training set? (Last two years of tweets?)
* Make training faster?
* Filter the training data better?
* Write the paper
* Add `paper/paper.pdf` to repository on final commit. (Don't want to track a binary file that will change a lot)
* Seed not taken directly from the corpus?
  * If the seed is taken from the corpus, make it not cross tweet boundaries?
* Is this the right data representation?
  * The dataset is treated as a single corpus, should it instead be a collection of corpuses?
* After training full models, look for elbows and train for less epochs
* Add `requirements.txt` for `pip` to install from?
