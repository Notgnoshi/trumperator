# Trumperator

Generating Trump tweets with deep learning

---

## Dependencies

* Keras (with Tensorflow GPU backend)
* Numpy
* Requests-HTML
* [trump_tweet_data_archive](http://www.trumptwitterarchive.com/)

## Getting the [dataset](https://github.com/bpb27/trump_tweet_data_archive)

Note that this will drastically inflate your `.git/` folder (3.2ish gigabytes). Saving the tweets in zipped files in a git repository is a mistake because it can't diff binary files very well. Thus, for every change, it just saves a binary snapshot.

Run `git submodule init` and then `git submodule update` to download the the Trump tweet repository.

Note that scraping the tweets myself isn't quite as simple as I'd hoped. The official Twitter API limits the number of tweets you can directly download from a user to 3200ish tweets. This is nowhere near what I need for a training set. So, the collection of tweets needs to parse the HTML of the Twitter use page.

## Generating tweets

For now I just have an experimental script that generates tweets as it trains.

## TODO

* Save the trained model
* Add regularization?
* Plot loss, etc.
* Figure out how the generation actually works
* What is the structure of the data?
* Given a trained model, generate a number of tweets
* Given a bunch of generated tweets, score them somehow (online poll?)
* Figure out what exactly the diversity stuff is
* Make training faster?
* Filter the training data better?
* Add validation data?
* Write the paper
* Add `paper/paper.pdf` to repository on final commit. (Don't want to track a binary file that will change a lot)
* Seed not taken directly from the corpus?
* Shorter seed
