# Pyku

Generating haikus with deep learning

---

## Dependencies

* Keras (with Tensorflow GPU backend)
* Numpy
* Matplotlib
* Requests-HTML

## Useful links

* [haiku_rnn](https://github.com/napsternxg/haiku_rnn)
* [stack question](https://stackoverflow.com/questions/504428/how-would-you-write-a-program-to-generate-haiku)
* [haiku-generator](https://github.com/halflings/haiku-generator)
* [Gaiku](http://www.cs.brandeis.edu/~marc/misc/proceedings/naacl-hlt-2009/CALC-09/pdf/CALC-0905.pdf)

## TODO

* Understand stuff
    * How does LSTM work?
    * How should the training data be set up?
    * How can I use training, validation, and test data?
    * What exactly am I training?
* Once I have a model, how do I generate stuff with it?
    * What is this 'diversity' crap?
    * Why is the generation so computational?
    * How do I use a seed not in the corpus?
* Start experimenting
* Find more haikus to train on
* Read the Gaiku paper above
    * What can be done with word association norms, cosine similarity, etc?
    * Can natural language tools come in handy?
* Data representation
* And then there's the paper...
