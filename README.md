# Part of Speech Tagger

This is part of the PyTorch tutorial on NLP found here:
http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py


Original author: Robert Guthrie


Modified by: Josh Morris


This tutorial originally showed how to create a part of speech tagger
based on word level representations using an LSTM. I extended it's
funcitonality to include character level representations as well.

### Usage:

```
python pos_tagger.py
```

### Example Output:

```
Variable containing:
-0.1986 -1.7407 -5.3472
-4.5588 -0.0134 -5.8828
-3.0268 -3.6396 -0.0777
-0.0330 -3.8059 -4.5860
-4.0360 -0.0193 -6.5085
[torch.FloatTensor of size 5x3]
```

This is the probability of the three pos tags:

```
"DET": 0, "NN": 1, "V": 2
```