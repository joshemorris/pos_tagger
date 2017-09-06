# This is part of the PyTorch tutorial on NLP found here:
# http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
# Original author: Robert Guthrie
# Modified by: Josh Morris

# This tutorial originally showed how to create a part of speech tagger
# based on word level representations. I extended it's funcitonality to 
# include character level representatoins as well.

import torch 
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	tensor = torch.LongTensor(idxs)

	return autograd.Variable(tensor)

training_data = [
		("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
		("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
char_to_ix = {}

for sent, tags in training_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
		for char in list(word):
			if char not in char_to_ix:
				char_to_ix[char] = len(char_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

WORD_EMBEDDING_DIM = 6
CHAR_EMBEDDING_DIM = 3
WORD_HIDDEN_DIM = 6

class LSTMTagger(nn.Module):

	def __init__(self, word_embedding_dim, char_embedding_dim, word_hidden_dim, alpha_size, vocab_size, tagset_size):
		super(LSTMTagger, self).__init__()
		self.word_hidden_dim = word_hidden_dim
		self.char_hidden_dim = char_embedding_dim

		#embeddings
		self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
		self.char_embeddings = nn.Embedding(alpha_size, char_embedding_dim)

		# create lstm. 
		self.lstm_char = nn.LSTM(char_embedding_dim, char_embedding_dim)
		self.lstm_word = nn.LSTM(word_embedding_dim + char_embedding_dim, word_hidden_dim)

		self.hidden2tag = nn.Linear(word_hidden_dim, tagset_size)
		self.hidden_word = self.init_hidden(self.word_hidden_dim)
		self.hidden_char = self.init_hidden(self.char_hidden_dim)

	def init_hidden(self, dim):
		#axes semantics are (num_layers, minibatch_size, word_hidden_dim)
		return (autograd.Variable(torch.zeros(1, 1, dim)),
						autograd.Variable(torch.zeros(1, 1, dim)))

	def forward(self, sentence, chars):
		#create embeddings for words and characters
		word_embeds = self.word_embeddings(sentence)

		char_embeds = [self.char_embeddings(word) for word in chars]

		# get character level representation of each word
		char_rep = []
		for word in char_embeds:
			char_lstm_out, self.hidden_char = self.lstm_char(
				word.view(len(word), 1, -1), self.hidden_char)
			#append final representation for each word
			char_rep.append(char_lstm_out[-1])

		char_rep = torch.squeeze(torch.stack(char_rep, 1))

		word_embeds = torch.cat((word_embeds, char_rep), 1)

		#lstm of word embeddings catenated with char representations
		word_lstm_out, self.hidden_word = self.lstm_word(
			word_embeds.view(len(sentence), 1, -1), self.hidden_word)

		tag_space = self.hidden2tag(word_lstm_out.view(len(sentence), -1))
		tag_scores = F.log_softmax(tag_space)

		return tag_scores


model = LSTMTagger(WORD_EMBEDDING_DIM, CHAR_EMBEDDING_DIM, WORD_HIDDEN_DIM, len(char_to_ix), len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# before training
# element i,j = score for tag j for word i
# inputs = prepare_sequence(training_data[0][0], word_to_ix)
# char_in = [prepare_sequence(word, char_to_ix) for word in training_data[0][0]]
# tag_scores = model(inputs, char_in)
# print(tag_scores)


for epoch in range(300):
	for sentence, tags in training_data:
		#clear out prev grads
		model.zero_grad()

		#clear LSTM state
		model.hidden_word = model.init_hidden(WORD_HIDDEN_DIM)
		model.hidden_char = model.init_hidden(CHAR_EMBEDDING_DIM)

		#prep inputs
		sentence_in = prepare_sequence(sentence, word_to_ix)
		char_in = [prepare_sequence(word, char_to_ix) for word in sentence]
			
		targets = prepare_sequence(tags, tag_to_ix)

		#forward pass
		tag_scores = model(sentence_in, char_in)

		loss = loss_function(tag_scores, targets)
		loss.backward()
		optimizer.step()

inputs = prepare_sequence(training_data[0][0], word_to_ix)
char_in = [prepare_sequence(word, char_to_ix) for word in training_data[0][0]]
tag_scores = model(inputs, char_in)
print(tag_scores)