# nltk.download()
# nltk.download('punkt')
from nltk.corpus import reuters
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.util import ngrams
import math
from more_itertools import windowed


# Load the 'reuters' corpus
sentences = reuters.sents()

# Splitting data into Training, Development and Test set
train_sents, test_sents = train_test_split(reuters.sents(), test_size=0.3, random_state=42)
dev_sents, test_sents = train_test_split(test_sents, test_size=0.5, random_state=42)


# Transform the train sentences into words
train_words = [word for sentence in train_sents for word in sentence]
freq_dist_train = FreqDist(train_words)

cleaned_train_sentences = []
for sentence in train_sents:
    cleaned_train_sentence = [word if freq_dist_train[word] > 10 else 'UNK' for word in sentence]
    cleaned_train_sentences.append(cleaned_train_sentence)


# Transform the development sentences into words
dev_words = [word for sentence in dev_sents for word in sentence]
freq_dist_dev = FreqDist(dev_words)

cleaned_dev_sentences = []
for sentence in dev_sents:
    cleaned_dev_sentence = [word if freq_dist_dev[word] > 10 else 'UNK' for word in sentence]
    cleaned_dev_sentences.append(cleaned_dev_sentence)

# Transform the test sentences into words
test_words = [word for sentence in test_sents for word in sentence]
freq_dist_test = FreqDist(test_words)

cleaned_test_sentences = []
for sentence in test_sents:
    cleaned_test_sentence = [word if freq_dist_test[word] > 10 else 'UNK' for word in sentence]
    cleaned_test_sentences.append(cleaned_test_sentence)


# Build unigram, bigram and trigram counters for our training set
unigram_counter = Counter()
bigram_counter = Counter()
trigram_counter = Counter()

for sent in cleaned_train_sentences:

    unigram_counter.update([gram for gram in ngrams(sent, 1, pad_left=True, pad_right=True,
                                                    left_pad_symbol='<s>', right_pad_symbol='<e>')])
    bigram_counter.update([gram for gram in ngrams(sent, 2, pad_left=True, pad_right=True,
                                                       left_pad_symbol='<s>', right_pad_symbol='<e>')])
    trigram_counter.update([gram for gram in ngrams(sent, 3, pad_left=True, pad_right=True,
                                                        left_pad_symbol='<s>', right_pad_symbol='<e>')])
# print(unigram_counter.most_common(50))
# print(bigram_counter.most_common(50))
# print(trigram_counter.most_common(50))


# Define the hyperparameter alpha. Fine-tuning on the development set
alpha = 0.1

# Sum the tokens for the whole corpus (training, dev & test sets)
tokens = [token for sent in sentences for token in sent]
# Calculate vocabulary size (including any special tokens)
special_tokens = ['<s>', '<e>', 'UNK']
vocab_size = len(set(tokens + special_tokens))
# print(vocab_size)


def calculate_bigram_probability(ngram_counter, ngram_minus_one_counter, ngram, alpha, vocab_size):
    """
    Calculate bigram probability with Laplace smoothing
    :param bigram_counter: Counter which the key is a tuple of ngram and value its frequency
    :param gram_counter: Counter which the key is a tuple of n-1gram and value its frequency
    :param ngram: tuple
    :param alpha: float hyperparameter for Laplace smoothing
    :param vocab_size: int value which defines the whole size of the corpus
    :return: float probability of the ngram inside the corpus
    """
    ngram_count = ngram_counter[ngram]
    ngram_minus_one_count = ngram_minus_one_counter[(ngram[0],)]
    ngram_prob = (ngram_count + alpha) / (ngram_minus_one_count + (alpha * vocab_size))
    # if ngram_prob>0.6:
    #     print(f'ngram: {ngram}, ngram_count: {ngram_count}, ngram_minus_one_count: {ngram_minus_one_count}')
    return ngram_prob


# Calculate probability and Cross-Entropy of sentences in the test set
cross_entropy_bigram = 0.0
for sent in test_sents:
    # Pad the sentence with '<s>' and '<e>' tokens
    padded_sent = ['<s>'] + sent + ['<e>']

    # Iterate over the bigrams of the sentence
    for first_token, second_token in windowed(padded_sent, 2):
        if first_token == '<s>':
            print((first_token, second_token))
        else:
            bigram = (first_token, second_token)
            bigram_prob = calculate_bigram_probability(bigram_counter, unigram_counter, bigram, alpha, vocab_size)
            cross_entropy_bigram += math.log2(bigram_prob)

cross_entropy_bigram = -cross_entropy_bigram
print(f"The total Cross-Entropy for our Test set is: {cross_entropy_bigram}")

# Calculate and print the perplexity for the test set
num_tokens = sum(len(sent) + 1 for sent in test_sents)  # Including only end token for each sentence
perplexity = 2 ** (cross_entropy_bigram / num_tokens)
print("Perplexity for Test Set: {:.3f}".format(perplexity))

