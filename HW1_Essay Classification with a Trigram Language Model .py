import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Spring 2025
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1
    """
    padded_sequence = ["START"] * (n - 1) + sequence + ["STOP"]
    ngrams = [
        tuple(padded_sequence[i: i + n])
        for i in range(len(padded_sequence) - n + 1)
    ]

    return ngrams


# test

# tokens = ["I", "love", "coding"]
# n = 3
# print(get_ngrams(tokens, n))


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.total_num_words = sum(count for gram, count in self.unigramcounts.items(
        )) - self.unigramcounts[('START',)]

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        # might want to use defaultdict or Counter instead
        self.unigramcounts = defaultdict()
        self.bigramcounts = defaultdict()
        self.trigramcounts = defaultdict()
        # Your code here
        count = 0
        for sent in corpus:
            count += 1
            uni = get_ngrams(sent, 1)
            for one in uni:
                if one in self.unigramcounts:
                    self.unigramcounts[one] += 1
                else:
                    self.unigramcounts[one] = 1

            bi = get_ngrams(sent, 2)
            for two in bi:
                if two in self.bigramcounts:
                    self.bigramcounts[two] += 1
                else:
                    self.bigramcounts[two] = 1

            tri = get_ngrams(sent, 3)
            for three in tri:
                if three in self.trigramcounts:
                    self.trigramcounts[three] += 1
                else:
                    self.trigramcounts[three] = 1

        self.unigramcounts[('START',)] = count
        self.bigramcounts[('START', 'START')] = count
        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        # if unseen bigram
        # if self.bigramcounts[trigram[:2]] == 0:
        #     # approximate trigram (t1, t2, t3) prob with unigram prob of t3
        #     # 0 if unigram not seen
        #     # uniform distribution
        #     return self.unigramcounts[trigram[2:]]/self.denominator
        # elif trigram not in self.trigramcounts:
        #     return 0
        # else:
        #     return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]
        if trigram in self.trigramcounts:
            if trigram[0:2] in self.bigramcounts:
                return self.trigramcounts[trigram]/self.bigramcounts[(trigram[0:2])]
            else:
                return 1/len(self.lexicon)
        else:
            if trigram[0:2] in self.bigramcounts:
                return 0
            else:
                return 1/len(self.lexicon)

        # start start return 2/3 bigram  divide directly  trigarm = 0 1/tatal number words
        # bigram 00 unigram

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # if unseen unigram, return 0   # do I contain this part?
        if bigram in self.bigramcounts:
            if bigram[0:1] in self.unigramcounts:
                return self.bigramcounts[bigram]/self.unigramcounts[(bigram[0:1])]
            else:
                return 1/len(self.lexicon)
        else:
            if bigram[0:1] in self.unigramcounts:
                return 0
            else:
                return 1/len(self.lexicon)

        # # if self.unigramcounts[(bigram[0],)] == 0:
        #     return self.bigramcounts[bigram] / float(self.total_num_words)

        # return self.bigramcounts[bigram] / float(self.unigramcounts[(bigram[0],)])

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        if unigram == ('START',):
            return 0
        return self.unigramcounts[unigram] / self.total_num_words

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.
        # how do I store in the TrigramModel instance

        # if hasattr(TrigramModel, 'denominator') == False:
        #     self.denominator = 0
        #     for i in self.unigramcounts:
        #         self.unigramcounts[i]
        #         self.denominator += self.unigramcounts[i]
        # return self.unigramcounts[unigram]/self.denominator  # hasattr()

    # def generate_sentence(self, t=20):
    #     """
    #     COMPLETE THIS METHOD (OPTIONAL)
    #     Generate a random sentence from the trigram model. t specifies the
    #     max length, but the sentence may be shorter if STOP is reached.
    #     """
    #     return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        return lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(trigram[1:]) + lambda3*self.raw_unigram_probability(trigram[2:])

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        trigrams = get_ngrams(sentence, 3)
        logprob = 0.0
        for tri in trigrams:
            logprob += math.log2(self.smoothed_trigram_probability(tri))
        return logprob
        # return float("-inf") I dont need to write return like this right?

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        # l = 0.0
        # M = 0.0
        # for sent in corpus:
        #     l += self.sentence_logprob(sent)
        #     # add stop occurrences
        #     M += len(sent) + 1
        # l = l/M  # what is the meaning of m ;I know M
        # return 2**(-l)

        # Total number of words except Start
        tokens = sum(self.unigramcounts.values()) - \
            self.unigramcounts[('START',)]
        sum_prob = 0
        sum_tokens = 0
        for sentence in corpus:
            sum_prob += self.sentence_logprob(sentence)
            sentence_tokens = get_ngrams(sentence, 1)
            sum_tokens += (len(sentence_tokens)-1)

        return 2**(-sum_prob/sum_tokens)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        total += 1
        pp = model1.perplexity(corpus_reader(
            os.path.join(testdir1, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(
            os.path.join(testdir1, f), model2.lexicon))
        if pp < pp2:
            correct += 1

    for f in os.listdir(testdir2):
        total += 1
        pp = model2.perplexity(corpus_reader(
            os.path.join(testdir2, f), model2.lexicon))
        pp2 = model1.perplexity(corpus_reader(
            os.path.join(testdir2, f), model1.lexicon))
        if pp < pp2:
            correct += 1

    return correct/total


if __name__ == "__main__":
    # Testing Part I get_ngrams:
    model = TrigramModel('brown_train.txt')
    print(get_ngrams(["natural", "language", "processing"], 1))
    print(get_ngrams(["natural", "language", "processing"], 2))
    print(get_ngrams(["natural", "language", "processing"], 3))
    # Testing Part II counting n-grams:
    print(model.trigramcounts[('START', 'START', 'the')])
    print(model.bigramcounts[('START', 'the')])
    print(model.unigramcounts[('the',)])

    # Testing Part III raw n-gram:
    print(model.raw_unigram_probability(('the',)))
    print(model.raw_bigram_probability(('START', 'the')))
    print(model.raw_trigram_probability(('START', 'the', 'fulton')))

    # Testing Part IV smoothed probability:
    print(model.smoothed_trigram_probability(('START', 'START', 'the')))

    # Testing Part 5 Computing Sentence Probability
    print(math.log2(0.8))

    # Testing Part 6 perplexity:
    dev_corpus = corpus_reader('brown_test.txt', model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("testing preplexity is {:.5f}".format(pp))
    # print(pp)

    # Testing part 7 - Essay scoring experiment:
    training_file1 = "/Users/janicemeng/Desktop/NLP_HW1/train_high.txt"
    training_file2 = "/Users/janicemeng/Desktop/NLP_HW1/train_low.txt"
    testdir1 = "/Users/janicemeng/Desktop/NLP_HW1/test_high"
    testdir2 = "/Users/janicemeng/Desktop/NLP_HW1/test_low"
    acc = essay_scoring_experiment(
        training_file1, training_file2, testdir1, testdir2)
    print("model accuracy is {:.5f}".format(acc))
