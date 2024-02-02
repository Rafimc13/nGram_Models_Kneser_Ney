from nltk.tokenize import word_tokenize

def beam_search_spelling(sentence, beam_width, l1, l2, generate_candidates_fn, score_fn):
    """
    Spelling correction with contect awereness using beam search
    :param sentence:
    :param beam_width:
    :param l1:
    :param l2:
    :param generate_candidates_fn:
    :param score_fn:
    :return:
    """
    print(l1,l2)
    initial_state = ['<s>','<s>']
    candidates = [(initial_state, 0)]
    # sentence = word_tokenize(sentence)
    max_depth = len(sentence)
    for depth in range(max_depth):
        new_candidates = []
        for candidate, prob in candidates:
            for next_state, dist in generate_candidates_fn(candidate, sentence[depth],vocab):

                # Prob we add the previous prob, the prob of the next state and the inverse of the distance
                new_prob = prob + score_fn(next_state,len(vocab), trigram_counter, bigram_counter, prefixes_counter_tri,'trigram', dist, l1, l2)

                new_candidates.append((next_state, new_prob))


        new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)

        candidates = new_candidates[:beam_width]
        print(candidates)
    best_sequence, best_prob = max(candidates, key=lambda x: x[1])
    return best_sequence[2:]


test_sentence = word_tokenize("The deparmt office reprts ")
beam_width = 5
best_sequence = beam_search_spelling(test_sentence, beam_width, 0.9, 0.1, generate_candidate_with_distance, score)
print("The corrected sentence is", end=': ')
print(' '.join(best_sequence))  # Excluding the "<start>" token