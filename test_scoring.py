# Tests that scores of the sentences in variable-length batches are computed
# properly

import numpy as np
from src import toy_data as td

def test_score():

  SYMBOL = 0
  MIN_LENGTH = 3
  MAX_LENGTH = 10
  EXCESS = 5

  FULL_SENTENCE = [SYMBOL] * (MAX_LENGTH + EXCESS)

  symbol_score = td.numerical_vocab[SYMBOL]

  for length in range(MIN_LENGTH, MAX_LENGTH+1):
    s = td.score(FULL_SENTENCE, length)

    assert (s == symbol_score*length).all()
