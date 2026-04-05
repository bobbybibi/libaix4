"""Unit tests for the BagOfWords vectorizer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from vectorizer import BagOfWords, tokenize


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("What is TCP?")
        assert "tcp" in tokens
        assert "is" not in tokens  # stop word

    def test_removes_punctuation(self):
        tokens = tokenize("Hello, world! How's it going?")
        assert all(t.isalnum() for t in tokens)

    def test_removes_short_tokens(self):
        tokens = tokenize("I am a big fan of AI")
        assert "a" not in tokens

    def test_lowercase(self):
        tokens = tokenize("TCP UDP HTTP")
        assert tokens == ["tcp", "udp", "http"]


class TestBagOfWords:
    def test_fit_builds_vocab(self):
        bow = BagOfWords()
        bow.fit(["hello world", "hello python"])
        assert "hello" in bow.vocab
        assert "world" in bow.vocab
        assert "python" in bow.vocab

    def test_vocab_size(self):
        bow = BagOfWords()
        bow.fit(["cat dog bird", "fish cat"])
        assert bow.vocab_size == 4  # cat, dog, bird, fish

    def test_transform_shape(self):
        bow = BagOfWords()
        bow.fit(["hello world", "foo bar baz"])
        X = bow.transform(["hello foo"])
        assert X.shape == (1, bow.vocab_size)

    def test_transform_before_fit_raises(self):
        bow = BagOfWords()
        with pytest.raises(RuntimeError, match="fit"):
            bow.transform(["test"])

    def test_fit_transform(self):
        bow = BagOfWords()
        X = bow.fit_transform(["hello world", "hello python"])
        assert X.shape == (2, bow.vocab_size)
        assert bow._fitted

    def test_tfidf_weighting(self):
        bow = BagOfWords()
        _X = bow.fit_transform(["tcp protocol", "udp protocol"])
        # "protocol" appears in both docs so gets lower IDF
        # "tcp" appears in 1 doc so gets higher IDF
        tcp_idx = bow.vocab["tcp"]
        proto_idx = bow.vocab["protocol"]
        assert bow.idf[tcp_idx] > bow.idf[proto_idx]

    def test_l2_normalization(self):
        bow = BagOfWords()
        X = bow.fit_transform(["tcp udp http", "dns dhcp"])
        norms = np.linalg.norm(X, axis=1)
        np.testing.assert_almost_equal(norms, [1.0, 1.0])

    def test_unknown_words_ignored(self):
        bow = BagOfWords()
        bow.fit(["hello world"])
        X = bow.transform(["xyz unknown words"])
        assert np.sum(X) == 0.0  # all zeros

    def test_save_load_round_trip(self):
        bow = BagOfWords()
        bow.fit(["tcp protocol networking", "http web server"])
        X_before = bow.transform(["tcp server"])

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vec.json"
            bow.save(path)
            loaded = BagOfWords.load(path)

        X_after = loaded.transform(["tcp server"])
        np.testing.assert_array_almost_equal(X_before, X_after)
        assert loaded.vocab == bow.vocab


class TestNgrams:
    def test_bigrams_in_vocab(self):
        bow = BagOfWords(max_n=2)
        bow.fit(["tcp protocol works", "udp protocol fast"])
        # Should have both unigrams and bigrams
        assert "tcp" in bow.vocab
        assert "tcp_protocol" in bow.vocab
        assert "protocol_works" in bow.vocab

    def test_bigram_increases_vocab(self):
        bow_uni = BagOfWords(max_n=1)
        bow_bi = BagOfWords(max_n=2)
        texts = ["tcp protocol works", "udp protocol fast"]
        bow_uni.fit(texts)
        bow_bi.fit(texts)
        assert bow_bi.vocab_size > bow_uni.vocab_size

    def test_trigrams(self):
        bow = BagOfWords(max_n=3)
        bow.fit(["tcp protocol works well"])
        assert "tcp_protocol_works" in bow.vocab

    def test_bigram_transform_shape(self):
        bow = BagOfWords(max_n=2)
        X = bow.fit_transform(["tcp protocol", "udp protocol"])
        assert X.shape == (2, bow.vocab_size)

    def test_bigram_save_load(self):
        bow = BagOfWords(max_n=2)
        bow.fit(["tcp protocol networking", "http web server"])
        X_before = bow.transform(["tcp protocol"])

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vec.json"
            bow.save(path)
            loaded = BagOfWords.load(path)

        assert loaded.max_n == 2
        X_after = loaded.transform(["tcp protocol"])
        np.testing.assert_array_almost_equal(X_before, X_after)

    def test_min_df_filters_rare(self):
        bow = BagOfWords(max_n=1, min_df=2)
        bow.fit(["tcp protocol", "udp protocol", "rare_unique_word"])
        # "protocol" appears in 2 docs, should be kept
        assert "protocol" in bow.vocab
        # "rare_unique_word" only in 1 doc, should be filtered out
        assert "rare_unique_word" not in bow.vocab

    def test_default_max_n_is_unigram(self):
        bow = BagOfWords()
        assert bow.max_n == 1

    def test_unigram_backward_compat(self):
        """Default max_n=1 should match old behaviour exactly."""
        bow_old = BagOfWords(max_n=1)
        bow_new = BagOfWords()
        texts = ["tcp protocol networking", "http web server"]
        X_old = bow_old.fit_transform(texts)
        X_new = bow_new.fit_transform(texts)
        np.testing.assert_array_equal(X_old, X_new)
