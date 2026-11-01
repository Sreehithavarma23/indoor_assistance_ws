"""Dataset integrity: splits, intents, and no leakage."""

from src.nlu.dataset import dataset_summary
from src.nlu.normalization import normalize_text


def test_all_intents_present_in_both_splits(train_examples, test_examples):
    required = {
        "NAVIGATE",
        "RETRIEVE",
        "DELIVER",
        "FOLLOW",
        "STOP",
        "RETURN",
        "UNSUPPORTED",
    }
    assert required <= set(dataset_summary(train_examples))
    assert required <= set(dataset_summary(test_examples))


def test_no_normalized_overlap(train_examples, test_examples):
    train_texts = {normalize_text(item.text) for item in train_examples}
    overlap = [item.text for item in test_examples if normalize_text(item.text) in train_texts]
    assert overlap == []


def test_examples_have_entity_annotations(train_examples, test_examples):
    annotated = sum(1 for item in list(train_examples) + list(test_examples) if item.entities)
    assert annotated > 50


def test_train_is_larger_than_test(train_examples, test_examples):
    assert len(train_examples) > len(test_examples)
    assert len(train_examples) >= 150
