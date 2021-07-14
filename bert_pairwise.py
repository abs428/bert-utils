"""This script contains code that allows a use the raw BERT emeddings in order to
train a classification model based on the same. The code below provides that allows 
the user to train a pairwise sentence classifier without getting into too much detail.

Example
-------
>>> import pandas as pd
>>> from bert_pairwise import encode_sentence_pair, extract_bert_embeddings, train_classifier
>>> RETAIL_DESC, SOURCE_DESC = 'ONLINE DESCRIPTION', 'SOURCE DESCRIPTION'
>>> LABEL = 'tag'
>>> BATCH_SIZE = 1000
>>> PADDING = 46

# Training the classifier
>>> classifier, test_set = train_classifier(train_df, LABEL, train_features)

# Using BERT and trained classifier
>>> vs_data = pd.read_excel("vs_pair_data.xlsx")
>>> vs_input_ids, vs_attn_mask = encode_sentence_pair(vs_data, RETAIL_DESC, SOURCE_DESC, tokenizer, PADDING)
>>> vs_features = extract_bert_embeddings(vs_input_ids, vs_attn_mask, model, BATCH_SIZE)
>>> vs_data['predictions'] = classifier.predict(vs_features)
>>> vs_data['predict_proba'] = classifier.predict_proba(vs_features)[:,1]
"""
import numpy as np
import pandas as pd
import torch
import tqdm
import transformers
from collections import namedtuple
import typing as t
from typing import NamedTuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split

ModelProperties = namedtuple(
    "ModelProperties", ["model_class", "tokenizer", "model_object"]
)
BERT_TYPES = {
    "distilbert": ModelProperties(
        model_class="DistilBertModel",
        tokenizer="DistilBertTokenizer",
        model_object="distilbert-base-uncased",
    )
}

class ModelingResult(NamedTuple):
    """The object that is returned after the training

    Parameters
    ----------
    classifier: 
        The trained classifier that takes in word embeddings as input
    
    test_set: pd.DataFrame
        The internal test set which is used to evaluate the performance
        of the model (and also tune it if one desires)
    
    tokenizer:
        The tokenizer object used
    
    padding: int
        The padding that was used
    
    bert_model:
        The pre-trained BERT model object that was used
    
    predictions: np.array
        The array of predictions on the production dataset (if provided as input)
    
    probabilities: np.array
        The array of predicted probabilities on the production dataset (if provided 
        as input)
    """
    classifier : t.Any
    test_set: pd.DataFrame
    tokenizer: t.Any
    padding: int
    bert_model: t.Any
    predictions: np.array
    probabilities: np.array

def encode_sentence_pair(
    data: pd.DataFrame, sent1: str, sent2: str, tokenizer, padding: int = None
):
    """Takes in a dataframe as input with columns containing
    fields to be used as sentence pairs. Returns input_ids tensor and
    the attention mask. If padding MAX_LEN is the default, however, please
    keep the padding the same while training and testing."""

    assert (sent1 in data) and (sent2 in data), "Fields not present in DF."

    # Tokenization
    tokenized1 = data[sent1].apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True))
    )
    tokenized2 = data[sent2].apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True))
    )

    # Bad code, please fix later
    sentences = []
    for i in range(len(data)):
        sentences.append(tokenized1.iloc[i] + tokenized2.iloc[i][1:])

    if not padding:
        max_len = 0
        for i in sentences:
            if len(i) > max_len:
                max_len = len(i)
        print(f"Note: Maximum length {max_len} chosen as padding.")
        padding = max_len

    # HACK: Will clip the sentence to the padding
    # FIXME: Hack is to prevent the code from breaking when len(i) > padding
    sentences = [sentence[:padding] for sentence in sentences]
    padded = np.array([i + [0] * (padding - len(i)) for i in sentences])
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    attention_mask_ = torch.tensor(attention_mask)
    return input_ids, attention_mask_


def extract_bert_embeddings(input_ids, attention_mask, model, batch_size: int):

    n_dim = 768  # Dimensionality of embeddings returned by BERT
    n_records = len(input_ids)
    n_iters = n_records // batch_size

    print(f"# of iterations: {n_iters}")

    # Initializing result tensor
    features = torch.zeros((0, n_dim))

    with torch.no_grad():
        for idx in tqdm.notebook.tqdm(range(0, n_records, batch_size)):
            last_hidden_states = model(
                input_ids[idx : idx + batch_size],
                attention_mask=attention_mask[idx : idx + batch_size],
            )
            features = torch.cat((features, last_hidden_states[0][:, 0, :]))

    return features


def train_classifier(
    data, label_col, features, clf=LogisticRegression(), random_state: int = 1978
):
    """Function that trains a classifier and returns a trained model object along
    with the predictions on the test_set.

    data: pd.DataFrame
      Data with original records

    label_col:str
      Field that contains the target variable

    features: 2D array-like
        An array containing feature. In this script it is generally the embeddings
        generated by the BERT model.

    clf: sklearn classfier
      Initialized classfier. Logistic Regression by default

    random_state: int
      Random state
    """
    labels = data[label_col]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, random_state=random_state
    )
    clf.fit(train_features, train_labels)
    print(f"Generating classfication report: ")

    preds = clf.predict(test_features)
    print(classification_report(y_true=test_labels, y_pred=preds))
    a, test_df, b, test_label_check = train_test_split(
        data, labels, random_state=random_state
    )

    # Sanity check
    assert (test_df[label_col] == test_label_check).all()
    assert "prediction" not in test_df
    test_df["prediction"] = preds
    test_df["predict_proba"] = clf.predict_proba(test_features)[:, 1]
    return clf, test_df


def train_model(
    train_data: pd.DataFrame,
    text_col_1: str,
    text_col_2: str,
    label_col: str,
    prod_data: pd.DataFrame = None,
    padding: int = None,
    batch_size: int = 1000,
    bert_type: str = "distilbert",
    sampling: str = None,
    classifier=LogisticRegression(),
) -> ModelingResult:
    """Main function that brings together all the pieces.

    Parameters
    ----------
    train_data: pd.DataFrame
        The DataFrame that contains two columns to be used as
        inputs to BERT along with a column that contains a label
        for every pair of text

    text_col_1: str
        The field that contains the text to be sent as input to BERT

    text_col_2: str
        Another field that contains the text to be sent as input to BERT

    label_col: str
        Field that contains the labels for the classifier

    prod_data: pd.DataFrame, default=None
        The production data that takes in a DataFrame similar to `train_data`
        but does not contain the label_col field. The field names must be
        identical to the ones used in `train_data`. If `prod_data` is not
        passed any input only the trained classier and the internal test set
        is returned as output. When given valid input, predictions and the
        corresponding probabilities are also returned additionally.

    padding: int, default=None
        Can be interpreted as one of the dimensions of the word embeddings
        generated ie:- the length of the vector used to represent each of
        the strings in the DataFrame.
        TODO: This description needs more work

    batch_size: int, default=1000
        The number of string-pairs whose embeddings are computed in one
        batch. Too big a number would result in a MemoryError.

    bert_type: str, default='distilbert'
        The version of the BERT model to be used

    sampling: str, {'under'}, default=None
        Undersamples the majority class if 'under' is passed. Other
        sampling methods have not been implemented yet.

    classifier: default=LogisticRegression()
        A sklearn classfier that is trained on the embeddings generated
    
    Returns
    -------
    A ModelResult object

    Raises
    ------
    AssertionError
        If some sanity checks fail
    """
    # Sanity checks
    assert bert_type in BERT_TYPES, f"BERT type {bert_type} not supported."
    assert train_data.isnull().any(None), "Nulls not permitted in train data."
    assert prod_data.isnull().any(None), "Nulls not permitted in production data."
    assert (
        (label_col in train_data)
        and (text_col_1 in train_data)
        and (text_col_2 in train_data)
    ), "Specified columns not present in training data."
    assert (text_col_1 in prod_data) and (
        text_col_2 in prod_data
    ), "Text fields specifie are not present in production data."

    properties = BERT_TYPES[bert_type]
    model_class, tokenizer_class, pretrained_weights = (
        getattr(transformers, properties.model_class),
        getattr(transformers, properties.tokenizer),
        properties.model_object,
    )
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Sampling if there is class imbalance
    # TODO: detect imbalance automatically

    if sampling:
        if sampling != "under":
            raise ValueError("Only undersampling has been implemented.")

        from imblearn.under_sampling import RandomUnderSampler

        sampler = RandomUnderSampler(sampling_strategy="majority")

        X = train_data[[col for col in train_data if col != label_col]]
        y = train_data[label_col]

        X_under, y_under = sampler.fit_resample(X, y)
        sampled_df = pd.DataFrame(X_under, columns=X.columns)
        sampled_df[label_col] = y_under
    else:
        sampled_df = train_data.copy()

    if padding is None:
        padding = max(
            (
                train_data[text_col_1].str.len().max(),
                prod_data[text_col_1].str.len().max(),
                train_data[text_col_2].str.len().max(),
                prod_data[text_col_2].str.len().max(),
            )
        )
        print(f"Padding set to {padding} because a default was not provided.")

    # Encoding input text pairs
    print("Encoding text pairs from training data...")
    train_input_ids, train_attn_mask = encode_sentence_pair(
        sampled_df, text_col_1, text_col_2, tokenizer, padding
    )
    train_features = extract_bert_embeddings(
        train_input_ids, train_attn_mask, model, batch_size
    )
    print("Training classifier...")
    classifier, test_set = train_classifier(
        sampled_df, label_col, train_features, clf=classifier
    )

    if prod_data is None:
        return ModelingResult(classifier=classifier,
        test_set=test_set,
        tokenizer=tokenizer,
        padding=padding,
        bert_model=model,
        predictions=None,
        probabilities=None,
        )

    print("Encoding text pairs from production data...")
    prod_input_ids, prod_attn_mask = encode_sentence_pair(
        prod_data, text_col_1, text_col_2, tokenizer, padding
    )
    prod_features = extract_bert_embeddings(
        prod_input_ids, prod_attn_mask, model, batch_size
    )

    print("Predicting on production data...")
    predictions = classifier.predict(prod_features)
    probabilities = classifier.predict_proba(prod_features)

    return ModelingResult(classifier=classifier,
        test_set=test_set,
        tokenizer=tokenizer,
        padding=padding,
        bert_model=model,
        predictions=predictions,
        probabilities=probabilities,
        )