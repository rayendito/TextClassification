import pandas as pd
import argparse
import os

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, f1_score

import pickle

import warnings
import logging
import time

logging.basicConfig(level=logging.INFO)

def preprocess(col):
    col_copy = col.copy()
    col_copy = col_copy.str.lower()
    col_copy = col_copy.str.replace(r'[^a-z\s]', ' ', regex=True)
    col_copy = col_copy.str.replace(r'\s+', ' ', regex=True).str.strip()
    return col_copy

parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_val', action='store_true')
parser.add_argument('--do_predict', action='store_true')
parser.add_argument('--model_path', type=str, default='model')
parser.add_argument('--train', type=str, default="../data_worthcheck/train.csv")
parser.add_argument('--dev', type=str, default="../data_worthcheck/dev.csv")
parser.add_argument('--test', type=str, default="../data_worthcheck/test.csv")
args = parser.parse_args()

if not args.do_train and not args.do_val and not args.do_predict:
    raise ValueError('At least one of --do_train, --do_val, --do_predict must be True.')

if args.do_train:
    train_df = pd.read_csv(args.train)
    sample_data = train_df.sample(1, random_state=42)
    sample_text = sample_data['text_a'].values[0]
    sample_label = sample_data['label'].values[0]

    train_df['text'] = preprocess(train_df['text_a'])

    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])

    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2),analyzer='word')),
        ('clf', MultinomialNB(alpha=0.1)),
    ])
    # log training
    logging.info('Training model...')
    logging.info(f"Sample text: {sample_text} | Sample label: {sample_label}")
    start = time.time()
    model.fit(train_df['text'], train_df['label'])
    end = time.time()
    logging.info(f"Training took {end - start} seconds for {len(train_df)} training data.")

    if os.path.exists(args.model_path):
        warnings.warn('Model file already exists. Overwriting?')
        confirmation = input('Type "yes" to confirm: ')
        if confirmation != 'yes':
            raise ValueError('Model file already exists. Aborting.')
        else:
            print('Overwriting model file.')
    # make directory
    os.makedirs(args.model_path, exist_ok=True)
    # save model
    with open(os.path.join(args.model_path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    # save label encoder
    with open(os.path.join(args.model_path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
if args.do_val or args.do_predict:
    if not os.path.exists(args.model_path):
        raise ValueError('Model file does not exist. Aborting.')
    with open(os.path.join(args.model_path, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(args.model_path, 'label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)

if args.do_val:
    val_df = pd.read_csv(args.dev)
    sample_data = val_df.sample(1, random_state=42)
    sample_text = sample_data['text_a'].values[0]
    sample_label = sample_data['label'].values[0]

    val_df['text'] = preprocess(val_df['text_a'])
    val_df['label'] = le.transform(val_df['label'])
    logging.info('Validating model...')
    logging.info(f"Sample text: {sample_text} | Sample label: {sample_label}")
    start = time.time()
    y_pred = model.predict(val_df['text'])
    end = time.time()
    logging.info(f"Validation took {end - start} seconds for {len(val_df)} validation data.")

    report = classification_report(val_df['label'], y_pred, target_names=le.classes_)
    f1 = f1_score(val_df['label'], y_pred)
    # save to "val_results.txt"
    with open(os.path.join(args.model_path, 'val_results.txt'), 'w') as f:
        f.write("Classification Report\n\n")
        f.write(report)
        f.write("\nAveraged F1 Score: {}\n".format(f1))
        f.write("\nInference time: {} seconds for {} samples".format(end - start, len(val_df)))

if args.do_predict:
    test_df = pd.read_csv(args.dev)
    sample_data = test_df.sample(1, random_state=42)
    sample_text = sample_data['text_a'].values[0]
    sample_label = sample_data['label'].values[0]

    test_df['text'] = preprocess(test_df['text_a'])
    test_df['label'] = le.transform(test_df['label'])
    logging.info('Predicting...')
    logging.info(f"Sample text: {sample_text} | Sample label: {sample_label}")
    start = time.time()
    y_pred = model.predict(test_df['text'])
    end = time.time()
    logging.info(f"Prediction took {end - start} seconds for {len(test_df)} test data.")

    report = classification_report(test_df['label'], y_pred, target_names=le.classes_)
    f1 = f1_score(test_df['label'], y_pred)
    # save to "test_results.txt"
    with open(os.path.join(args.model_path, 'test_results.txt'), 'w') as f:
        f.write("Classification Report\n\n")
        f.write(report)
        f.write("\nAveraged F1 Score: {}\n".format(f1))
        f.write("\nInference time: {} seconds for {} samples".format(end - start, len(test_df)))