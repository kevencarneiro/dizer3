import itertools
import logging
import os

from dizer3.syntax import nlu
from dizer3.utils import read_file, segments_file_extension, relations_file_extension


def evaluate(data_path: str, evaluation_path: str):
    file_paths = []
    if os.path.isfile(data_path):
        if os.path.isfile(data_path+segments_file_extension):
            file_paths.append(data_path+segments_file_extension)
        if os.path.isfile(data_path+relations_file_extension):
            file_paths.append(data_path+relations_file_extension)
    else:
        file_paths = [os.path.join(data_path, file) for file in os.listdir(data_path)
                      if file.endswith(relations_file_extension) or file.endswith(segments_file_extension)]

    true_negatives = 0
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    for file_path in file_paths:
        validation_file_path = os.path.join(evaluation_path, os.path.basename(file_path))
        if not os.path.isfile(validation_file_path):
            logging.warning(f"Validation file {validation_file_path} does not exists")
            continue
        file = read_file(file_path)
        validation_file = read_file(validation_file_path)

        predicted_sentences = get_tokens(file)
        actual_sentences = get_tokens(validation_file)

        predicted_tokens = list(itertools.chain.from_iterable(predicted_sentences))
        actual_tokens = list(itertools.chain.from_iterable(actual_sentences))
        for idx, (predicted_token, actual_token) in enumerate(zip(predicted_tokens, actual_tokens)):
            tokens_are_equal = predicted_token['token'] == actual_token['token']
            predicted_segment = predicted_token['segment']
            actual_segment = actual_token['segment']

            if tokens_are_equal and predicted_segment == actual_segment:
                if actual_segment == True:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if actual_segment == True:
                    false_negatives += 1
                else:
                    false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    f1_score = 2*true_positives / (2*true_positives + false_positives + false_negatives)

    metrics = {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy,
        'f1_score': f1_score
    }
    for key, value in metrics.items():
        print(f'{key}: {value}')
    return metrics


def get_tokens(file):
    file_tokens = []
    previous_paragraph = None
    previous_sentence = None
    for line in file:
        paragraph, sentence, segment, text = str.split(line, ':', 3)
        segment_tokens = nlu([text])[0]
        for idx, token in enumerate(segment_tokens):
            is_last_token = idx == (len(segment_tokens) - 1)
            token_info = {'token': token['token'], 'segment': is_last_token}
            if paragraph == previous_paragraph and sentence == previous_sentence:
                file_tokens[-1].append(token_info)
            else:
                file_tokens.append([token_info])
        previous_paragraph = paragraph
        previous_sentence = sentence
    return file_tokens
