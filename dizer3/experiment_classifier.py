import logging
import pickle
import re

from ufal._udpipe import Model, Pipeline

# model = Model.load('https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131/portuguese-bosque-ud-2.5-191206.udpipe?sequence=75&isAllowed=y')
model = Model.load('models/portuguese-bosque-ud-2.5-191206.udpipe')
pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT,
                    Pipeline.DEFAULT, 'conllu')

def prepare_syn_annotation(sentence_annotation):
    annotation = []

    lines = sentence_annotation.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match("^\d+", line):
            parts = line.split("\t")
            try:
                if parts[3] == '_':
                    token_plus = lines[i + 1].split("\t")
                    token_plus_plus = lines[i + 2].split("\t")
                    if token_plus[3] == 'ADP' and token_plus_plus[3] == 'DET':
                        parts[3] = 'ADP+DET'
                        parts[5] = token_plus_plus[5]
                        parts[6] = token_plus[6]
                        parts[7] = token_plus[7]
                        i += 2
            except:
                logging.error("Error preparing syntax annotation")
            token = {'token': parts[1],
                     'lemma': parts[2],
                     'pos': parts[3],
                     'morpho': parts[5],
                     'head': parts[6],
                     'dep': parts[7]}
            annotation.append(token)
        i += 1

    return annotation


def nlu(sentence):
  ann = pipeline.process(sentence)
  return prepare_syn_annotation(ann)


with open('sentences.pth', 'rb') as file:
    sentences = pickle.load(file)

for segments in sentences:
  sentence = ' '.join(segments)
  syntax = nlu(sentence)
  breaks = [0] * len(sentence.split())
  latest_break = 0
  for idx, segment in enumerate(segments):
    if idx == 0:
      continue
    latest_break += len(segment.split())
    breaks[latest_break] = 1

  print(sentence)
  print(sentences)
  print(syntax)
  print(breaks)
  break