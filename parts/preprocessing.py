import spacy
import pandas as pd
import requests as req
from bs4 import BeautifulSoup
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

def normalize(text, lowercase, remove_stopwords):
    if lowercase:
        text = text.lower()
    doc = nlp(text)
    normalized = list()
    for word in doc:
        if not remove_stopwords or (remove_stopwords and not word.is_stop):
            normalized.append(word.text)

    return " ".join(normalized)


def clean(text, no_questions, root_pos):
    terms = ["wohin", "wie", "woher", "was", "wieso", "warum", "wer", "welche", "wen", "wem", "wo", "?"]
    matcher = PhraseMatcher(nlp.vocab)

    patterns = [nlp.make_doc(text) for text in terms]
    matcher.add("QuestionList", None, *patterns)

    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        if sent.root.pos_ != root_pos:
            sentences.append(sent.string)
        if no_questions:
            matches = matcher(doc)
            if matches:
                doc.sents.remove(sent)

    return sentences
