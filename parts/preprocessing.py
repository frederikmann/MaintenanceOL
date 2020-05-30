import spacy
import string
from spacy.pipeline import Sentencizer
from spacy.matcher import PhraseMatcher

nlp = spacy.load("de_core_news_md")
sentencizer = Sentencizer(punct_chars=[".", "?", "!", ",", ";"])
nlp.add_pipe(sentencizer, name="sentence_segmenter", before="parser")


def get_abbr():
    # load list of german abbreviations for normalizing
    # taken from https://de.wiktionary.org/wiki/Kategorie:Abk%C3%BCrzung_(Deutsch)

    with open("resources/abbreviations_ger.txt", "r") as f:
        x = f.readlines()
    abbreviations = [item.rstrip('\n') for item in x]

    return abbreviations


def normalize(corpus, lowercase, remove_stopwords):
    abbreviations = get_abbr()

    if lowercase:
        corpus = corpus.lower()
    doc = nlp(corpus)
    normalized = list()
    for word in doc:
        if not remove_stopwords or (remove_stopwords and not word.is_stop):
            if word.text not in abbreviations:
                normalized.append(word.text)
    return " ".join(normalized)


def clean(corpus, no_questions):
    # potentially add root form support
    terms = ["wohin", "wie", "woher", "was", "wieso", "warum", "wer", "welche", "wen", "wem", "wo", "?"]
    matcher = PhraseMatcher(nlp.vocab)

    patterns = [nlp.make_doc(text) for text in terms]
    matcher.add("QuestionList", None, *patterns)

    doc = nlp(corpus)
    sents = []

    for sent in doc.sents:
        doc = nlp(sent.text)
        matches = matcher(doc)
        if sent.root.pos_ and len(sent.text) > 3:
            if not no_questions or (no_questions and not matches):
                sents.append(sent.text.translate(str.maketrans('', '', string.punctuation)))

    return "; ".join(sents)

