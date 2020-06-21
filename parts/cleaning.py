import spacy
import string
import re
from collections import Counter
from spacy.pipeline import Sentencizer
from spacy.matcher import PhraseMatcher

nlp = spacy.load("de_core_news_md")
sentencizer = Sentencizer(punct_chars=[".", "?", "!", ",", ";", ":"])
nlp.add_pipe(sentencizer, name="sentence_segmenter", before="parser")


def get_abbr():
    # load list of german abbreviations for normalizing
    # taken from https://de.wiktionary.org/wiki/Kategorie:Abk%C3%BCrzung_(Deutsch)

    with open("resources/abbreviations_ger.txt", "r", encoding="utf-8") as f:
        x = f.readlines()
    abbreviations = [item.rstrip('\n') for item in x]

    return abbreviations


def sentences(corpus, no_questions):
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


def uniques(terms, label = 0):
    clean_terms = []
    labels = []
    counter = 0
    tf = Counter([item for sublist in terms for item in sublist])
    for doc in terms:
        clean_doc = []
        doc_labels = {}
        for term in doc:
            doc_labels[term] = 0
            if tf[term] > 1:
                clean_doc.append(term)
                doc_labels[term] = 1
            else:
                counter += 1
        clean_terms.append(clean_doc)
        labels.append(doc_labels)
    print(counter, "terms removed due to uniques")
    if label:
        return labels
    else:
        return clean_terms


def terms(terms, label = 0):
    # clean zitate, datum, zeichen/ zahlen only terms, zeit, links
    clean_terms = []
    labels = []
    time, date, link, zitat, ireg, abbr = 0, 0, 0, 0, 0, 0
    abbreviations = get_abbr()

    for doc in terms:
        clean_doc = []
        doc_labels = {}
        for term in doc:
            term = term.strip()
            doc_labels[term] = 0
            if len(term) < 2:
                pass
            elif re.search(r"\d\d:\d\d:\d\d", term):
                time += 1
            elif re.search(r"\d\d.\d\d.\d\d\d\d", term):
                date += 1
            elif re.search("www", term):
                link += 1
            elif re.search("@", term):
                zitat += 1
            elif term in abbreviations:
                abbr += 1
            elif re.search(r"\w", term):
                clean_doc.append(term)
                doc_labels[term] = 1
            elif re.search(r"^\W", term):
                ireg += 1
            else:
                print(term)
                clean_doc.append(term)
                doc_labels[term] = 1

        clean_terms.append(clean_doc)
        labels.append(doc_labels)

    print("deleted time references:", time)
    print("deleted date references:", date)
    print("deleted links:", link)
    print("deleted quotes:", zitat)
    print("deleted ireg expressions:", ireg)
    print("deleted abbreviations:", abbr)

    if label:
        return labels
    else:
        return clean_terms
