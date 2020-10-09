import spacy
import string
from spacy.pipeline import Sentencizer

nlp = spacy.load("de_core_news_md")
sentencizer = Sentencizer(punct_chars=[char for char in string.punctuation])
nlp.add_pipe(sentencizer, name="sentence_segmenter", before="parser")


def get_oie(corpus):
    # decision logic for extracting roots and terms - for better analysis sentences are passed as well

    roots = []
    terms = []
    sents = []

    doc = nlp(corpus.lower())

    for sent in doc.sents:
        t = set()
        # get sentences
        sents.append(sent.text)

        # get important tokens from sentence
        pd, oc, ng = "", "", ""
        for token in sent:
            if token.dep_ == "pd":
                pd = token.lemma_
            if token.dep_ == "oc":
                oc = token.lemma_
            if token.dep_ == "ng" and token.head.dep_ == "ROOT":
                ng = token.lemma_
            if token.pos_ == "NOUN":
                t.add(token.text)
            if token.pos_ == "PROPN":
                t.add(token.text)

        for chunk in sent.noun_chunks:
            c = []
            for token in chunk:
                if not token.is_stop and not token.pos_ == "DET":
                    c.append(token.text)
            if len(c)>1:
                t.add(" ".join(c))

        # get roots / predicate depending on sentence structure
        r = []
        if ng:
            r.append(ng)
        if sent.root.pos_ == "AUX" and pd:
            r.append(pd)
        if sent.root.pos_ == "AUX" and oc:
            r.append(oc)
        else:
            r.append(sent.root.lemma_)

        roots.append(' '.join(r))
        terms.append(t)

    return roots, terms, sents


def get_terms(corpus):

    terms = []

    doc = nlp(corpus.lower())

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            terms.append(token.lemma_)

    for chunk in doc.noun_chunks:
        c = []
        for token in chunk:
            if not token.is_stop and not token.pos_ == "DET":
                c.append(token.text)
        if len(c) > 1:
            terms.append(" ".join(c))

    return terms
