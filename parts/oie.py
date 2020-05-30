import spacy
from spacy.pipeline import Sentencizer

nlp = spacy.load("de_core_news_md")
sentencizer = Sentencizer(punct_chars=[".", "?", "!", ",", ";", ":"])
nlp.add_pipe(sentencizer, name="sentence_segmenter", before="parser")


def get_oie(corpus, just_terms=None):
    # decision logic for extracting roots and terms - for better analysis sentences are passed as well

    roots = []
    terms = []
    sents = []

    doc = nlp(corpus)

    for sent in doc.sents:
        t = set()
        # get sentences
        sents.append(sent.text)

        # get important tokens from sentence
        pd, oc, ng = "", "", ""
        for token in sent:
            if token.dep_ == "sb":
                t.add(token.lemma_)
            if token.dep_ == "pd":
                pd = token.lemma_
            if token.dep_ == "oc":
                oc = token.lemma_
            if token.dep_ == "ng" and token.head.dep_ == "ROOT":
                ng = token.lemma_
            if token.pos_ == "NOUN":
                t.add(token.text)

        # get noun chunks and remove stopwords
        for chunk in sent.noun_chunks:
            c = []
            for token in chunk:
                if not token.is_stop:
                    c.append(token.text)
            t.add(" ".join(c))

        # get entities and remove stopwords
        for ent in sent.ents:
            e = []
            for token in ent:
                if not token.is_stop:
                    e.append(token.text)
            t.add(" ".join(e))

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

    if just_terms:
        return [item for sublist in terms for item in sublist]
    else:
        return roots, terms, sents

