import spacy
import numpy as np
from spacy.pipeline import Sentencizer
from collections import Counter

from .collect import get_text
from .oie import get_oie

nlp = spacy.load("de_core_news_md")
sentencizer = Sentencizer(punct_chars=[".", "?", "!", ",", ";", ":"])
nlp.add_pipe(sentencizer, name="sentence_segmenter", before="parser")


def get_tf(terms):
    flat_terms = [item for sublist in terms for item in sublist]
    tf = Counter(flat_terms)
    max_freq = Counter(flat_terms).most_common(1)[0][1]
    for t in tf:
        tf[t] = (tf[t] / max_freq)

    return tf


def get_idf(terms):
    flat_terms = [item for sublist in terms for item in set(sublist)]
    idf = Counter(flat_terms)
    for t in idf:
        idf[t] = np.log2(len(terms) / idf[t])

    return idf


def get_tdf(terms):
    flat_terms = [item for sublist in terms for item in set(sublist)]
    tdf = Counter(flat_terms)
    max_freq = Counter(flat_terms).most_common(1)[0][1]
    for t in tdf:
        tdf[t] = tdf[t] / max_freq

    return tdf


def get_tf_idf(terms):
    tf_idf = {}
    tf = get_tf(terms)
    idf = get_idf(terms)

    for term in set([item for sublist in terms for item in sublist]):
        tf_idf[term] = tf[term] * idf[term]

    return tf_idf


def get_dr(target_domain, contrastive_domain, candidates):
    # candidates should be part of the target domain
    dr = {}
    target_tf = get_tf(target_domain)
    contrastive_tf = get_tf(contrastive_domain)

    for term in candidates:
        try:
            dr[term] = target_tf[term] / contrastive_tf[term]
        except ZeroDivisionError:
            dr[term] = 0

    return dr


def get_dc(target_domain, candidates):
    dc = {}
    target_tdf = get_tdf(target_domain)

    for term in candidates:
        dc[term] = target_tdf[term]*np.log2(1/target_tdf[term])

    return dc


def get_dw(target_domain, contrastive_domain, candidates, alpha):
    dw = {}
    dr = get_dr(target_domain, contrastive_domain, candidates)
    dc = get_dc(target_domain, candidates)

    for term in candidates:
        dw[term] = alpha * dr[term] + (1 - alpha) * dc[term]

    return dw


def get_llr(target_domain, contrastive_domain, candidates):
    # candidates should be part of the target domain
    llr = {}
    target_tf = get_tf(target_domain)
    contrastive_tf = get_tf(contrastive_domain)

    for term in candidates:
        target_tf[term] = 0.1 if not target_tf[term] else target_tf[term]
        contrastive_tf[term] = 0.1 if not contrastive_tf[term] else contrastive_tf[term]

        llr[term] = np.log(target_tf[term]) - np.log(contrastive_tf[term])

    return llr


def main(target_link, target_terms, domain_relevance_measure):
    domain_relevance = {}

    # get terms from both domains for each link
    car_links.append(target_link)
    car_terms = get_car_terms(car_links)
    cook_terms = get_cook_terms(cook_links)

    flat_car_terms = [item for sublist in car_terms for item in sublist]
    flat_cook_terms = [item for sublist in cook_terms for item in sublist]

    candidates = set([item for sublist in target_terms for item in sublist])
    target_domain = get_words(flat_car_terms)
    contrastive_domain = get_words(flat_cook_terms)

    if domain_relevance_measure == "DR_DC":
        for candidate in candidates:
            dr = get_dr(target_domain, contrastive_domain, candidate)
            dc = get_dc(car_terms, candidate)
            domain_relevance[candidate] = get_dw(dr, dc, 0.5)

    if domain_relevance_measure == "LOR":
        # IMPLEMENT LOR
        for candidate in candidates:
            p_i = get_mean(car_terms, candidate)
            p_j = get_mean(cook_terms, candidate)
            domain_relevance[candidate] = get_log(p_i) - get_log(p_j)

    return domain_relevance
