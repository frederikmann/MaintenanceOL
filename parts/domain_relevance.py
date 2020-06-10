import spacy
import numpy as np
from spacy.pipeline import Sentencizer
from collections import Counter

from .collect import get_text
from .preprocessing import normalize
from .oie import get_oie

nlp = spacy.load("de_core_news_md")
sentencizer = Sentencizer(punct_chars=[".", "?", "!", ",", ";", ":"])
nlp.add_pipe(sentencizer, name="sentence_segmenter", before="parser")

car_links = ["https://www.motor-talk.de/forum/123d-wie-oft-batterie-laden-t6833745.html",
             "https://www.motor-talk.de/forum/fensterheber-defekt-tuerverkleidung-demontieren-t5183259.html",
             "https://www.motor-talk.de/forum/fahrzeug-hat-keine-leistung-mehr-ab-2300-touren-t6861763.html"]

cook_links = ["https://www.chefkoch.de/forum/2,27,549451/Kein-Glueck-mit-Salbei.html"
              "https://www.chefkoch.de/forum/2,53,712593/Spargel-aus-dem-Ofen.html",
              "https://www.chefkoch.de/forum/2,13,770061/Wie-schaelt-ihr-den-Spargel.html",
              "https://www.chefkoch.de/forum/2,10,770039/Tiefkuehlspinat-knirscht-fuehlt-sich-sandig-im-Mund-an.html"]


def get_words(terms):
    # get dictionary of word frequency from corpus
    words = {}
    for term in terms:
        if term not in words:
            words[term] = 1
        else:
            words[term] += 1

    return words


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
    for t in tdf:
        tdf[t] = tdf[t] / len(terms)

    return tdf


def get_tf_idf(terms):
    tf_idf = {}
    tf = get_tf(terms)
    idf = get_idf(terms)

    for term in set([item for sublist in terms for item in sublist]):
        tf_idf[term] = tf[term] * idf[term]

    return tf_idf


def get_dr(dic_a, dic_b, term):
    total_a, total_b = sum(dic_a.values()), sum(dic_b.values())

    try:
        freq_a = dic_a[term]
    except KeyError:
        return 0
    try:
        freq_b = dic_b[term]
    except KeyError:
        return 1

    return (freq_a / total_a) / (freq_b / total_b)


def get_dc(domain, term):
    dc = 0
    for link in domain:
        words = get_words(link)

        try:
            prob_d = words[term] / sum(words.values())
            dc += prob_d * np.log(1 / prob_d)
        except KeyError:
            pass

    return dc


def get_dw(dr, dc, alpha):
    return alpha * dr + (1 - alpha) * dc


def get_mean(domain, candidate):
    prob_d = 0
    for link in domain:
        words = get_words(link)

        try:
            prob_d += words[candidate] / sum(words.values())
        except KeyError:
            pass

    return prob_d/len(domain)


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
