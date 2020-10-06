import spacy
import numpy as np
from spacy.pipeline import Sentencizer

from .collect import get_text_car, get_text_cook
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


def get_cook_terms(links):
    # getting cooking terms for domain relevance
    cook_terms = []
    for link in links:
        cook = []
        cook.extend(get_text_cook(link))
        cook_corpus = "; ".join(cook)
        cook_norm = normalize(cook_corpus, 1, 0)
        terms = get_oie(cook_norm, 1)
        cook_terms.append(terms)

    return cook_terms


def get_car_terms(links):
    # getting car terms for domain relevance per link
    car_terms = []
    for link in links:
        car = []
        car.extend(get_text_car(link))
        car_corpus = "; ".join(car)
        car_norm = normalize(car_corpus, 1, 0)
        terms = get_oie(car_norm, 1)
        car_terms.append(terms)

    return car_terms


def main(target_link, target_terms):
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

    for candidate in candidates:
<<<<<<< Updated upstream
        dr = get_dr(target_domain, contrastive_domain, candidate)
        dc = get_dc(car_terms, candidate)
        domain_relevance[candidate] = get_dw(dr, dc, 0.5)

    return domain_relevance
=======
        label[candidate] = 0
        try:
            background_relevance[candidate]
        except KeyError:
            background_relevance[candidate] = 0

        try:
            contrastive_relevance[candidate]
        except KeyError:
            contrastive_relevance[candidate] = 0

        if background_relevance[candidate] > contrastive_relevance[candidate]:
            label[candidate] = 1
            counter1 += 1
        elif contrastive_relevance[candidate]:
            if shared_concept_labels[candidate]:
                label[candidate] = 1
                counter2 += 1
        elif target_tf[candidate] > 1:
            label[candidate] = 1
            counter3 += 1

    print("Chosen via background domain:", counter1)
    print("Chosen via metric:", counter2)
    print("Chosen via tf > 1 limit:", counter3)
    
    return label
>>>>>>> Stashed changes
