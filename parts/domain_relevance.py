import numpy as np
import math
from collections import Counter
from tqdm import tqdm

from .collect import get_text
from .oie import get_oie


'''
definition of all metrics needed for evaluation of domain relevancy
'''


def get_tf(terms, norm=0):
    flat_terms = [item for sublist in terms for item in sublist]
    tf = Counter(flat_terms)
    max_freq = Counter(flat_terms).most_common(1)[0][1]
    for t in tf:
        if norm:
            tf[t] = (tf[t] / max_freq)
        else:
            pass

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


def get_lor(target_domain, contrastive_domain, candidates):
    # candidates should be part of the target domain
    lor = {}
    target_tf = get_tf(target_domain)
    contrastive_tf = get_tf(contrastive_domain)

    for term in candidates:
        target_tf[term] = 0.0001 if not target_tf[term] else target_tf[term]
        contrastive_tf[term] = 0.0001 if not contrastive_tf[term] else contrastive_tf[term]

        lor[term] = np.log2(target_tf[term] / (1 - target_tf[term])) - np.log2(
            contrastive_tf[term] / (1 - contrastive_tf[term]))

    return lor


def get_lor_bg(target_domain, contrastive_domain, candidates, normalized=0):
    # candidates should be part of the target domain
    lor_bg = {}

    # get term frequency for each domain - combining both gives background knowledge
    target_flat_terms = [item for sublist in target_domain for item in sublist]
    target_tf = Counter(target_flat_terms)

    contrastive_flat_terms = [item for sublist in contrastive_domain for item in sublist]
    contrastive_tf = Counter(contrastive_flat_terms)

    combine_flat_terms = target_flat_terms + contrastive_flat_terms
    combine_tf = Counter(combine_flat_terms)

    n_i = len(target_flat_terms)
    n_j = len(contrastive_flat_terms)
    a_0 = len(combine_flat_terms)

    for term in candidates:
        target_tf[term] = 0.1 if not target_tf[term] else target_tf[term]
        contrastive_tf[term] = 0.1 if not contrastive_tf[term] else contrastive_tf[term]
        combine_tf[term] = 0.1 if not combine_tf[term] else contrastive_tf[term]

        lor_bg[term] = np.log2(
            (target_tf[term] + combine_tf[term]) / n_i + a_0 - (target_tf[term] + combine_tf[term])) - np.log2(
            (contrastive_tf[term] + combine_tf[term]) / n_i + a_0 - (contrastive_tf[term] + combine_tf[term]))
        if normalized:
            sigma = 1 / (target_tf[term] + combine_tf[term]) + 1 / (contrastive_tf[term] + combine_tf[term])
            lor_bg[term] = lor_bg[term] / math.sqrt(sigma)
    return lor_bg


def get_relevance(terms, metric):
    tf = get_tf(terms)
    idf = get_idf(terms)
    tdf = get_tdf(terms)
    tf_tdf = {}
    tf_idf = {}

    if metric == "tf":
        return tf
    elif metric == "idf":
        return idf
    elif metric == "tdf":
        return tdf
    elif metric == "tf_tdf":
        for term in set([item for sublist in terms for item in sublist]):
            tf_tdf[term] = tf[term] * tdf[term]
        return tf_tdf
    elif metric == "tf_idf":
        for term in set([item for sublist in terms for item in sublist]):
            tf_idf[term] = tf[term] * idf[term]
        return tf_idf
    else:
        return False


'''
definition of all functions needed to run the domain relevancy evaluation
'''


def get_shared_domain(domain_a, domain_b):
    shared_domain_a = []
    shared_domain_b = []

    terms_a = set([item for sublist in domain_a for item in sublist])
    terms_b = set([item for sublist in domain_b for item in sublist])

    for docs in tqdm(domain_a):
        doc = []
        for term in docs:
            if term in terms_b:
                doc.append(term)
        shared_domain_a.append(doc)

    for docs in tqdm(domain_b):
        doc = []
        for term in docs:
            if term in terms_a:
                doc.append(term)
        shared_domain_b.append(doc)

    return shared_domain_a, shared_domain_b


def get_metrics(target_domain, contrastive_domain, metric):
    target_relevance = get_relevance(target_domain, metric)
    contrast_relevance = get_relevance(contrastive_domain, metric)

    alpha = 0.5
    candidates = set([item for sublist in target_domain for item in sublist])
    dw = get_dw(target_domain, contrastive_domain, candidates, alpha)

    llr = get_llr(target_domain, contrastive_domain, candidates)

    lor_bg = get_lor_bg(target_domain, contrastive_domain, candidates)

    return target_relevance, contrast_relevance, dw, llr, lor_bg


def label_shared_concepts(target_domain, contrastive_domain, method, threshold, metric):
    label = {}
    shared_target_domain, shared_contrastive_domain = get_shared_domain(target_domain, contrastive_domain)
    tf_target, tf_contrast, dw, llr, lor_bg = get_metrics(shared_target_domain, shared_contrastive_domain, metric)

    candidates = set([item for sublist in shared_target_domain for item in sublist])

    for candidate in candidates:
        label[candidate] = 0
        if tf_target[candidate] > tf_contrast[candidate] and not method:
            label[candidate] = 1
        elif llr[candidate] > threshold and method == "llr":
            label[candidate] = 1
        elif lor_bg[candidate] > threshold and method == "lor_bg":
            label[candidate] = 1
        elif dw[candidate] > threshold and method == "dw":
            label[candidate] = 1
        elif method == "del":
            pass

    return label


def label_concepts(target_domain, background_domain, contrastive_domain, method=0, threshold = 0, metric="tf"):
    label = {}

    target_tf = get_tf(target_domain)
    candidates = set([item for sublist in target_domain for item in sublist])

    background_relevance = get_relevance(background_domain, metric)
    contrastive_relevance = get_relevance(contrastive_domain, metric)
    shared_concept_labels = label_shared_concepts(target_domain, contrastive_domain, method, threshold, metric)

    counter1 = 0
    counter2 = 0
    counter3 = 0

    for candidate in candidates:
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


def main():
    domain_relevance = {}

    return domain_relevance
