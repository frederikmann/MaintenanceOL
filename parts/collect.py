import spacy
import pandas as pd
import requests as req
from bs4 import BeautifulSoup
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

nlp = spacy.load("de_core_news_md")


def get_links_car(topic):
    r = req.get(topic)
    soup = BeautifulSoup(r.text, "html.parser")
    h = []

    print("first page car")

    ul = soup.find("ul", {"class": "_2DRQIffNCzdlprT-xoye1"})

    h += [link["href"] for link in ul.find_all("a", href=True)]

    while soup.find("a", rel="next", href=True):
        print("next Page")
        r = req.get("https://www.motor-talk.de" + soup.find("a", rel="next", href=True)["href"])
        soup = BeautifulSoup(r.text, 'html.parser')
        ul = soup.find("ul", {"class": "_2DRQIffNCzdlprT-xoye1"})
        h += [link["href"] for link in ul.find_all("a", href=True)]
        break

    return h


def get_text_car(link):
    r = req.get(link)
    soup = BeautifulSoup(r.text, "html.parser")
    p = []

    for div in soup.find_all("div", {"itemprop": "text"}):
        p += [p.text for p in div.find_all("p")]

    while soup.find("a", rel="next", href=True):
        r = req.get("https://www.motor-talk.de" + soup.find("a", rel="next", href=True)["href"])
        soup = BeautifulSoup(r.text, 'html.parser')
        for div in soup.find_all("section", {"itemprop": "comment"}):
            p += [p.text for p in div.find_all("p")]

    return ' '.join(p)


def get_text_cook(link):
    r = req.get(link)
    soup = BeautifulSoup(r.text, "html.parser")
    p = []

    p += [div.text for div in soup.find_all("div", {"class": "forum-message-content"})]

    while soup.find("a", {
        "class": "ck-pagination__link ck-pagination__link-prevnext ck-pagination__link-prevnext--next qa-pagination-next"},
                    href=True):
        print("next")
        r = req.get(link + soup.find("a", {
            "class": "ck-pagination__link ck-pagination__link-prevnext ck-pagination__link-prevnext--next qa-pagination-next"},
                                     href=True)["href"])
        soup = BeautifulSoup(r.text, 'html.parser')
        p += [div.text for div in soup.find_all("div", {"class": "forum-message-content"})]

    return ' '.join(p)