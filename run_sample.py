import spacy
import pandas as pd
import requests as req
from bs4 import BeautifulSoup
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

from parts import collect, cleaning

print("loading spacy")
nlp = spacy.load("de_core_news_md")
print("complete")


def run_sample():
    # Collect sample forum post
    print("collecting forum post")
    car_link = "https://www.motor-talk.de/forum/audi-80-90-100-200-v8-b158.html?page=1"

    car = collect.get_links_car(car_link)
    print(len(car))

    # normalizing, cleaning and pre-processing the corpus
    print("preprocessing")
    normalized = cleaning.normalize(car, "true", "false")
    cleaned = cleaning.clean(normalized, "true", "NOUN")

    print(cleaned)


if __name__ == "__main__":
    run_sample()
