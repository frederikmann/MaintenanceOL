import spacy
import requests as req
from bs4 import BeautifulSoup
from tqdm import tqdm

nlp = spacy.load("de_core_news_md")


def get_links(topic, domain, limit):
    counter = 0
    links = []

    if domain == "car":
        r = req.get(topic)
        soup = BeautifulSoup(r.text, "html.parser")
        ul = soup.find("ul", {"class": "_2DRQIffNCzdlprT-xoye1"})

        links.extend(["https://www.motor-talk.de"+link["href"] for link in ul.find_all("a", href=True)])

        while soup.find("a", rel="prev", href=True) and counter < limit:
            r = req.get("https://www.motor-talk.de" + soup.find("a", rel="prev", href=True)["href"])
            soup = BeautifulSoup(r.text, 'html.parser')
            ul = soup.find("ul", {"class": "_2DRQIffNCzdlprT-xoye1"})
            links.extend(["https://www.motor-talk.de"+link["href"] for link in ul.find_all("a", href=True)])
            counter += 1
        links.pop(0)

    elif domain == "adac":
        link = "https://www.adac.de/infotestrat/reparatur-pflege-und-wartung/rueckrufe/suchergebnis.aspx?Kategorie=Pkw"
        make_list = []

        r = req.get(link)
        soup = BeautifulSoup(r.text, "html.parser")
        makes = soup.find("select", {"class": "w190"})
        for option in makes.find_all("option"):
            make_list.append(option.text)

        make_list.remove("Alle Hersteller")
        make_list.remove("")

        for make in tqdm(make_list):
            make_link = link + "&Hersteller=" + make
            r = req.get(make_link)
            soup = BeautifulSoup(r.text, "html.parser")
            models = soup.find("select", {"class": "w190 left"})
            if models:
                for option in models.find_all("option"):
                    links.append(make_link + "&Modelle=" + option.text)

    elif domain == "chefkoch":
        r = req.get(topic)
        soup = BeautifulSoup(r.text, "html.parser")

        for link in soup.find_all("a", {"class": "search-result-title"}):
            links.append("https://www.chefkoch.de" + link["href"])

        while soup.find("a", {"class": "pagination-item pagination-next"}) and counter < limit:
            page = "https://www.chefkoch.de" + soup.find("a", {"class": "pagination-item pagination-next"})["href"]
            r = req.get(page)
            soup = BeautifulSoup(r.text, "html.parser")
            for link in soup.find_all("a", {"class": "search-result-title"}):
                links.append("https://www.chefkoch.de" + link["href"])
            counter += 1

    else:
        print("choose domain from: car, adac or chefkoch")

    return links


def get_text(link, domain):
    counter = 0
    try:
        r = req.get(link)
        soup = BeautifulSoup(r.text, "html.parser")
    except req.exceptions.ConnectionError as e:
        return False

    paragraphs = []

    if domain == "car":
        for div in soup.find_all("div", {"itemprop": "text"}):
            paragraphs += [p.text for p in div.find_all("p")]

        while soup.find("a", rel="next", href=True) and counter < 10:
            r = req.get("https://www.motor-talk.de" + soup.find("a", rel="next", href=True)["href"])
            soup = BeautifulSoup(r.text, 'html.parser')
            for div in soup.find_all("section", {"itemprop": "comment"}):
                paragraphs += [p.text for p in div.find_all("p")]
            counter += 1

    elif domain == "adac":
        paragraphs = set()

        r = req.get(link)
        soup = BeautifulSoup(r.text, "html.parser")

        for p in soup.find_all("p", {"class": "pl13"}):
            paragraphs.add(p.text)

    elif domain == "chefkoch":
        paragraphs += [div.text for div in soup.find_all("div", {"class": "forum-message-content"})]

        while soup.find("a",
                        {
                            "class": "ck-pagination__link ck-pagination__link-prevnext ck-pagination__link-prevnext--next qa-pagination-next"},
                        href=True) and counter < 10:
            r = req.get(link + soup.find("a", {
                "class": "ck-pagination__link ck-pagination__link-prevnext ck-pagination__link-prevnext--next qa-pagination-next"},
                                         href=True)["href"])
            soup = BeautifulSoup(r.text, 'html.parser')
            paragraphs += [div.text for div in soup.find_all("div", {"class": "forum-message-content"})]
            counter += 1

    return paragraphs
