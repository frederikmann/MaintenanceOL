import requests as req
from bs4 import BeautifulSoup
from tqdm import tqdm

from .oie import get_terms
from .cleaning import terms as clean_terms


def get_links(topic, domain, limit):
    # get links from topic pages - give topic link and specify domain and limit of pages to be scraped

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
    # get text elements from page and return combined string - give link of page and specify domain

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

    return [x.strip() for x in paragraphs]


def get_corpus(topic, domain, limit, export):
    # get text from several links listed in topic page and return list string elements (one entry per link) - give
    # topic link and specify domain + limit of link pages to be scraped, also wether text should be exported

    corpus = []
    path = "resources/txt/"
    counter = 0
    links = get_links(topic, domain, limit)
    for link in tqdm(links):
        if get_text(link, domain):
            text = " ; ".join(get_text(link, domain))
            if len(text) < 100000 and text:
                corpus.append(text)
                if export:
                    with open(path+domain+"/"+str(counter)+".txt", "w", encoding="utf-8") as file:
                        file.write(text)
                    counter += 1
    return corpus


def load_domain_terms(domain, limit, clean=0):
    # load previously exported text for each domain - give domain and limit of text files to load

    counter = 0
    path = "resources/txt/"
    corpus = []
    terms = []
    while counter < limit:
        try:
            with open(path + domain + "/" + str(counter) + ".txt", "r") as file:
                corpus.append(file.read())
            counter += 1
        except UnboundLocalError:
            print("unicode error")
            counter += 1

    for doc in tqdm(corpus):
        doc_terms = get_terms(doc)
        terms.append(doc_terms)
    if clean:
        terms = clean_terms(terms)

    return terms
