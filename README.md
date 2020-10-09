# MaintenanceOL
Ontology Learning System for my Bachelor Thesis

The following section is taken from the thesis and explains how the different parts work together to perform ontology learning!

The first part is the collect.py, which combines all collecting tasks. It contains functions for scraping link pages of the different domains (recalls, car, cooking) and scraping the individual posts to be stored in separate text files. This is done with the help of beautiful soup like explained above. The get_corpus function is used to automatically scrape and store link pages and their individual forum posts until a limit is reached. For this stage forum posts with less than 10000 characters are skipped since they are likely to be incomplete. The last function load_domain_terms is used to read the text files previously stored and automatically extract all terms for domain relevancy analysis. Then next parts all work with the text files scraped in this part.

The next stage is the domain relevance analysis which is performed with functions from domain_relevance.py. It contains several functions implementing domain relevance measures presented in section 3.5 as well as the final decision-making mechanism. This is split up into several sub tasks: get_relevance returns the relevance of a list of terms for a given metric, get_shared_domain identifies terms which occur in both the target (car) and contrastive (cooking) domain, get_metrics returns metrics for terms in the shared domain, label_shared_concepts returns true and false labels for terms in the shared domain, finally label_concepts implements the decision tree which combines all functions introduced above and returns true or false labels for all terms based on the domain relevancy mechanism presented in section 4.2.

The third part defined two functions for open information extraction from individual posts in oie.py – which is heavily based on the spacy functions introduced above. Two different approaches are implemented: get_oie splits the text into sentences and stores terms and relations based on POS tags and dependencies identified by the spacy language model; get_terms skips the sentence segmentation and extracts terms based only on POS tags directly from the corpus which leads to a improved runtime. 

The last part contains functions to remove unwanted terms and sentences – these are collected in cleaning.py. Firstly, text files are cleaned from questions with the help of the spacy PhraseMatcher in the sentences function. After terms have been extracted the terms function deletes all unwanted terms via regular expression matching and a list of common abbreviation from Wikipedia. This way date and time references, links, quotes, abbreviations and terms consisting only of irregular expressions are removed. 

These four program parts serve as the backbone and are accessed from different jupyter notebooks, which initialize the functions described above to generate knowledge graphs, domain relevancy analysis and ontology learning.  