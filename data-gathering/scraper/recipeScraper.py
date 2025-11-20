import requests
from bs4 import BeautifulSoup
from scraperABC import Scraper
from pageinfo import PageInfo

def normalize_ws(string: str):
    string = string.replace('\n', '')
    string = string.replace('\t', '') 
    return string

class RecipeScraper(Scraper):
    def __init__(self):
        pass

    def info(self, soup: BeautifulSoup):
        ingredients = self.ingredients(soup)        # get ingr. and instrc.
        paragraphs = self.paragraphs(soup)
        page_info = f"{ingredients}\n{paragraphs}\n"    # concatenate
        return page_info

    def ingredients(self, soup: BeautifulSoup):
        ingredients = []

        # Optionally narrow it down to the ingredients container(s)
        for block in soup.select("div.gz-ingredients.gz-outer, dl.gz-list-ingredients"):
            for dd in block.select("dd.gz-ingredient"):
                name_tag = dd.find("a")                 # get name
                name = name_tag.get_text(strip=True) if name_tag else None

                span = dd.find("span")                  # get quantity + info
                if span:
                    quantity = span.get_text(" ", strip=True)
                else:
                    quantity = ""

                if name:
                    ingredients.append((name, quantity))
        
        unique_ingredients = self.remove_dup(ingredients)   # clean it

        return self.tup2txt(unique_ingredients)

    def tup2txt(self, li: list) -> str:
        text = ""
        for name, quantity in li:               # add ingredients in format
            quantity = normalize_ws(quantity)
            text += f"{name}: {quantity}\n"
        return text

    def paragraphs(self, soup: BeautifulSoup) -> str:
        text_tags = soup.select('p')        # filter paragraphs and join texts
        paragraphs = [t.get_text(" ", strip=True) for t in text_tags]
        joined_paragraphs = '\n'.join(paragraphs)
        return joined_paragraphs

    def title(self, soup: BeautifulSoup) -> str:
        title = soup.select_one('h1').get_text(strip=True)
        return title

    def remove_dup(self, some_list: list):
        return list(dict.fromkeys(some_list))

scraper = RecipeScraper()

sc = scraper.scrape("https://www.giallozafferano.com/recipes/baked-pasta-with-zucchini-cream.html")

print(sc)
