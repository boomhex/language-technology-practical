import requests
from bs4 import BeautifulSoup
from .scraperABC import Scraper
from .pageinfo import PageInfo
from typing import List

def normalize_ws(string: str):
    string = string.replace('\n', '')
    string = string.replace('\t', '') 
    return string

def concat_info(info):
    page_info = ""
    for (label, item) in info:
        page_info += f"{label}\n{item}\n" 
    return page_info

class RecipeScraper(Scraper):
    def __init__(self):
        pass

    def info(self, soup: BeautifulSoup):
        ingredients = self.ingredients(soup)        # get ingr. and instrc.
        paragraphs = self.paragraphs(soup)
        recipe_metainfo = self.metainfo(soup)

        info = []
        info.append(("ingredients:", ingredients))
        info.append(("info:", recipe_metainfo))
        info.append(("instructions:", paragraphs))

        page_info = concat_info(info)

        return page_info

    def metainfo(self, soup: BeautifulSoup):
        info = ""
        # Select all summary rows
        for item in soup.select("span.gz-name-featured-data"):
            text = item.get_text(" ", strip=True)
            info += text + '\n'
        return info

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
        # 1. Remove the "You might also like" section if we can identify it
        #    by heading or visible text.
        for node in soup.find_all(string=lambda s: s and "you might also like" in s.lower()):
            container = node.parent
            # Walk up a bit and remove a reasonably-sized block
            for _ in range(3):                  # go up at most 3 levels
                if container is None:
                    break
                # Heuristic: likely wrappers for that block
                if container.name in {"section", "aside", "div"}:
                    container.decompose()
                    break
                container = container.parent

        # 2. Remove numbered step spans as you already do
        for span in soup.find_all("span", class_="num-step"):
            span.decompose()

        # 3. Collect <p> tags, but stop if we hit a "You might also like" paragraph
        text_tags = soup.select("p")

        paragraphs: List[str] = []
        for p in text_tags:
            txt = p.get_text(" ", strip=True)
            if txt.lower().startswith("you might also like"):
                break
            paragraphs.append(txt)

        joined_paragraphs = "\n".join(paragraphs)
        return joined_paragraphs

    def title(self, soup: BeautifulSoup) -> str:
        title = soup.select_one('h1').get_text(strip=True)
        return title

    def remove_dup(self, some_list: list):
        return list(dict.fromkeys(some_list))
