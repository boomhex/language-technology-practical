import requests
from bs4 import BeautifulSoup

def normalize_ws(string: str):
    string = string.replace('\n', '')
    string = string.replace('\t', '') 
    return string

class RecipeScraper:
    def __init__(self):
        self.scraped_page = ""

    def scrape(self, url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()

        page = self.page_info(response.text)

        return page, url

    def page_info(self, page_html):
        soup = BeautifulSoup(page_html, "html.parser")  # parse html

        title = self.title(soup)                        # extract relevant info
        ingredients = self.ingredients(soup)
        paragraphs = self.paragraphs(soup)

        # concatenate the info as text
        page = f"{title}\n\n"\
               f"{ingredients}\n"\
               f"{paragraphs}\n"

        return page

    def ingredients(self, soup):
        ingredients = []

        # Optionally narrow it down to the ingredients container(s)
        for block in soup.select("div.gz-ingredients.gz-outer, dl.gz-list-ingredients"):
            # Find all ingredient lines in this block
            for dd in block.select("dd.gz-ingredient"):
                # Name is in the <a> tag
                name_tag = dd.find("a")
                name = name_tag.get_text(strip=True) if name_tag else None

                # Quantity + notes are inside the <span>
                span = dd.find("span")
                if span:
                    # get_text collapses inner tags like <i>
                    quantity = span.get_text(" ", strip=True)
                else:
                    quantity = ""

                if name:  # only keep valid rows
                    ingredients.append((name, quantity))
        # If the page repeats the same ingredient section several times,
        # you can deduplicate them:
        unique_ingredients = self.remove_dup(ingredients)
        return self.tup2txt(unique_ingredients)

    def tup2txt(self, li: list) -> str:
        text = ""
        for name, quantity in li:   # add each name and quantity in nice format
            text += f"{name}: {normalize_ws(quantity)}\n"

        return text

    def paragraphs(self, soup: BeautifulSoup) -> str:
        text_tags = soup.select('p')    # filter paragraphs and join texts
        paragraphs = '\n'.join([t.get_text(" ", strip=True) for t in text_tags])
        return paragraphs

    def title(self, soup: BeautifulSoup) -> str:
        title = soup.select_one('h1').get_text(strip=True)
        return title

    def remove_dup(self, some_list: list):
        return list(dict.fromkeys(some_list))

scraper = RecipeScraper()

sc = scraper.scrape("https://www.giallozafferano.com/recipes/baked-pasta-with-zucchini-cream.html")

with open("test", 'w') as f:
    f.write(sc)
