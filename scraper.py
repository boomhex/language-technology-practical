# scraper.py
# To run this script, paste `python3 scraper.py` in the terminal

import requests
from bs4 import BeautifulSoup
import re


def normalize_spaces(text: str) -> str:
    if text is None:
        return ""
    # Convert non-breaking spaces to normal spaces
    text = text.replace("\u00a0", " ")
    # Collapse any run of whitespace into a single space
    text = re.sub(r"\s+", " ", text)
    # Strip spaces at start and end
    return text.strip()


def process_ingredient_block(block):
    """Return a list of (name, quantity) from one ingredient block."""
    ingredients = []

    # Find all ingredient lines in this block
    for dd in block.select("dd.gz-ingredient"):
        # Name is in the <a> tag
        name_tag = dd.find("a")
        if not name_tag:
            continue
        raw_name = name_tag.get_text(" ", strip=True)

        # Quantity + notes are inside the <span>
        span = dd.find("span")
        raw_quantity = span.get_text(" ", strip=True) if span else ""

        name = normalize_spaces(raw_name)
        quantity = normalize_spaces(raw_quantity)

        if name:
            ingredients.append((name, quantity))

    return ingredients


def scrape():
    url = "https://www.giallozafferano.com/recipes/baked-pasta-with-zucchini-cream.html"
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.select_one("h1").get_text(strip=True)

    all_ingredients = []

    # Prefer the main ingredients containers only
    for block in soup.select("div.gz-ingredients.gz-outer"):
        all_ingredients.extend(process_ingredient_block(block))

    # If for some reason there are also loose <dl> blocks with ingredients,
    # you can optionally include them too:
    # for block in soup.select("dl.gz-list-ingredients"):
    #     all_ingredients.extend(process_ingredient_block(block))

    # Deduplicate by ingredient name (keep the first occurrence)
    seen = set()
    unique_ingredients = []
    for name, quantity in all_ingredients:
        if name not in seen:
            seen.add(name)
            unique_ingredients.append((name, quantity))

    # Example: first link and text paragraphs (as you had)
    text = [p.get_text(" ", strip=True) for p in soup.select("p")]
    link = soup.select_one("a").get("href")

    print(title)
    for name, quant in unique_ingredients:
        print(f"{name} : {quant}")

    # Optional: print the rest
    print('\n'.join(text))
    # print(link)


if __name__ == "__main__":
    scrape()