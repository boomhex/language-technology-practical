import requests
from bs4 import BeautifulSoup

url = "https://www.giallozafferano.com/recipes/baked-pasta-with-zucchini-cream.html"  # replace with the real URL

resp = requests.get(url)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

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
unique_ingredients = list(dict.fromkeys(ingredients))

for name, quantity in unique_ingredients:
    print(f"{name}: {quantity}")