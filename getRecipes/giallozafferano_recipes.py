import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.giallozafferano.com"
START_URL = f"{BASE_URL}/recipes-list/"
TARGET_COUNT = 200

def is_recipe_url(href: str) -> bool:
    if not href:
        return False
    # Normalize: absolute or relative
    # We only care that it is an English recipe on giallozafferano.com
    if "/recipes/" not in href:
        return False
    if href.endswith(".html"):
        return True
    return False

def collect_recipe_links(start_url=START_URL, target_count=TARGET_COUNT):
    collected = []
    seen = set()
    url = start_url

    while url and len(collected) < target_count:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # 1) collect links that look like recipe pages
        page_new = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]

            # Convert relative links to absolute
            full = urljoin(BASE_URL, href)

            if not is_recipe_url(full):
                continue

            if full not in seen:
                seen.add(full)
                collected.append(full)
                page_new += 1

                if len(collected) >= target_count:
                    break

        # print(f"  -> found {page_new} new recipes on this page "
        #       f"(total: {len(collected)})")

        # 2) find "Next page" link to continue through the catalogue
        next_link = None
        for a in soup.find_all("a", href=True):
            text = (a.get_text() or "").strip().lower()
            if "next page" in text:
                next_link = urljoin(BASE_URL, a["href"])
                break

        if not next_link:
            print("No more pages found, stopping.")
            break

        url = next_link

    return collected

if __name__ == "__main__":
    links = collect_recipe_links()
    for i, link in enumerate(links, start=1):
        print(f"{link}")