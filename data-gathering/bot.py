from scraper.scraper import RecipeScraper

from pathlib import Path

def remove_ws(string: str):
    return string.replace(' ', '')

class Bot:
    def __init__(self, scraper: RecipeScraper) -> None:
        self.scraper = scraper

    def scrape(self,
               links: list[str],
               output_dir: Path,
               log: Path) -> None:
        # visit and scrape links
        for url in links:
            try:
                info = self.scraper.scrape(url)
                self.save_info(info, output_dir)
            except Exception as exception:
                self.write_error_log(exception, log)

    def write_error_log(error: Exception, log: Path):
        with open(log, 'w') as file:
            file.write(f"[ERROR] : {error}")

    def save_info(info: str, dest: Path):
        if not Path.exists(dest):   # create path if not existing
            Path.mkdir(dest)
        
        dest_recipe = dest + "/" + info.name
        if not Path.exists(dest_recipe):
            pass
    