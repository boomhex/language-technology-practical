from scraper.recipeScraper import RecipeScraper
from scraper.scraperABC import Scraper
from scraper.pageinfo import PageInfo
from pathlib import Path
import json

def remove_newline(line: str):
    return line.replace("\n", "")

def line(text: str):
    return text + '\n'

def tab():
    return "    "

class CrawlBot:
    def __init__(self, scraper: Scraper, url_file: Path, dest_folder: Path) -> None:
        self.scraper = scraper
        self.url_file = url_file
        self.dest_dir = dest_folder

    def crawl(self) -> None:
        url_list = CrawlBot.urls(self.url_file)     # convert file to list
        self.visit(url_list)

    @staticmethod
    def urls(fp: Path) -> list:
        urls = []
        with open(fp, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = remove_newline(line)
                urls.append(line)
        return urls

    def visit(self, urls: list[str]) -> None:
        for url in urls:        # scrape and save
            page_info = self.scraper.scrape(url)
            self.save_page(page_info)

    def save_page(self, page: PageInfo) -> None:
        CrawlBot.make_dirs(self.dest_dir)    # make new directory
        CrawlBot.create_datafile(page, self.dest_dir)

    def create_info_file(page: PageInfo, dest: Path) -> None:
        dest = dest.joinpath("info.txt")
        dest.touch()
        with open(dest, 'w') as file:
            dictionary = {
                'name' : page.ident,
                'url' : page.url
            }

            json.dump(dictionary, file)

    @staticmethod
    def create_datafile(page: PageInfo, dest: Path) -> None:
        n_file = CrawlBot.n_thfile(dest)
        dest = dest.joinpath(n_file + ".txt")
        dest.touch()
        with open(dest, 'w') as file:
            text = str(page)
            file.write(text)

    @staticmethod
    def make_dirs(dest_dir):
        if not Path.exists(dest_dir):
            Path.mkdir(dest_dir)

    @staticmethod
    def n_thfolder(dir: Path) -> str:
        all_files = [f.name for f in dir.iterdir()]
        return str(len(all_files))
    
    @staticmethod
    def n_thfile(dir: Path) -> str:
        all_files = [f.name for f in dir.iterdir()]
        return str(len(all_files))

if __name__ == "__main__":
    scraper = RecipeScraper()
    path = Path("getRecipes/recipes.txt")
    dest = Path("../model/data")
    bot = CrawlBot(scraper, path, dest)
    bot.crawl()
