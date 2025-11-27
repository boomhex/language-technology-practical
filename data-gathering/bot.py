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
        dest = CrawlBot.make_dirs(self.dest_dir)    # make new directory
        CrawlBot.create_info_file(page, dest)
        CrawlBot.create_datafile(page, dest)

    def create_info_file(page: PageInfo, dest: Path) -> None:
        dest = dest.joinpath("info.txt")
        dest.touch()
        with open(dest, 'w') as file:
        #     text = line("{") + \
        #     tab() + line(f"name: {page.ident}") + \
        #     tab() + line(f"url: {page.url}") + \
        #     line("}")
        # file.write(text)
            dictionary = {
                'name' : page.ident,
                'url' : page.url
            }

            json.dump(dictionary, file)

    @staticmethod
    def create_datafile(page: PageInfo, dest: Path) -> None:
        dest = dest.joinpath("data.txt")
        dest.touch()
        with open(dest, 'w') as file:
            text = str(page)
            file.write(text)

    @staticmethod
    def make_dirs(dest_dir) -> Path:
        if not Path.exists(dest_dir):
            Path.mkdir(dest_dir)

        max_dest = CrawlBot.n_thfolder(dest_dir)
        print(max_dest)
        dest = Path(dest_dir, max_dest)
        Path.mkdir(dest)
        return dest

    @staticmethod
    def n_thfolder(dir: Path) -> str:
        return str(len([f.name for f in dir.iterdir()]))

scraper = RecipeScraper()
path = Path("getRecipes/recipes.txt")
dest = Path("data/")
bot = CrawlBot(scraper, path, dest)
bot.crawl()