from abc import ABC, abstractmethod
import requests
from .pageinfo import PageInfo
from bs4 import BeautifulSoup

class Scraper(ABC):
    @staticmethod
    def get(url: str):
        response = requests.get(url)
        response.raise_for_status()
        return response

    def scrape(self, url: str) -> PageInfo:
        response = Scraper.get(url)             # retrieve page request
        page_info = self.order(response.text)   # format info
        page_info.url = url                     # add url to info
        return page_info                        # return

    def order(self, html: str):
        soup = BeautifulSoup(html, "html.parser")       # parse html
        title = self.title(soup)                        # extract relevant info
        info = self.info(soup)

        page_info = PageInfo(ident=title, text=info)    # store info

        return page_info

    @abstractmethod
    def title(self, soup: BeautifulSoup) -> str:
        pass

    @abstractmethod
    def info(self, soup: BeautifulSoup) -> str:
        pass
