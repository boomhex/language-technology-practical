import json

class PageInfo:
    def __init__(self,
                 ident: str = "",
                 text: str = "",
                 url: str = "") -> None:
        self.ident = ident
        self.text = text
        self.url = url

    def __str__(self) -> str:
        info = {
            'name':self.ident,
            
        }
        string = f"name:\n{self.ident}\n" \
                 f"text:\n{self.text}\n" \
                 f"url:\n{self.url}\n"
    
        return string