import requests
from bs4 import BeautifulSoup


def get_text_from_url(url: str) -> BeautifulSoup:
    # Responseオブジェクト生成
    response = requests.get(url)
    # 文字化け防止
    response.encoding = response.apparent_encoding
    # BeautifulSoupオブジェクト生成
    return BeautifulSoup(response.text, "html.parser")
