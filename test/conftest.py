import sys
from os import path
from selenium.webdriver.chrome.options import Options

sys.path.append(path.dirname(path.dirname(__file__)))


def pytest_setup_options():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')
    return options
