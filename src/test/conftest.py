from selenium.webdriver.chrome.options import Options


def pytest_setup_options():
    options = Options()
    # options.remove_argument("--webdriver")
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')
    return options
