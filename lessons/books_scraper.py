import json
from playwright.sync_api import sync_playwright
from typing import List

def scrape_books_with_playwright() -> List[dict]:
    """
    Scrape book information from https://books.toscrape.com/ using Playwright.

    Returns:
        List[dict]: A list of dictionaries containing book title, price, and availability.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('https://books.toscrape.com/')

        book_elements = page.query_selector_all('.product_pod')
        books = []

        for book_element in book_elements:
            try:
                title = book_element.query_selector('h3 > a').get_attribute('title')
                price = book_element.query_selector('.price_color').text_content()
                availability = book_element.query_selector('.availability').text_content().strip()

                books.append({
                    'title': title,
                    'price': price,
                    'availability': availability
                })
            except Exception as e:
                print(f"Error while scraping book: {e}")

        browser.close()
        return books

def main():
    """
    Main function that calls the scrape_books_with_playwright function and saves the result to a JSON file.
    """
    book_list = scrape_books_with_playwright()

    with open('book_list.json', 'w') as output_file:
        json.dump(book_list, output_file)

    print(book_list)
    # return book_list


if __name__ == '__main__':
    main()