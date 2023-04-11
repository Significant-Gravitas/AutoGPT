import os

from playwright.sync_api import sync_playwright


def scrape_books(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigating to the URL
        page.goto(url)

        # Scraping the book information
        book_list = page.query_selector_all('.product_pod')
        books = []
        for book in book_list:
            title = book.query_selector('h3 > a').get_attribute('title')
            price = book.query_selector('.price_color').inner_text
            books.append({'title': title, 'price': price})

        browser.close()
        return books


def main():
    books = scrape_books('https://books.toscrape.com/')
    print(books)
    with open('book_list.txt', 'w') as f:
        for book in books:
            f.write(f"Title: {book['title']}, Price: {book['price']}\n")


if __name__ == '__main__':
    main()
