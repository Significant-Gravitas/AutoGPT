const puppeteer = require("puppeteer");

const scrape = async () => {
    let result = null;
    console.log('Scraping: https://books.toscrape.com');

    try {
        const browser = await puppeteer.launch(chromeOptions);
        const page = await browser.newPage();
        await page.goto("https://books.toscrape.com/");

        result = await page.$$eval("h3 > a", el => el.map(x => x.getAttribute("title")));

        console.log("bookTitles: ", result);

        await browser.close();
        console.log('Finished scraper.');

    } catch (error) {
        console.log("#### ERR: ", error)
    }
    return result
};

const chromeOptions = {
    args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--hide-scrollbars',

        "--disable-gpu",
        "--disable-extensions",
    ],
    headless: true,
    ignoreHTTPSErrors: true
}

module.exports = {
    scrape
};

scrape().then((result) => {
    console.log("result: ", result);
});

