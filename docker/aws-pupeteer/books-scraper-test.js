const assert = require('assert');
const { scrape } = require('./books-scraper.js');

describe('scraper', () => {

    it('should scrape book titles from https://books.toscrape.com', async () => {
        const bookTitles = await scrape();
        assert(Array.isArray(bookTitles));
        assert(bookTitles.length > 0);
        assert(typeof bookTitles[0] === 'string');
    }).timeout(30000);

});
