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

const chromeOptionsForAWSLambda = {
    args: [
        // Flags for running in Docker on AWS Lambda
        // https://www.howtogeek.com/devops/how-to-run-puppeteer-and-headless-chrome-in-a-docker-container
        // https://github.com/alixaxel/chrome-aws-lambda/blob/f9d5a9ff0282ef8e172a29d6d077efc468ca3c76/source/index.ts#L95-L118
        // https://github.com/Sparticuz/chrome-aws-lambda/blob/master/source/index.ts#L95-L123
        '--allow-running-insecure-content',
        '--autoplay-policy=user-gesture-required',
        '--disable-background-timer-throttling',
        '--disable-component-update',
        '--disable-dev-shm-usage',
        '--disable-domain-reliability',
        '--disable-features=AudioServiceOutOfProcess,IsolateOrigins,site-per-process',
        '--disable-ipc-flooding-protection',
        '--disable-print-preview',
        '--disable-setuid-sandbox',
        '--disable-site-isolation-trials',
        '--disable-speech-api',
        '--disable-web-security',
        '--disk-cache-size=33554432',
        '--enable-features=SharedArrayBuffer',
        '--hide-scrollbars',
        '--ignore-gpu-blocklist',
        '--in-process-gpu',
        '--mute-audio',
        '--no-default-browser-check',
        '--no-first-run',
        '--no-pings',
        '--no-sandbox',
        '--no-zygote',
        '--single-process',
        '--use-angle=swiftshader',
        '--use-gl=swiftshader',
        '--window-size=1920,1080',
    ],
    headless: true,
    ignoreHTTPSErrors: true,
    // defaultViewport: null,
    defaultViewport: {
        width: 1200,
        height: 900
    }
}
const chromeOptions = {
    args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--hide-scrollbars',

        "--disable-gpu",
        "--disable-extensions",
        "user-data-dir=/home/node",
        // "--profile-directory=Profile 6";
    ],
    headless: true,
    ignoreHTTPSErrors: true,
    // defaultViewport: null,
    // defaultViewport: {
    //     width: 1200,
    //     height: 900
    // }
}

module.exports = {
    scrape
};

scrape().then((result) => {
    console.log("result: ", result);
});

