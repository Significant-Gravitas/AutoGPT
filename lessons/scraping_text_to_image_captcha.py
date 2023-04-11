import os

from playwright.sync_api import sync_playwright
from python_anticaptcha import AnticaptchaClient, NoCaptchaTaskProxylessTask

ANTICAPTCHA_KEY = os.getenv("ANTICAPTCHA_KEY")


# Example of Python function to ise playwright for web scraping ahd anticaptcha for solving captcha.
def submit_form(search_query):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigating to the URL
        page.goto('https://egov.sos.state.or.us/br/pkg_web_name_srch_inq.login')

        # Resolving Captcha
        client = AnticaptchaClient(ANTICAPTCHA_KEY)
        task = NoCaptchaTaskProxylessTask('https://egov.sos.state.or.us/br/pkg_web_name_srch_inq.login',
                                          page.frames[1].url)
        job = client.createTask(task)
        job.join()

        # Getting the solution
        solution = job.get_solution_response()

        # Filling the form
        page.fill('input[name=p_name]', search_query)
        page.frames[1].evaluate('document.getElementById("g-recaptcha-response").innerHTML = "{}"'.format(solution))

        # Submitting the form
        page.click('input[value=SEARCH]')

        # Waiting for the result page to load
        page.wait_for_load_state()

        # Scraping the result page
        results = extract_results(page)
        browser.close()

        return results

