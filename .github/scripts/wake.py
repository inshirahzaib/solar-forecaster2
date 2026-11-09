import os
from playwright.sync_api import sync_playwright

URL = os.environ["APP_URL"]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(8000)

    button = page.get_by_role("button", name="Yes, get this app back up!")
    if button.count() > 0:
        print("App was asleep. Clicking the wake-up button...")
        button.click()
        page.wait_for_timeout(90000)
    else:
        print("App looks awake. Staying on the page for 30 seconds...")
        page.wait_for_timeout(30000)

    print("Page title:", page.title())
    browser.close()
print("Done.")
