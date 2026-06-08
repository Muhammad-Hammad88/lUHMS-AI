import json
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag

import fitz
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://www.lumhs.edu.pk"
START_URLS = ["https://www.lumhs.edu.pk", "https://www.lumhs.edu.pk/home/"]
OUTPUT_FILE = "scraped_data.json"

# ---------- STATE ----------
queue = deque()
seen_urls = set()
all_data = []

def normalize(u):
    return urldefrag(u)[0].rstrip("/")

for u in START_URLS:
    u = normalize(u)
    queue.append(u)
    seen_urls.add(u)

# ---------- SELENIUM ----------
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

selenium_count = 0
SELENIUM_RESTART_EVERY = 100

def restart_driver():
    global driver
    driver.quit()
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

def restart_if_needed():
    global selenium_count
    if selenium_count > 0 and selenium_count % SELENIUM_RESTART_EVERY == 0:
        restart_driver()

# ---------- HELPERS ----------
def clean_text(t):
    # Remove repeated whitespace only - no hardcoded strings
    return re.sub(r'\s+', ' ', t).strip()

def extract_pdf(url, content):
    try:
        pdf = fitz.open(stream=content, filetype="pdf")
        text = "".join(p.get_text() for p in pdf)
        pdf.close()
        return text.strip()
    except:
        return ""

def is_valid_link(url):
    path = urlparse(url).path.lower()
    if len(path) <= 1:
        return False
    if path.endswith((".jpg", ".jpeg", ".png", ".gif", ".css", ".js", ".ico", ".svg", ".mp4", ".mp3", ".zip", ".rar")):
        return False
    if url.startswith("#"):
        return False
    return True

def selenium_render(url):
    global selenium_count
    try:
        driver.get("about:blank")
        time.sleep(0.2)
        driver.get(url)
        time.sleep(2)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Expand dropdowns
        for sel in [".dropdown-toggle", ".menu-item-has-children > a", "[data-toggle='dropdown']", "[aria-haspopup='true']"]:
            try:
                for elem in driver.find_elements(By.CSS_SELECTOR, sel):
                    try:
                        driver.execute_script("arguments[0].click();", elem)
                        time.sleep(0.3)
                    except:
                        pass
            except:
                pass

        # Scroll to load lazy content
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = BeautifulSoup(driver.page_source, "html.parser")
        selenium_count += 1
        restart_if_needed()

        # Remove unwanted tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extract all text
        title = driver.title.strip()
        text_parts = []
        if title:
            text_parts.append("TITLE: " + title)

        for i in range(1, 7):
            for h in soup.find_all(f"h{i}"):
                text_parts.append(f"H{i}: {h.get_text(strip=True)}")

        body_text = soup.get_text(separator=' ', strip=True)
        text_parts.append(body_text)

        text = clean_text(" ".join(text_parts))
        return text, soup

    except Exception as e:
        print(f"Selenium error {url}: {e}", flush=True)
        return "", None

# ---------- MAIN LOOP ----------
records = 0
pdfs = 0

print("Starting scraper...", flush=True)

while queue:
    url = normalize(queue.popleft())

    try:
        time.sleep(0.5)

        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True
        )

        if resp.status_code != 200:
            print(f"Skipped {url} - {resp.status_code}", flush=True)
            continue

        ctype = resp.headers.get("Content-Type", "").lower()

        # Handle PDFs
        if "pdf" in ctype or url.endswith(".pdf"):
            text = extract_pdf(url, resp.content)
            if text and len(text) >= 100:
                all_data.append({"url": url, "content": text})
                pdfs += 1
                print(f"[PDF {pdfs}] {url}", flush=True)
            continue

        if "html" not in ctype:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        quick = soup.get_text(strip=True)

        # Use Selenium for dynamic or start pages
        if len(quick) < 500 or url in [normalize(u) for u in START_URLS]:
            text, soup = selenium_render(url)
            if not text:
                continue
        else:
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            title = soup.title.string if soup.title else ""
            text = clean_text(title + " " + soup.get_text(separator=' ', strip=True))

        if not text or len(text) < 100:
            continue

        all_data.append({"url": url, "content": text})
        records += 1
        print(f"[HTML {records}] {url}", flush=True)

        # Discover new links
        if soup:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue
                full = normalize(urljoin(url, href))
                parsed = urlparse(full)
                if parsed.netloc.endswith("lumhs.edu.pk") and is_valid_link(full):
                    if full not in seen_urls:
                        queue.append(full)
                        seen_urls.add(full)

    except Exception as e:
        print(f"Error {url}: {e}", flush=True)

driver.quit()

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"\nDone! {records} HTML pages, {pdfs} PDFs scraped.", flush=True)
print(f"Saved to {OUTPUT_FILE}", flush=True)