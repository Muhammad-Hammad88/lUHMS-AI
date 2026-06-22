"""
scraper.py
----------
LUMHS website scraper with two modes:

1. FULL SCRAPE  — scrapes entire website (used by update.py)
2. SECTION SCRAPE — scrapes only specific section URLs (used by admin panel)

Handles:
- Static HTML pages
- Dynamic JavaScript-rendered pages (Selenium)
- PDF text extraction (PyMuPDF)
- Duplicate URL prevention
- Automatic driver restart every 100 pages
"""

import json
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from typing import Optional

import fitz
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

from sections import SECTIONS, get_section_for_url, get_all_section_urls

# =========================
# CONSTANTS
# =========================
BASE_URL = "https://www.lumhs.edu.pk"
START_URLS = [
    "https://www.lumhs.edu.pk",
    "https://www.lumhs.edu.pk/home/"
]
OUTPUT_FILE = "scraped_data.json"
SELENIUM_RESTART_EVERY = 100
REQUEST_DELAY = 0.5  # seconds between requests
REQUEST_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (compatible; LUMHSBot/1.0)"

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".css",
    ".js", ".ico", ".svg", ".mp4", ".mp3",
    ".zip", ".rar"
)

# =========================
# URL UTILITIES
# =========================
def normalize_url(url: str) -> str:
    """Remove fragment and trailing slash for consistent deduplication."""
    return urldefrag(url)[0].rstrip("/")

def is_valid_url(url: str) -> bool:
    """Check if URL should be scraped."""
    path = urlparse(url).path.lower()
    if len(path) <= 1:
        return False
    if path.endswith(SKIP_EXTENSIONS):
        return False
    if url.startswith("#"):
        return False
    return True

def is_lumhs_url(url: str) -> bool:
    """Check if URL belongs to LUMHS domain."""
    return urlparse(url).netloc.endswith("lumhs.edu.pk")

# =========================
# SELENIUM DRIVER
# =========================
def create_driver() -> webdriver.Chrome:
    """Create a headless Chrome driver."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(f"--user-agent={USER_AGENT}")
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

# =========================
# TEXT EXTRACTION
# =========================
def extract_pdf_text(url: str, content: bytes) -> str:
    """Extract text from PDF binary content."""
    try:
        pdf = fitz.open(stream=content, filetype="pdf")
        text = "".join(page.get_text() for page in pdf)
        pdf.close()
        return text.strip()
    except Exception as e:
        print(f"  [PDF ERROR] {url}: {e}", flush=True)
        return ""

def clean_text(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_page_text_static(soup: BeautifulSoup, title: str = "") -> str:
    """Extract clean text from static HTML soup."""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = (title + " " + soup.get_text(separator=' ', strip=True))
    return clean_text(text)

def selenium_render(driver: webdriver.Chrome, url: str) -> tuple[str, Optional[BeautifulSoup]]:
    """
    Render a dynamic page using Selenium.
    Returns (cleaned_text, soup) or ("", None) on failure.
    """
    try:
        driver.get("about:blank")
        time.sleep(0.2)
        driver.get(url)
        time.sleep(2)

        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Expand dropdown menus to reveal hidden links
        for selector in [
            ".dropdown-toggle",
            ".menu-item-has-children > a",
            "[data-toggle='dropdown']",
            "[aria-haspopup='true']"
        ]:
            try:
                for elem in driver.find_elements(By.CSS_SELECTOR, selector):
                    try:
                        driver.execute_script("arguments[0].click();", elem)
                        time.sleep(0.3)
                    except Exception:
                        pass
            except Exception:
                pass

        # Scroll to trigger lazy loading
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Extract headings separately for better chunking later
        title = driver.title.strip()
        text_parts = []
        if title:
            text_parts.append("TITLE: " + title)
        for i in range(1, 7):
            for h in soup.find_all(f"h{i}"):
                text_parts.append(f"H{i}: {h.get_text(strip=True)}")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text_parts.append(soup.get_text(separator=' ', strip=True))
        return clean_text(" ".join(text_parts)), soup

    except Exception as e:
        print(f"  [SELENIUM ERROR] {url}: {e}", flush=True)
        return "", None

# =========================
# CORE SCRAPER CLASS
# =========================
class LUMHSScraper:
    """
    Main scraper class.
    Supports full scrape and section-specific scrape.
    """

    def __init__(self):
        self.driver: Optional[webdriver.Chrome] = None
        self.selenium_count = 0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _ensure_driver(self):
        if self.driver is None:
            self.driver = create_driver()

    def _restart_driver_if_needed(self):
        self.selenium_count += 1
        if self.selenium_count % SELENIUM_RESTART_EVERY == 0:
            print("  [DRIVER] Restarting Chrome driver...", flush=True)
            self.driver.quit()
            self.driver = create_driver()

    def _scrape_url(self, url: str) -> tuple[str, Optional[BeautifulSoup]]:
        """
        Scrape a single URL.
        Returns (text, soup). Text is empty string on failure.
        """
        try:
            time.sleep(REQUEST_DELAY)
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)

            if resp.status_code != 200:
                print(f"  [SKIP] {url} — HTTP {resp.status_code}", flush=True)
                return "", None

            ctype = resp.headers.get("Content-Type", "").lower()

            # PDF
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                text = extract_pdf_text(url, resp.content)
                return text, None

            # Non-HTML
            if "html" not in ctype:
                return "", None

            # Check if page is dynamic (very little text)
            soup = BeautifulSoup(resp.text, "html.parser")
            quick_text = soup.get_text(strip=True)

            if len(quick_text) < 500 or url in [normalize_url(u) for u in START_URLS]:
                self._ensure_driver()
                text, soup = selenium_render(self.driver, url)
                self._restart_driver_if_needed()
                return text, soup
            else:
                title = soup.title.string if soup.title else ""
                text = extract_page_text_static(soup, title)
                return text, soup

        except Exception as e:
            print(f"  [ERROR] {url}: {e}", flush=True)
            return "", None

    def _discover_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        """Extract all valid LUMHS links from a page."""
        links = []
        if not soup:
            return links
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            full = normalize_url(urljoin(base_url, href))
            if is_lumhs_url(full) and is_valid_url(full):
                links.append(full)
        return links

    def scrape_full(self, progress_callback=None) -> list[dict]:
        """
        Full website scrape starting from START_URLS.
        Follows all internal links recursively.
        Returns list of {url, content, section} dicts.
        """
        queue = deque()
        seen = set()
        results = []
        html_count = 0
        pdf_count = 0

        for url in START_URLS:
            u = normalize_url(url)
            queue.append(u)
            seen.add(u)

        print("Starting FULL scrape...", flush=True)

        while queue:
            url = queue.popleft()

            if progress_callback:
                progress_callback(f"Visiting: {url}")

            text, soup = self._scrape_url(url)

            if not text or len(text) < 100:
                continue

            section = get_section_for_url(url)

            if url.lower().endswith(".pdf"):
                pdf_count += 1
                print(f"  [PDF {pdf_count}] {url}", flush=True)
            else:
                html_count += 1
                print(f"  [HTML {html_count}] {url}", flush=True)

            results.append({
                "url": url,
                "content": text,
                "section": section
            })

            # Discover new links
            if soup:
                for link in self._discover_links(soup, url):
                    if link not in seen:
                        seen.add(link)
                        queue.append(link)

        self._cleanup()
        print(f"\nFull scrape done: {html_count} HTML, {pdf_count} PDFs", flush=True)
        return results

    def scrape_section(self, section_name: str, progress_callback=None) -> list[dict]:
        """
        Scrape only URLs belonging to a specific section.
        Also follows links found on those pages that belong to same section.
        Returns list of {url, content, section} dicts.
        """
        section_urls = SECTIONS.get(section_name, [])

        if not section_urls:
            print(f"  [WARN] No URLs defined for section: {section_name}", flush=True)
            return []

        queue = deque()
        seen = set()
        results = []
        count = 0

        for url in section_urls:
            u = normalize_url(url)
            queue.append(u)
            seen.add(u)

        print(f"Starting SECTION scrape: {section_name} ({len(section_urls)} seed URLs)", flush=True)

        while queue:
            url = queue.popleft()

            if progress_callback:
                progress_callback(f"[{section_name}] Visiting: {url}")

            text, soup = self._scrape_url(url)

            if not text or len(text) < 100:
                continue

            count += 1
            print(f"  [{count}] {url}", flush=True)

            results.append({
                "url": url,
                "content": text,
                "section": section_name
            })

            # Follow links only if they belong to the same section
            if soup:
                for link in self._discover_links(soup, url):
                    if link not in seen:
                        link_section = get_section_for_url(link)
                        if link_section == section_name:
                            seen.add(link)
                            queue.append(link)

        self._cleanup()
        print(f"\nSection scrape done: {section_name} — {count} pages", flush=True)
        return results

    def _cleanup(self):
        """Quit Selenium driver if running."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

# =========================
# SAVE HELPERS
# =========================
def save_full(data: list[dict], path: str = OUTPUT_FILE) -> None:
    """Save full scrape results to JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} pages to {path}", flush=True)

def load_existing(path: str = OUTPUT_FILE) -> list[dict]:
    """Load existing scraped data JSON."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def merge_section_data(existing: list[dict], new_data: list[dict], section: str) -> list[dict]:
    """
    Replace existing data for a section with new data.
    Keeps all other sections intact.
    """
    filtered = [p for p in existing if p.get("section") != section]
    merged = filtered + new_data
    print(f"Merged: removed old {section} data, added {len(new_data)} new pages", flush=True)
    return merged

# =========================
# STANDALONE ENTRY POINT
# =========================
if __name__ == "__main__":
    scraper = LUMHSScraper()
    data = scraper.scrape_full()
    save_full(data)