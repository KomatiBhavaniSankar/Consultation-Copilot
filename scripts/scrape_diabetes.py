
import os
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urljoin

URL = "https://my.clevelandclinic.org/health/diseases/7104-diabetes"
OUTPUT_DIR = os.path.join("knowledge", "lifestyle_disease")   # then run: python -m app.ingest

os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/130.0.0.0 Safari/537.36"
}


def scrape_diabetes_page():
    try:
        print(f"Fetching: {URL}")

        response = requests.get(
            URL,
            headers=HEADERS,
            timeout=30
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract page title
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(" ", strip=True) if title_tag else "Diabetes"

        # Extract the main article content
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

        # Remove navigation, advertisements and scripts
        for tag in main.find_all([
            "script", "style", "noscript",
            "nav", "footer", "iframe", "form"
        ]):
            tag.decompose()

        # Extract headings and paragraphs in reading order
        sections = []
        for element in main.find_all([
            "h1", "h2", "h3", "h4",
            "p", "li", "table"
        ]):
            text = element.get_text(" ", strip=True)

            if not text:
                continue

            if element.name in ["h1", "h2", "h3", "h4"]:
                sections.append({
                    "type": "heading",
                    "level": element.name,
                    "text": text
                })
            elif element.name == "table":
                sections.append({
                    "type": "table",
                    "text": text
                })
            else:
                sections.append({
                    "type": element.name,
                    "text": text
                })

        # Extract article text
        article_text = "\n\n".join(
            section["text"] for section in sections
        )

        # Extract article images
        images = []
        for img in main.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src:
                images.append({
                    "url": urljoin(URL, src),
                    "alt": img.get("alt", "")
                })

        # Extract links
        links = []
        for a in main.find_all("a", href=True):
            links.append({
                "text": a.get_text(" ", strip=True),
                "url": urljoin(URL, a["href"])
            })

        # Package the data
        data = {
            "source_url": URL,
            "title": title,
            "scraped_at": datetime.now().isoformat(),
            "sections": sections,
            "article_text": article_text,
            "images": images,
            "links": links
        }

        # Save readable text
        with open(
            os.path.join(OUTPUT_DIR, "diabetes_article.txt"),
            "w",
            encoding="utf-8"
        ) as file:
            file.write(f"{title}\n")
            file.write(f"Source: {URL}\n\n")
            file.write(article_text)

        # Save structured JSON
        with open(
            os.path.join(OUTPUT_DIR, "diabetes_article.json"),
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        # Save original HTML response
        with open(
            os.path.join(OUTPUT_DIR, "diabetes_article.html"),
            "w",
            encoding="utf-8"
        ) as file:
            file.write(response.text)

        print("\nScraping completed!")
        print(f"Title: {title}")
        print(f"Sections extracted: {len(sections)}")
        print(f"Images found: {len(images)}")
        print(f"Links found: {len(links)}")
        print(f"Output folder: {os.path.abspath(OUTPUT_DIR)}")

    except requests.RequestException as error:
        print(f"Request failed: {error}")
        print("Check your internet connection or website access.")
    except Exception as error:
        print(f"Scraping failed: {error}")


if __name__ == "__main__":
    scrape_diabetes_page()