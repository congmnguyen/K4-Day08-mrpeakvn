"""Crawl bài giải thích chính sách giao thông từ các cổng thông tin chính thức."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
USER_AGENT = "K4-Day08-RAG-Lab/1.0 (educational corpus; contact: local project)"

ARTICLES = [
    {
        "slug": "quy-dinh-chuyen-huong-lui-xe-tranh-xe",
        "url": "https://xaydungchinhsach.chinhphu.vn/quy-dinh-chuyen-huong-xe-lui-xe-tranh-xe-di-nguoc-chieu-119241219130310079.htm",
        "category": "driving_rules",
        "audience": "driver",
    },
    {
        "slug": "quy-dinh-toc-do-khoang-cach-an-toan-2025",
        "url": "https://xaydungchinhsach.chinhphu.vn/quy-dinh-moi-ve-toc-do-khoang-cach-an-toan-cua-phuong-tien-tham-gia-giao-thong-tren-duong-bo-119241127153008518.htm",
        "category": "speed_and_distance",
        "audience": "driver",
    },
    {
        "slug": "giai-quyet-tai-nan-giao-thong-thu-tuc-hanh-chinh",
        "url": "https://xaydungchinhsach.chinhphu.vn/giai-quyet-vu-tai-nan-giao-thong-duong-bo-theo-thu-tuc-hanh-chinh-119241129085759077.htm",
        "category": "accident_procedure",
        "audience": "all_road_users",
    },
    {
        "slug": "nghi-dinh-168-ban-hanh-theo-thu-tuc-rut-gon",
        "url": "https://baochinhphu.vn/nghi-dinh-168-2024-nd-cp-ban-hanh-theo-thu-tuc-rut-gon-102250112163724595.htm",
        "category": "penalties",
        "audience": "all_road_users",
    },
    {
        "slug": "nghi-dinh-168-giam-tai-nan-giao-thong",
        "url": "https://csgt.bocongan.gov.vn/m/tintuc/19860/Nghi-dinh-168-tao-hieu-ung-tich-cuc-gop-phan-giam-tai-nan-giao-thong.html",
        "category": "policy_impact",
        "audience": "all_road_users",
    },
]
ARTICLE_URLS = [article["url"] for article in ARTICLES]


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _extract_article(url: str) -> dict:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    soup = BeautifulSoup(response.text, "html.parser")

    for unwanted in soup.select(
        "script, style, noscript, nav, footer, form, iframe, .social, .related, .advertisement"
    ):
        unwanted.decompose()

    title_tag = soup.select_one("h1")
    title = title_tag.get_text(" ", strip=True) if title_tag else "Unknown"
    published_meta = soup.select_one(
        'meta[property="article:published_time"], meta[name="pubdate"], meta[name="publishdate"]'
    )
    published_at = published_meta.get("content") if published_meta else None

    selectors = (
        "article",
        ".article__body",
        ".article-body",
        ".detail-content",
        ".detail__content",
        ".content-detail",
        "main",
    )
    container = next((soup.select_one(selector) for selector in selectors if soup.select_one(selector)), None)
    if container is None:
        raise ValueError(f"Could not locate article body: {url}")

    blocks: list[str] = []
    for element in container.find_all(["h2", "h3", "h4", "p", "li"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if len(text) < 20:
            continue
        if element.name in {"h2", "h3", "h4"}:
            blocks.append(f"## {text}")
        elif element.name == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)

    content_markdown = "\n\n".join(dict.fromkeys(blocks))
    if len(content_markdown) < 500:
        raise ValueError(f"Article content too short ({len(content_markdown)} chars): {url}")

    article_config = next(item for item in ARTICLES if item["url"] == url)
    return {
        "url": url,
        "title": title,
        "published_at": published_at,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "publisher": response.url.split("/", 3)[2],
        "category": article_config["category"],
        "audience": article_config["audience"],
        "content_markdown": content_markdown,
    }


async def crawl_article(url: str) -> dict:
    """Crawl một bài bằng HTTP trong worker thread để không block event loop."""
    return await asyncio.to_thread(_extract_article, url)


async def crawl_all() -> list[dict]:
    """Crawl tuần tự, có delay nhỏ để không tạo tải dồn lên nguồn công khai."""
    setup_directory()
    results: list[dict] = []
    for index, config in enumerate(ARTICLES, 1):
        print(f"[{index}/{len(ARTICLES)}] Crawling: {config['url']}")
        article = await crawl_article(config["url"])
        filepath = DATA_DIR / f"{config['slug']}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(article)
        print(f"  ✓ {filepath.name} ({len(article['content_markdown']):,} chars)")
        if index < len(ARTICLES):
            await asyncio.to_thread(time.sleep, 1)
    return results


if __name__ == "__main__":
    asyncio.run(crawl_all())
