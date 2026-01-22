"""
pip install requests beautifulsoup4 tqdm
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from requests import Response
from tqdm import tqdm


# =========================
# Configuration
# =========================
BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"

OUTPUT_ROOT = Path(r"C:\Users\Student\Documents\fall_detection\Datasets\URFD")
FALLS_DIR = OUTPUT_ROOT / "falls"
ADLS_DIR = OUTPUT_ROOT / "adls"

TIMEOUT = 30  # seconds
MAX_RETRIES = 6
CHUNK_SIZE = 1024 * 1024  # 1 MiB

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".mpg", ".mpeg"}


# =========================
# Helpers
# =========================
INVALID_WIN_CHARS = r'<>:"/\|?*'
INVALID_WIN_TRANS = str.maketrans({c: "_" for c in INVALID_WIN_CHARS})


def sanitize_filename(name: str) -> str:
    name = name.translate(INVALID_WIN_TRANS)
    name = re.sub(r"\s+", " ", name).strip()
    # avoid trailing dots/spaces (Windows)
    name = name.rstrip(" .")
    if not name:
        name = "file"
    return name


def is_video_url(url: str) -> bool:
    path = urlparse(url).path
    ext = Path(path).suffix.lower()
    return ext in VIDEO_EXTENSIONS


def parse_seq_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b(\d{1,3})\b", text.strip())
    if not m:
        return None
    return m.group(1).zfill(2)


def backoff_sleep(attempt: int) -> None:
    # exponential backoff with jitter
    base = min(60.0, 2.0 ** attempt)
    time.sleep(base + random.uniform(0.0, 1.0))


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    stream: bool = False,
    allow_redirects: bool = True,
    timeout: int = TIMEOUT,
    max_retries: int = MAX_RETRIES,
    headers: Optional[dict] = None,
) -> Response:
    last_err: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            resp = session.request(
                method,
                url,
                stream=stream,
                allow_redirects=allow_redirects,
                timeout=timeout,
                headers=headers,
            )
            if resp.status_code in (429,) or 500 <= resp.status_code <= 599:
                # transient
                try:
                    resp.close()
                except Exception:
                    pass
                backoff_sleep(attempt)
                continue
            resp.raise_for_status()
            return resp
        except (requests.RequestException,) as e:
            last_err = e
            backoff_sleep(attempt)
            continue
    raise RuntimeError(f"Request failed after {max_retries} retries: {method} {url}") from last_err


def ensure_dirs() -> None:
    FALLS_DIR.mkdir(parents=True, exist_ok=True)
    ADLS_DIR.mkdir(parents=True, exist_ok=True)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(1, 10_000):
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
    return parent / f"{stem}__{int(time.time())}{suffix}"


def extract_video_links_from_html(page_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(page_url, href)
        if is_video_url(abs_url):
            links.add(abs_url)
    return sorted(links)


def discover_download_urls(session: requests.Session, url: str) -> List[str]:
    # If URL itself is a direct video, use it.
    if is_video_url(url):
        return [url]

    # Probe with GET stream to inspect content-type and possible redirects without loading content.
    try:
        resp = request_with_retries(session, "GET", url, stream=True)
    except Exception:
        return []

    final_url = resp.url
    content_type = (resp.headers.get("Content-Type") or "").lower()

    # If redirect landed on a direct video, use final url.
    if is_video_url(final_url) and ("text/html" not in content_type and "html" not in content_type):
        try:
            resp.close()
        except Exception:
            pass
        return [final_url]

    # If content looks like a file (non-html), treat as a downloadable file.
    if "text/html" not in content_type and "html" not in content_type and content_type:
        try:
            resp.close()
        except Exception:
            pass
        return [final_url]

    # Otherwise, fetch full HTML and scrape video links inside.
    try:
        resp.close()
    except Exception:
        pass

    try:
        html_resp = request_with_retries(session, "GET", url, stream=False)
        html_text = html_resp.text
        html_resp.close()
    except Exception:
        return []

    found = extract_video_links_from_html(final_url, html_text)
    if found:
        return found

    # As a last fallback, if the original URL (or redirected URL) looks like a video even without extension,
    # attempt to download it directly.
    return [final_url]


def get_remote_content_length(session: requests.Session, url: str) -> Optional[int]:
    # Try HEAD first, then fallback to a lightweight GET.
    try:
        head = request_with_retries(session, "HEAD", url, stream=False, allow_redirects=True)
        cl = head.headers.get("Content-Length")
        head.close()
        if cl and cl.isdigit():
            return int(cl)
    except Exception:
        pass

    try:
        get = request_with_retries(session, "GET", url, stream=True, allow_redirects=True)
        cl = get.headers.get("Content-Length")
        get.close()
        if cl and cl.isdigit():
            return int(cl)
    except Exception:
        pass

    return None


def download_file(
    session: requests.Session,
    url: str,
    target_path: Path,
) -> bool:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    remote_size = get_remote_content_length(session, url)

    if target_path.exists():
        if remote_size is not None:
            try:
                if target_path.stat().st_size == remote_size:
                    print(f"SKIP (size match): {target_path.name}")
                    return True
            except OSError:
                pass
        else:
            print(f"SKIP (exists, unknown remote size): {target_path.name}")
            return True

    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    last_err: Optional[BaseException] = None
    for attempt in range(MAX_RETRIES):
        try:
            with request_with_retries(session, "GET", url, stream=True) as r:
                total = None
                cl = r.headers.get("Content-Length")
                if cl and cl.isdigit():
                    total = int(cl)

                # If server redirected and we now have a better filename (still keep target naming scheme)
                # we do not change target_path here.

                pbar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=target_path.name,
                    leave=False,
                )
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))
                pbar.close()

            tmp_path.replace(target_path)
            print(f"OK: {target_path.name}")
            return True
        except (requests.RequestException, OSError, RuntimeError) as e:
            last_err = e
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            backoff_sleep(attempt)
            continue

    print(f"FAIL: {url} -> {target_path} ({last_err})")
    return False


@dataclass(frozen=True)
class VideoTask:
    seq_type: str  # "fall" or "adl"
    seq_id: str    # "01", "02", ...
    camera: str    # "cam0", "cam1"
    link_url: str  # from the main table
    out_dir: Path


def find_section_table(soup: BeautifulSoup, section_title_substring: str) -> Optional[BeautifulSoup]:
    header = soup.find(lambda t: t and t.name in {"h1", "h2", "h3", "h4"} and section_title_substring.lower() in t.get_text(" ", strip=True).lower())
    if header:
        table = header.find_next("table")
        if table:
            return table
    return None


def classify_tables_fallback(tables: Sequence[BeautifulSoup]) -> Tuple[Optional[BeautifulSoup], Optional[BeautifulSoup]]:
    fall_table = None
    adl_table = None
    for t in tables:
        txt = t.get_text(" ", strip=True).lower()
        if "fall-" in txt and ("cam1" in txt or "fall sequences" in txt):
            fall_table = fall_table or t
        if "adl-" in txt or "activities of daily living" in txt:
            adl_table = adl_table or t
    # If still missing, use order heuristic
    if fall_table is None and len(tables) >= 1:
        fall_table = tables[0]
    if adl_table is None and len(tables) >= 2:
        adl_table = tables[1]
    return fall_table, adl_table


def extract_tasks_from_table(table: BeautifulSoup, seq_type: str, out_dir: Path) -> List[VideoTask]:
    tasks: List[VideoTask] = []
    if table is None:
        return tasks

    for tr in table.find_all("tr"):
        try:
            tds = tr.find_all(["td", "th"])
            if not tds:
                continue

            seq_id = parse_seq_id(tds[0].get_text(" ", strip=True))
            if not seq_id:
                continue

            # Video links: anchors with text cam0/cam1 in the row
            anchors = tr.find_all("a", href=True)
            for a in anchors:
                label = a.get_text(" ", strip=True).lower()
                if label not in {"cam0", "cam1"}:
                    continue
                href = a.get("href")
                if not href:
                    continue
                abs_url = urljoin(BASE_URL, href)
                tasks.append(VideoTask(seq_type=seq_type, seq_id=seq_id, camera=label, link_url=abs_url, out_dir=out_dir))
        except Exception:
            # fail gracefully for a bad row
            continue

    # Deduplicate tasks (some pages might have repeated anchors)
    dedup: dict[Tuple[str, str, str, str], VideoTask] = {}
    for t in tasks:
        key = (t.seq_type, t.seq_id, t.camera, t.link_url)
        dedup[key] = t
    return list(dedup.values())


def build_target_name(seq_type: str, seq_id: str, camera: str, original_name: str, index: Optional[int] = None) -> str:
    prefix = f"{seq_type}-{seq_id}_{camera}_"
    if index is not None:
        prefix = f"{seq_type}-{seq_id}_{camera}_{index:02d}_"
    return sanitize_filename(prefix + original_name)


def original_filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    name = unquote(name)
    if not name:
        name = "video"
    return sanitize_filename(name)


def scrape_main_page_for_tasks(session: requests.Session) -> Tuple[List[VideoTask], List[VideoTask]]:
    resp = request_with_retries(session, "GET", BASE_URL, stream=False)
    html = resp.text
    resp.close()

    soup = BeautifulSoup(html, "html.parser")

    fall_table = find_section_table(soup, "Fall sequences")
    adl_table = find_section_table(soup, "Activities of Daily Living")

    if fall_table is None or adl_table is None:
        tables = soup.find_all("table")
        ft, at = classify_tables_fallback(tables)
        fall_table = fall_table or ft
        adl_table = adl_table or at

    fall_tasks = extract_tasks_from_table(fall_table, "fall", FALLS_DIR)
    adl_tasks = extract_tasks_from_table(adl_table, "adl", ADLS_DIR)

    return fall_tasks, adl_tasks


def run() -> int:
    ensure_dirs()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) URFD-downloader/1.0"
        }
    )

    try:
        fall_tasks, adl_tasks = scrape_main_page_for_tasks(session)
    except Exception as e:
        print(f"ERROR: Failed to scrape main page: {e}")
        return 1

    all_tasks = fall_tasks + adl_tasks
    if not all_tasks:
        print("No video links found on the page.")
        return 2

    print(f"Found {len(fall_tasks)} fall video links and {len(adl_tasks)} ADL video links.")
    print("Starting downloads...")

    ok_count = 0
    fail_count = 0

    for task in all_tasks:
        try:
            download_urls = discover_download_urls(session, task.link_url)
            if not download_urls:
                print(f"WARN: No downloadable URLs found for {task.seq_type}-{task.seq_id} {task.camera}: {task.link_url}")
                fail_count += 1
                continue

            multiple = len(download_urls) > 1
            for i, dl_url in enumerate(download_urls, start=1):
                orig = original_filename_from_url(dl_url)
                fname = build_target_name(task.seq_type, task.seq_id, task.camera, orig, index=(i if multiple else None))
                target = task.out_dir / fname

                # If we still risk collision (same fname from multiple sources), make it unique.
                if target.exists() and multiple:
                    target = unique_path(target)

                success = download_file(session, dl_url, target)
                if success:
                    ok_count += 1
                else:
                    fail_count += 1
        except Exception as e:
            print(f"WARN: Failed task {task.seq_type}-{task.seq_id} {task.camera} ({task.link_url}): {e}")
            fail_count += 1
            continue

    print(f"Done. Successful: {ok_count}, Failed: {fail_count}")
    return 0 if fail_count == 0 else 3


if __name__ == "__main__":
    raise SystemExit(run())
