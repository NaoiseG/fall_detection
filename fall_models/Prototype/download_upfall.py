#!/usr/bin/env python3
"""
UP-Fall (HAR-UP) downloader — DOWNLOAD-AS-YOU-GO

What it does
- Opens https://sites.google.com/up.edu.mx/har-up/
- Scrolls until the Downloads widget appears (Subject/Activity/Trial buttons)
- For each Subject -> Activity -> Trial:
    - grabs the Google Drive href for "Camera1" and "Features1&0.5"
    - downloads immediately
    - unzips Camera1 zip into SubjectX/ActivityY/TrialZ/SubjectXActivityYTrialZCamera1/
    - deletes the zip after successful unzip (unless --keep-zips)

Run
  pip install playwright requests
  playwright install chromium
  python download_upfall.py --headful --out ..\\..\\Datasets\\UPFall
"""

from __future__ import annotations

import argparse
import os
import re
import time
import zipfile
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, parse_qs

import requests
from playwright.sync_api import sync_playwright, Page, Frame, Locator


START_URL_DEFAULT = "https://sites.google.com/up.edu.mx/har-up/"

# ---------------------------
# Google Drive download utils
# ---------------------------

DRIVE_FILE_ID_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
DRIVE_OPEN_ID_RE = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


def extract_drive_file_id(url: str) -> Optional[str]:
    if not url:
        return None
    m = DRIVE_FILE_ID_RE.search(url)
    if m:
        return m.group(1)
    m = DRIVE_OPEN_ID_RE.search(url)
    if m:
        return m.group(1)
    qs = parse_qs(urlparse(url).query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]
    return None


def drive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def is_zip_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def looks_like_html(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(512).lower()
        return b"<!doctype html" in head or b"<html" in head
    except Exception:
        return False


def download_drive_url(url: str, out_path: str, session: requests.Session, retries: int = 5) -> str:
    """
    Downloads a Google Drive file using the "uc?export=download&id=" endpoint.
    Handles both cookie-based confirm tokens AND HTML confirm tokens.
    """
    fid = extract_drive_file_id(url)
    if not fid:
        raise ValueError(f"Not a supported Google Drive file link: {url}")

    base = drive_direct_url(fid)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://drive.google.com/",
    }

    def get_confirm_token_from_cookies(r: requests.Response) -> Optional[str]:
        for k, v in r.cookies.items():
            if k.startswith("download_warning"):
                return v
        return None

    def get_confirm_token_from_html(html: bytes) -> Optional[str]:
        try:
            s = html.decode("utf-8", errors="ignore")
        except Exception:
            return None

        # Common patterns Drive uses
        m = re.search(r"confirm=([0-9A-Za-z_]+)", s)
        if m:
            return m.group(1)

        m = re.search(r'name="confirm"\s+value="([^"]+)"', s)
        if m:
            return m.group(1)

        return None

    last_err = None
    for attempt in range(1, retries + 1):
        tmp_path = out_path + ".part"
        try:
            # First request: may already be the file, or may be an HTML confirm page
            r = session.get(base, stream=True, allow_redirects=True, timeout=180, headers=headers)
            r.raise_for_status()

            # Buffer a small amount + rest (so we can parse HTML if needed)
            content = b""
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    content += chunk
                    # Don't let this blow memory if it's actually a huge file already:
                    # If it doesn't look like HTML after first ~512KB, assume it's the file.
                    if len(content) > 512 * 1024:
                        break
            # If we broke early, we should continue streaming to disk (likely actual file)
            broke_early = len(content) > 512 * 1024
            r.close()

            token = get_confirm_token_from_cookies(r) or get_confirm_token_from_html(content)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if token:
                # Confirmed download request
                r2 = session.get(
                    base + f"&confirm={token}",
                    stream=True,
                    allow_redirects=True,
                    timeout=180,
                    headers=headers,
                )
                r2.raise_for_status()

                with open(tmp_path, "wb") as f:
                    for chunk in r2.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                r2.close()

            else:
                # No token: either the content we buffered is the file,
                # OR Drive returned an HTML quota page without a token.
                with open(tmp_path, "wb") as f:
                    f.write(content)
                    if broke_early:
                        # Continue streaming the remainder from the first response
                        # (we need to re-request because we closed r already)
                        r3 = session.get(base, stream=True, allow_redirects=True, timeout=180, headers=headers)
                        r3.raise_for_status()
                        # Skip already-buffered bytes by discarding until we reach that point
                        remaining_to_skip = len(content)
                        for chunk in r3.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            if remaining_to_skip > 0:
                                if len(chunk) <= remaining_to_skip:
                                    remaining_to_skip -= len(chunk)
                                    continue
                                else:
                                    chunk = chunk[remaining_to_skip:]
                                    remaining_to_skip = 0
                            f.write(chunk)
                        r3.close()

            # Validate
            if looks_like_html(tmp_path):
                raise RuntimeError("Google Drive returned HTML (quota/rate-limit/warning/confirm) instead of the file.")

            if out_path.lower().endswith(".zip") and not is_zip_file(tmp_path):
                raise RuntimeError("Downloaded file is not a valid ZIP (likely throttled/partial download).")

            # Move into place
            if os.path.exists(out_path):
                os.remove(out_path)
            os.replace(tmp_path, out_path)
            return out_path

        except Exception as e:
            last_err = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            # Backoff
            time.sleep(min(30, 2 * attempt))

    raise RuntimeError(f"Failed to download after {retries} attempts: {last_err}")


# ---------------------------
# Playwright widget helpers
# ---------------------------

def list_unique_texts(loc: Locator, regex_pattern: str) -> list[str]:
    """
    Extract unique strings matching regex_pattern from all text contents
    contained in a Playwright Locator.
    """
    pat = re.compile(regex_pattern)
    seen = set()
    out: list[str] = []

    try:
        texts = loc.all_text_contents()
    except Exception:
        texts = []
        n = loc.count()
        for i in range(n):
            try:
                texts.append(loc.nth(i).text_content() or "")
            except Exception:
                pass

    for t in texts:
        if not t:
            continue
        for m in pat.findall(t):
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def wait_for_downloads_widget(page: Page, timeout_ms: int = 180_000) -> Frame:
    """
    Scroll until Subject buttons appear in any frame (main or iframe), return that frame.
    Google Sites often lazy-loads the widget only after scrolling.
    """
    deadline = time.time() + timeout_ms / 1000.0
    scroll_steps = [800, 1200, 1600, 2000]

    while time.time() < deadline:
        for fr in page.frames:
            try:
                if fr.locator("text=/Subject\\d+/").count() > 0:
                    return fr
            except Exception:
                pass

        for step in scroll_steps:
            page.mouse.wheel(0, step)
            time.sleep(0.6)

    raise TimeoutError("Downloads widget (Subject buttons) did not appear in time.")


def subject_scope(frame: Frame, subject: str) -> Locator:
    return frame.locator(f"#{subject}")


def activity_scope(frame: Frame, subject: str, activity: str) -> Locator:
    return frame.locator(f"#{subject}{activity}")


def unzip_to_folder(zip_path: str, out_folder: str) -> None:
    os.makedirs(out_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        if len(z.namelist()) == 0:
            raise RuntimeError("ZIP contains zero files.")
        z.extractall(out_folder)


def click_exact(scope, text_value: str) -> None:
    """
    Robust click for this janky Google Sites widget:
    - Try normal click quickly
    - Fall back to force click
    """
    loc = scope.get_by_text(text_value, exact=True).first
    try:
        loc.click(timeout=3000)
        return
    except Exception:
        pass
    loc.click(timeout=10_000, force=True)


def get_visible_link_href(frame: Frame, exact_text: str) -> Optional[str]:
    """
    Find href of the first visible <a> whose text matches exact_text.
    """
    candidates = frame.locator("a", has_text=exact_text)
    n = candidates.count()
    for i in range(n):
        a = candidates.nth(i)
        try:
            if a.is_visible():
                href = a.get_attribute("href")
                if href:
                    return href
        except Exception:
            continue
    return None


# ---------------------------
# Download-as-you-go pipeline
# ---------------------------

@dataclass
class Stats:
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


def handle_trial(
    frame: Frame,
    sess: requests.Session,
    out_dir: str,
    subj: str,
    act: str,
    tri: str,
    no_unzip: bool,
    keep_zips: bool,
    stats: Stats,
) -> None:
    base = f"{subj}{act}{tri}"
    combo_dir = os.path.join(out_dir, subj, act, tri)
    os.makedirs(combo_dir, exist_ok=True)

    cam1_href = get_visible_link_href(frame, "Camera1")
    feat_href = get_visible_link_href(frame, "Features1&0.5")

    # ---- Camera1 ----
    if cam1_href:
        cam_zip = os.path.join(combo_dir, f"{base}Camera1.zip")
        cam_folder = os.path.join(combo_dir, f"{base}Camera1")

        already_done = (not no_unzip and os.path.isdir(cam_folder) and os.listdir(cam_folder)) or (
            no_unzip and os.path.exists(cam_zip)
        )
        if already_done:
            print("  Camera1: already exists, skipping")
            stats.skipped += 1
        else:
            try:
                # Small cooldown helps avoid Drive throttling
                time.sleep(2.0)

                print("  Camera1: downloading zip...")
                download_drive_url(cam1_href, cam_zip, sess)
                print(f"  Camera1: saved {cam_zip}")
                stats.downloaded += 1

                if not no_unzip:
                    print("  Camera1: unzipping...")
                    unzip_to_folder(cam_zip, cam_folder)
                    print(f"  Camera1: extracted to {cam_folder}")

                    if not os.listdir(cam_folder):
                        raise RuntimeError("Extracted Camera1 folder is empty after unzip.")

                    if not keep_zips:
                        os.remove(cam_zip)
                        print("  Camera1: zip deleted")

            except Exception as e:
                stats.failed += 1
                print(f"  Camera1: FAILED ({e})")
                # Longer cooldown after a failure
                time.sleep(20.0)
    else:
        print("  Camera1: link not present")

    # ---- Features1&0.5 ----
    if feat_href:
        feat_csv = os.path.join(combo_dir, f"{base}Features1&0.5.csv")
        if os.path.exists(feat_csv):
            print("  Features1&0.5: already exists, skipping")
            stats.skipped += 1
        else:
            try:
                time.sleep(0.5)
                print("  Features1&0.5: downloading csv...")
                download_drive_url(feat_href, feat_csv, sess)
                print(f"  Features1&0.5: saved {feat_csv}")
                stats.downloaded += 1
            except Exception as e:
                stats.failed += 1
                print(f"  Features1&0.5: FAILED ({e})")
    else:
        print("  Features1&0.5: link not present")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START_URL_DEFAULT)
    ap.add_argument("--out", default="UPFall")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--max-subjects", type=int, default=None, help="Limit subjects for testing")
    ap.add_argument("--no-unzip", action="store_true", help="Do not unzip Camera1 zips")
    ap.add_argument("--keep-zips", action="store_true", help="Keep zips after unzip (default deletes)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Saving files to: {os.path.abspath(args.out)}")

    stats = Stats()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        page = browser.new_page()
        page.goto(args.start, wait_until="domcontentloaded", timeout=120_000)

        frame = wait_for_downloads_widget(page, timeout_ms=180_000)
        print("Downloads widget found, starting downloads...")

        sess = requests.Session()

        # Import Playwright browser cookies into requests session (helps reduce Drive blocking)
        try:
            cookies = page.context.cookies()
            for c in cookies:
                sess.cookies.set(
                    c["name"],
                    c["value"],
                    domain=c.get("domain"),
                    path=c.get("path"),
                )
        except Exception:
            pass

        subjects = list_unique_texts(frame.locator("text=/Subject\\d+/"), r"Subject\d+")
        if not subjects:
            print("ERROR: No subjects found.")
            return 1
        if args.max_subjects is not None:
            subjects = subjects[: args.max_subjects]

        for subj in subjects:
            page.mouse.wheel(0, 300)
            time.sleep(0.2)

            print(f"\n=== {subj} ===")
            click_exact(frame, subj)

            frame.locator(f"#{subj}").wait_for(state="visible", timeout=20_000)
            time.sleep(0.3)

            subj_sc = subject_scope(frame, subj)
            activities = list_unique_texts(subj_sc.locator("text=/Activity\\d+/"), r"Activity\d+")
            if not activities:
                time.sleep(1.0)
                activities = list_unique_texts(subj_sc.locator("text=/Activity\\d+/"), r"Activity\d+")

            for act in activities:
                print(f"\n  -- {act} --")
                click_exact(subj_sc, act)
                time.sleep(0.3)

                act_sc = activity_scope(frame, subj, act)
                act_sc.wait_for(state="visible", timeout=20_000)

                trials = list_unique_texts(act_sc.locator("text=/Trial\\d+/"), r"Trial\d+")
                if not trials:
                    time.sleep(1.0)
                    trials = list_unique_texts(act_sc.locator("text=/Trial\\d+/"), r"Trial\d+")

                for tri in trials:
                    print(f"\n  [{subj}{act}{tri}]")
                    click_exact(act_sc, tri)
                    time.sleep(0.6)

                    handle_trial(
                        frame=frame,
                        sess=sess,
                        out_dir=args.out,
                        subj=subj,
                        act=act,
                        tri=tri,
                        no_unzip=args.no_unzip,
                        keep_zips=args.keep_zips,
                        stats=stats,
                    )

        browser.close()

    print("\nDone.")
    print(f"Downloaded: {stats.downloaded}  Skipped: {stats.skipped}  Failed: {stats.failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
