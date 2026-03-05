#!/usr/bin/env python3
"""
HAR-UP Camera2 Downloader (Playwright, restart-safe)

Requirements
1. Install packages:
   pip install playwright requests
   playwright install chromium

2. Run (headless by default):
   python dataset_helpers/download_harup_camera2.py

3. Run with visible browser:
   python dataset_helpers/download_harup_camera2.py --headful

4. Resume behavior:
   For each Subject/Activity/Trial, if
   ROOT/Subject{n}/Activity{m}/Trial{k}/Subject{n}Activity{m}Trial{k}Camera2/
   already exists and contains files, the item is skipped.

5. Logs:
   - Text log (append mode): download_log.txt
   - Failure CSV (append mode): failed_downloads.csv
   Defaults are inside the dataset root, configurable with CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union
from urllib.parse import parse_qs, urljoin, urlparse

import requests

from playwright.sync_api import (
    BrowserContext,
    Frame,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


START_URL_DEFAULT = "https://sites.google.com/up.edu.mx/har-up/"
ROOT_DEFAULT = Path(r"C:\Users\Student\Documents\fall_detection\Datasets\UPFall")

RATE_LIMIT_TEXT_PATTERNS = (
    "too many requests",
    "rate limit",
    "error 429",
    "http 429",
    "status code 429",
)

DRIVE_FILE_ID_RE = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
DRIVE_OPEN_ID_RE = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")


FrameHost = Union[Page, Frame]


def all_hosts(page: Page) -> list[FrameHost]:
    return [page] + page.frames


@dataclass(frozen=True, order=True)
class Combo:
    subject: int
    activity: int
    trial: int

    @property
    def subject_label(self) -> str:
        return f"Subject{self.subject}"

    @property
    def activity_label(self) -> str:
        return f"Activity{self.activity}"

    @property
    def trial_label(self) -> str:
        return f"Trial{self.trial}"

    @property
    def base_name(self) -> str:
        return f"{self.subject_label}{self.activity_label}{self.trial_label}"

    def trial_dir(self, root: Path) -> Path:
        return root / self.subject_label / self.activity_label / self.trial_label

    def camera2_dir(self, root: Path) -> Path:
        return self.trial_dir(root) / f"{self.base_name}Camera2"

    def short(self) -> str:
        return f"{self.subject_label}/{self.activity_label}/{self.trial_label}"


@dataclass
class Stats:
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass
class Config:
    start_url: str
    root: Path
    staging_dir: Path
    log_path: Path
    failed_csv: Path
    headless: bool
    max_attempts: int
    backoff_base: float
    backoff_max: float
    backoff_jitter: float
    discovery_timeout_s: float
    navigation_timeout_ms: int
    action_timeout_ms: int
    immediate_download_timeout_ms: int
    followup_download_timeout_ms: int
    max_subjects: Optional[int]
    max_combos: Optional[int]


class RetryableDownloadError(RuntimeError):
    pass


class RateLimitError(RetryableDownloadError):
    pass


class DualLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, combo: Optional[Combo] = None, level: str = "INFO") -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        combo_text = f" [{combo.short()}]" if combo else ""
        line = f"{ts} [{level}]{combo_text} {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "download.zip"


def cleanup_path(path: Path) -> None:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def folder_has_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(p.is_file() for p in path.rglob("*"))


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def is_exact_camera2_label(text: str) -> bool:
    return re.fullmatch(r"camera\s*2", normalize_space(text).strip(), re.IGNORECASE) is not None


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
    vals = qs.get("id", [])
    return vals[0] if vals else None


def drive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def looks_like_html(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(512).lower()
        return b"<html" in head or b"<!doctype html" in head
    except Exception:
        return False


def is_zip_magic(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def copy_playwright_cookies_to_session(context: BrowserContext, session: requests.Session) -> None:
    try:
        cookies = context.cookies()
    except Exception:
        cookies = []
    for c in cookies:
        try:
            session.cookies.set(
                c.get("name", ""),
                c.get("value", ""),
                domain=c.get("domain"),
                path=c.get("path"),
            )
        except Exception:
            continue


def download_drive_url(url: str, out_path: Path, session: requests.Session, retries: int = 4) -> Path:
    file_id = extract_drive_file_id(url)
    if not file_id:
        raise RetryableDownloadError(f"Could not parse Google Drive file id from url: {url}")

    base = drive_direct_url(file_id)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://drive.google.com/",
    }

    def token_from_cookies(resp: requests.Response) -> Optional[str]:
        for k, v in resp.cookies.items():
            if str(k).startswith("download_warning"):
                return v
        return None

    def token_from_html(raw: bytes) -> Optional[str]:
        txt = raw.decode("utf-8", errors="ignore")
        m = re.search(r"confirm=([0-9A-Za-z_]+)", txt)
        if m:
            return m.group(1)
        m = re.search(r'name="confirm"\s+value="([^"]+)"', txt)
        if m:
            return m.group(1)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_err: Optional[Exception] = None
    for _attempt in range(1, retries + 1):
        part = out_path.with_suffix(out_path.suffix + ".part")
        cleanup_path(part)
        try:
            r1 = session.get(base, stream=True, timeout=180, allow_redirects=True, headers=headers)
            r1.raise_for_status()

            buf = b""
            for chunk in r1.iter_content(chunk_size=128 * 1024):
                if chunk:
                    buf += chunk
                    if len(buf) > 512 * 1024:
                        break
            r1.close()

            token = token_from_cookies(r1) or token_from_html(buf)
            url2 = base + (f"&confirm={token}" if token else "")
            r2 = session.get(url2, stream=True, timeout=180, allow_redirects=True, headers=headers)
            r2.raise_for_status()
            with part.open("wb") as f:
                for chunk in r2.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            r2.close()

            if looks_like_html(part):
                raise RetryableDownloadError("Drive returned HTML instead of file (confirm/quota/rate-limit).")
            if not is_zip_magic(part):
                raise RetryableDownloadError("Downloaded file is not ZIP content.")

            if out_path.exists():
                out_path.unlink(missing_ok=True)
            part.replace(out_path)
            return out_path
        except Exception as exc:
            last_err = exc
            cleanup_path(part)
            time.sleep(1.5)

    raise RetryableDownloadError(f"Google Drive direct download failed: {last_err}")


def extract_numeric_labels(scope: Union[FrameHost, Locator], prefix: str, visible_only: bool = False) -> list[int]:
    pattern = re.compile(rf"{prefix}\s*(\d+)", re.IGNORECASE)
    loc = scope.locator(f"text=/{prefix}\\s*\\d+/i")
    numbers: set[int] = set()

    count = min(loc.count(), 400)
    for i in range(count):
        item = loc.nth(i)
        try:
            if visible_only and not item.is_visible():
                continue
            text = item.text_content() or ""
        except Exception:
            continue
        for raw in pattern.findall(text):
            numbers.add(int(raw))

    if not numbers:
        try:
            texts = loc.all_text_contents()
        except Exception:
            texts = []
        for text in texts:
            for raw in pattern.findall(text or ""):
                numbers.add(int(raw))

    return sorted(numbers)


def extract_numeric_labels_any(page: Page, prefix: str, visible_only: bool = False) -> list[int]:
    numbers: set[int] = set()
    for host in all_hosts(page):
        try:
            numbers.update(extract_numeric_labels(host, prefix, visible_only=visible_only))
        except Exception:
            continue
    return sorted(numbers)


def wait_for_labels_any(page: Page, prefix: str, timeout_s: float, visible_only: bool) -> list[int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        labels = extract_numeric_labels_any(page, prefix, visible_only=visible_only)
        if labels:
            return labels
        time.sleep(0.2)
    return []


def click_locator(locator: Locator, timeout_ms: int) -> None:
    last_exc: Optional[Exception] = None
    for force in (False, True):
        try:
            locator.scroll_into_view_if_needed(timeout=timeout_ms)
        except Exception:
            pass
        try:
            locator.click(timeout=timeout_ms, force=force)
            return
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to click locator.")


def click_label(scope: Union[FrameHost, Locator], label: str, timeout_ms: int) -> None:
    m = re.match(r"^(Subject|Activity|Trial)(\d+)$", label)
    if not m:
        raise ValueError(f"Unsupported label format: {label}")
    prefix, number = m.group(1), m.group(2)

    exact_re = re.compile(rf"^{prefix}\s*{number}$", re.IGNORECASE)
    candidates = scope.get_by_text(exact_re)
    count = candidates.count()
    for i in range(min(count, 120)):
        cand = candidates.nth(i)
        try:
            click_locator(cand, timeout_ms=timeout_ms)
            return
        except Exception:
            continue

    loose = scope.get_by_text(re.compile(rf"{prefix}\s*{number}", re.IGNORECASE))
    loose_count = loose.count()
    for i in range(min(loose_count, 120)):
        cand = loose.nth(i)
        try:
            click_locator(cand, timeout_ms=timeout_ms)
            return
        except Exception:
            continue

    raise RetryableDownloadError(f"Could not click {label}.")


def click_label_any(page: Page, label: str, timeout_ms: int) -> None:
    for host in all_hosts(page):
        try:
            click_label(host, label, timeout_ms=timeout_ms)
            return
        except Exception:
            continue
    raise RetryableDownloadError(f"Could not click {label} on any page/frame.")


def has_label_any(page: Page, label: str) -> bool:
    m = re.match(r"^(Subject|Activity|Trial)(\d+)$", label)
    if not m:
        return False
    prefix, number = m.group(1), m.group(2)
    exact_re = re.compile(rf"^{prefix}\s*{number}$", re.IGNORECASE)
    loose_re = re.compile(rf"{prefix}\s*{number}", re.IGNORECASE)

    for host in all_hosts(page):
        try:
            if host.get_by_text(exact_re).count() > 0:
                return True
            if host.get_by_text(loose_re).count() > 0:
                return True
        except Exception:
            continue
    return False


def wait_and_click_label_any(
    page: Page,
    label: str,
    action_timeout_ms: int,
    wait_timeout_s: float,
) -> None:
    deadline = time.time() + wait_timeout_s
    last_error: Optional[Exception] = None

    while time.time() < deadline:
        try:
            click_label_any(page, label, timeout_ms=action_timeout_ms)
            return
        except Exception as exc:
            last_error = exc
            # Keep revealing lazy UI sections while waiting for the target label.
            try:
                page.mouse.wheel(0, 300)
            except Exception:
                pass
            time.sleep(0.25)

    raise RetryableDownloadError(f"Could not click {label}: {last_error}")


def scope_for_subject(host: FrameHost, subject_label: str) -> Locator:
    by_id = host.locator(f"#{subject_label}")
    if by_id.count() > 0:
        return by_id
    return host.locator("body")


def scope_for_activity(host: FrameHost, subject_label: str, activity_label: str) -> Locator:
    by_id = host.locator(f"#{subject_label}{activity_label}")
    if by_id.count() > 0:
        return by_id
    return host.locator("body")


def wait_for_widget_host(page: Page, timeout_s: float) -> FrameHost:
    deadline = time.time() + timeout_s
    scroll_steps = (650, 900, 1200)

    while time.time() < deadline:
        for host in all_hosts(page):
            try:
                if host.locator("text=/Subject\\s*\\d+/i").count() > 0:
                    return host
            except Exception:
                continue

        for step in scroll_steps:
            page.mouse.wheel(0, step)
            time.sleep(0.35)

    raise RetryableDownloadError("Timed out waiting for Subject selector widget.")


def is_rate_limited_text(text: str) -> bool:
    low = (text or "").lower()
    return any(token in low for token in RATE_LIMIT_TEXT_PATTERNS)


def assert_not_rate_limited(page: Page) -> None:
    try:
        body_text = page.locator("body").first.inner_text(timeout=4000) or ""
    except Exception:
        body_text = ""
    if is_rate_limited_text(body_text):
        raise RateLimitError("Page indicates rate limit / too many requests.")


def compute_backoff(attempt: int, base: float, max_delay: float, jitter: float) -> float:
    delay = min(max_delay, base * (2 ** max(0, attempt - 1)))
    if jitter > 0:
        factor = random.uniform(max(0.0, 1.0 - jitter), 1.0 + jitter)
        delay *= factor
    return max(0.0, min(max_delay, delay))


def write_failed_csv(csv_path: Path, combo: Combo, reason: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "subject", "activity", "trial", "reason"])
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                combo.subject,
                combo.activity,
                combo.trial,
                reason,
            ]
        )


def ensure_failed_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "subject", "activity", "trial", "reason"])


def normalize_exception_message(exc: Exception) -> str:
    text = str(exc).strip()
    return text if text else exc.__class__.__name__


def goto_start(page: Page, cfg: Config) -> None:
    response = page.goto(
        cfg.start_url,
        wait_until="domcontentloaded",
        timeout=cfg.navigation_timeout_ms,
    )
    if response is not None and response.status == 429:
        raise RateLimitError("HTTP 429 when opening HAR-UP page.")
    assert_not_rate_limited(page)


def discover_combos_from_anchor_metadata(page: Page) -> set[Combo]:
    pattern = re.compile(
        r"Subject\D*(\d+)\D*Activity\D*(\d+)\D*Trial\D*(\d+)\D*Camera\D*2",
        re.IGNORECASE,
    )
    found: set[Combo] = set()

    for host in all_hosts(page):
        try:
            anchors = host.locator("a")
            count = min(anchors.count(), 2000)
        except Exception:
            continue

        for i in range(count):
            a = anchors.nth(i)
            try:
                text = (a.text_content() or "").strip()
            except Exception:
                text = ""
            try:
                href = (a.get_attribute("href") or "").strip()
            except Exception:
                href = ""

            blob = f"{text} {href}"
            for s_raw, a_raw, t_raw in pattern.findall(blob):
                found.add(Combo(subject=int(s_raw), activity=int(a_raw), trial=int(t_raw)))

    return found


def discover_combos_from_local_structure(root: Path) -> set[Combo]:
    found: set[Combo] = set()
    if not root.exists():
        return found

    subj_re = re.compile(r"^Subject(\d+)$", re.IGNORECASE)
    act_re = re.compile(r"^Activity(\d+)$", re.IGNORECASE)
    tri_re = re.compile(r"^Trial(\d+)$", re.IGNORECASE)

    for subj_dir in root.glob("Subject*"):
        if not subj_dir.is_dir():
            continue
        sm = subj_re.match(subj_dir.name)
        if not sm:
            continue
        s = int(sm.group(1))

        for act_dir in subj_dir.glob("Activity*"):
            if not act_dir.is_dir():
                continue
            am = act_re.match(act_dir.name)
            if not am:
                continue
            a = int(am.group(1))

            for tri_dir in act_dir.glob("Trial*"):
                if not tri_dir.is_dir():
                    continue
                tm = tri_re.match(tri_dir.name)
                if not tm:
                    continue
                t = int(tm.group(1))
                found.add(Combo(subject=s, activity=a, trial=t))

    return found


def discover_combinations(page: Page, cfg: Config, logger: DualLogger) -> list[Combo]:
    goto_start(page, cfg)
    wait_for_widget_host(page, timeout_s=cfg.discovery_timeout_s)

    local_found = discover_combos_from_local_structure(cfg.root)
    if local_found:
        logger.log(f"Local dataset structure discovered {len(local_found)} combinations.")

    anchor_found = discover_combos_from_anchor_metadata(page)
    if anchor_found:
        logger.log(f"Anchor metadata discovered {len(anchor_found)} combinations.")

    found: set[Combo] = set(local_found) | set(anchor_found)
    if not found:
        logger.log("Local/anchor discovery found no combinations; trying UI traversal.", level="WARN")
        subjects = extract_numeric_labels_any(page, "Subject", visible_only=False)
        if not subjects:
            subjects = extract_numeric_labels_any(page, "Subject", visible_only=True)
        if cfg.max_subjects is not None:
            subjects = subjects[: cfg.max_subjects]

        if not subjects:
            raise RetryableDownloadError("No Subjects discovered in HAR-UP widget.")

        logger.log(f"UI discovered {len(subjects)} subjects.")
        for s in subjects:
            subject = f"Subject{s}"
            try:
                click_label_any(page, subject, timeout_ms=cfg.action_timeout_ms)
                time.sleep(0.35)
            except Exception as exc:
                logger.log(f"Skipping subject {subject} due to click error: {exc}", level="WARN")
                continue

            activities = wait_for_labels_any(page, "Activity", timeout_s=4.0, visible_only=True)
            if not activities:
                activities = wait_for_labels_any(page, "Activity", timeout_s=4.0, visible_only=False)
            if not activities:
                logger.log(f"No activities visible after selecting {subject}.", level="WARN")
                continue

            for a in activities:
                activity = f"Activity{a}"
                try:
                    click_label_any(page, activity, timeout_ms=cfg.action_timeout_ms)
                except Exception as exc:
                    logger.log(
                        f"Skipping {subject}/{activity} due to click error: {exc}",
                        level="WARN",
                    )
                    continue

                time.sleep(0.25)
                trials = wait_for_labels_any(page, "Trial", timeout_s=3.0, visible_only=True)
                if not trials:
                    trials = wait_for_labels_any(page, "Trial", timeout_s=3.0, visible_only=False)
                if not trials:
                    logger.log(f"No trials visible after selecting {subject}/{activity}.", level="WARN")
                    continue

                for t in trials:
                    found.add(Combo(subject=s, activity=a, trial=t))

    if cfg.max_subjects is not None and found:
        keep_subjects = sorted({c.subject for c in found})[: cfg.max_subjects]
        keep_set = set(keep_subjects)
        found = {c for c in found if c.subject in keep_set}

    all_combos = sorted(found)
    if not all_combos:
        raise RetryableDownloadError("No Subject/Activity/Trial combinations discovered.")

    if cfg.max_combos is not None:
        limited = all_combos[: cfg.max_combos]
        logger.log(
            f"Discovered {len(all_combos)} total combinations; limiting to {len(limited)} due to --max-combos.",
        )
        return limited

    logger.log(f"Discovered {len(all_combos)} total Subject/Activity/Trial combinations.")
    return all_combos


def find_host_with_camera2(page: Page, timeout_s: float) -> FrameHost:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for host in all_hosts(page):
            try:
                _ = find_camera2_locator(host)
                return host
            except Exception:
                continue
        time.sleep(0.2)
    raise RetryableDownloadError("Camera2 control did not appear after selecting trial.")


def select_combo(page: Page, cfg: Config, combo: Combo, logger: DualLogger) -> FrameHost:
    goto_start(page, cfg)
    wait_for_widget_host(page, timeout_s=cfg.discovery_timeout_s)

    wait_and_click_label_any(
        page=page,
        label=combo.subject_label,
        action_timeout_ms=cfg.action_timeout_ms,
        wait_timeout_s=20.0,
    )
    time.sleep(0.3)

    # Activity buttons may appear lazily after subject selection.
    deadline = time.time() + 12.0
    while time.time() < deadline:
        if has_label_any(page, combo.activity_label):
            break
        time.sleep(0.25)

    wait_and_click_label_any(
        page=page,
        label=combo.activity_label,
        action_timeout_ms=cfg.action_timeout_ms,
        wait_timeout_s=20.0,
    )
    time.sleep(0.3)

    # Trial buttons may appear lazily after activity selection.
    deadline = time.time() + 12.0
    while time.time() < deadline:
        if has_label_any(page, combo.trial_label):
            break
        time.sleep(0.25)

    wait_and_click_label_any(
        page=page,
        label=combo.trial_label,
        action_timeout_ms=cfg.action_timeout_ms,
        wait_timeout_s=20.0,
    )

    logger.log("Selected Subject -> Activity -> Trial path.", combo)
    time.sleep(0.4)
    return find_host_with_camera2(page, timeout_s=10.0)


def find_camera2_locator(host: FrameHost) -> Locator:
    for cand in find_camera2_candidates(host):
        return cand
    raise RetryableDownloadError("Camera2 control was not found for this trial.")


def find_camera2_candidates(host: FrameHost) -> list[Locator]:
    selectors = ["a", "button", "[role='button']", "text=/Camera\\s*2/i"]
    out: list[Locator] = []
    seen: set[str] = set()
    for selector in selectors:
        loc = host.locator(selector)
        count = min(loc.count(), 200)
        for i in range(count):
            cand = loc.nth(i)
            try:
                if not cand.is_visible():
                    continue
                text = normalize_space(cand.text_content() or "")
                if not is_exact_camera2_label(text):
                    continue
                sig = candidate_description(cand)
            except Exception:
                continue
            if sig in seen:
                continue
            seen.add(sig)
            out.append(cand)
    return out


def _candidate_score(text: str, href: str) -> int:
    blob = f"{text} {href}".lower()
    score = 0
    if "camera2_of" in blob or "camera 2_of" in blob or "camera2 of" in blob:
        return -100
    if "camera2" in blob or "camera 2" in blob:
        score += 120
    if "download" in blob:
        score += 45
    if ".zip" in blob:
        score += 35
    if "export=download" in blob:
        score += 30
    if "drive.google.com" in blob:
        score += 20
    return score


def iter_download_candidates(page: Page, prefer_camera2: bool = False) -> Sequence[Locator]:
    selectors = [
        "a[href]",
        "button",
        "[role='button']",
    ]
    scopes: list[FrameHost] = [page] + page.frames
    ranked: list[tuple[int, Locator]] = []
    seen: set[str] = set()

    for scope in scopes:
        for selector in selectors:
            loc = scope.locator(selector)
            count = min(loc.count(), 80)
            for i in range(count):
                cand = loc.nth(i)
                try:
                    if not cand.is_visible():
                        continue
                    text = normalize_space((cand.text_content() or "").strip())
                    href = (cand.get_attribute("href") or "").strip()
                except Exception:
                    continue

                if not href and len(text) > 120:
                    continue

                if href:
                    resolved = urljoin(page.url, href)
                    host = (urlparse(resolved).netloc or "").lower()
                    allowed_hosts = {
                        "sites.google.com",
                        "drive.google.com",
                        "drive.usercontent.google.com",
                        "docs.google.com",
                    }
                    if host and host not in allowed_hosts:
                        continue

                blob = f"{text} {href}".lower()
                if "camera2_of" in blob or "camera 2_of" in blob or "camera2 of" in blob:
                    continue

                score = _candidate_score(text=text, href=href)
                if prefer_camera2 and not is_exact_camera2_label(text):
                    continue

                key = f"{text}|{href}"
                if key in seen:
                    continue
                seen.add(key)
                ranked.append((score, cand))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [cand for _score, cand in ranked]


def candidate_description(locator: Locator) -> str:
    try:
        text = (locator.text_content() or "").strip()
    except Exception:
        text = ""
    try:
        href = (locator.get_attribute("href") or "").strip()
    except Exception:
        href = ""
    if text and href:
        return f"text='{text}' href='{href}'"
    if text:
        return f"text='{text}'"
    if href:
        return f"href='{href}'"
    return "unlabeled clickable"


def get_camera2_href(host: FrameHost, base_url: str) -> Optional[str]:
    for cand in find_camera2_candidates(host):
        try:
            href = (cand.get_attribute("href") or "").strip()
        except Exception:
            href = ""
        if not href:
            continue
        return urljoin(base_url, href)
    return None


def download_camera2_direct_from_href(
    context: BrowserContext,
    href: str,
    cfg: Config,
    combo: Combo,
    logger: DualLogger,
) -> Path:
    suggested = f"{combo.base_name}Camera2.zip"
    out_zip = cfg.staging_dir / f"{combo.base_name}_{int(time.time())}_{sanitize_filename(suggested)}"

    session = requests.Session()
    copy_playwright_cookies_to_session(context, session)
    download_drive_url(href, out_zip, session=session, retries=4)

    if not out_zip.exists() or out_zip.stat().st_size <= 0:
        raise RetryableDownloadError("Direct Camera2 download produced empty file.")
    if not is_zip_magic(out_zip):
        raise RetryableDownloadError("Direct Camera2 download is not ZIP content.")

    logger.log(f"Direct Camera2 ZIP saved to staging: {out_zip}", combo)
    return out_zip


def save_download_to_staging(download, cfg: Config, combo: Combo, logger: DualLogger) -> Path:
    cfg.staging_dir.mkdir(parents=True, exist_ok=True)
    suggested = sanitize_filename(download.suggested_filename or f"{combo.base_name}Camera2.zip")
    if not suggested.lower().endswith(".zip"):
        suggested += ".zip"
    zip_path = cfg.staging_dir / f"{combo.base_name}_{int(time.time())}_{suggested}"

    download.save_as(str(zip_path))
    if not zip_path.exists():
        raise RetryableDownloadError("Playwright reported download but no file was saved.")
    if zip_path.stat().st_size <= 0:
        raise RetryableDownloadError("Downloaded ZIP is zero bytes.")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m and not m.endswith("/")]
            if not members:
                raise RetryableDownloadError("Downloaded ZIP contains no files.")
            bad_member = zf.testzip()
            if bad_member:
                raise RetryableDownloadError(f"Downloaded ZIP is corrupted (bad member: {bad_member}).")
    except zipfile.BadZipFile as exc:
        raise RetryableDownloadError(f"Downloaded file is not a valid ZIP: {exc}") from exc

    logger.log(f"ZIP saved to staging: {zip_path}", combo)
    return zip_path


def extract_zip_flatten(zip_path: Path, target_camera2_dir: Path, combo: Combo, logger: DualLogger) -> None:
    tmp_extract = target_camera2_dir.parent / f".{target_camera2_dir.name}_tmp_extract"
    cleanup_path(tmp_extract)
    cleanup_path(target_camera2_dir)

    tmp_extract.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_extract)

        top_entries = list(tmp_extract.iterdir())
        src_root = tmp_extract
        if len(top_entries) == 1 and top_entries[0].is_dir():
            src_root = top_entries[0]

        target_camera2_dir.mkdir(parents=True, exist_ok=True)
        for item in src_root.iterdir():
            shutil.move(str(item), str(target_camera2_dir / item.name))

        if not folder_has_files(target_camera2_dir):
            raise RetryableDownloadError("Extraction finished but Camera2 folder has no files.")

        logger.log(f"Extraction complete: {target_camera2_dir}", combo)
    except Exception as exc:
        cleanup_path(target_camera2_dir)
        raise RetryableDownloadError(f"Extraction failed: {exc}") from exc
    finally:
        cleanup_path(tmp_extract)


def wait_for_context_download(context: BrowserContext, timeout_ms: int):
    try:
        return context.wait_for_event("download", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        return None


def close_all_context_pages(context: BrowserContext) -> None:
    for p in list(context.pages):
        try:
            p.close()
        except Exception:
            pass


def open_new_pages_since(context: BrowserContext, old_ids: set[int]) -> list[Page]:
    return [p for p in context.pages if id(p) not in old_ids and not p.is_closed()]


def resolve_followup_download(
    context: BrowserContext,
    initial_pages: Sequence[Page],
    cfg: Config,
    combo: Combo,
    logger: DualLogger,
) -> object:
    queue: list[Page] = [p for p in initial_pages if not p.is_closed()]
    visited_states: set[tuple[int, str]] = set()
    known_page_ids: set[int] = {id(p) for p in queue}

    while queue and len(visited_states) <= 30:
        page = queue.pop(0)
        state_key = (id(page), page.url)
        if state_key in visited_states:
            continue
        visited_states.add(state_key)

        try:
            page.wait_for_load_state("domcontentloaded", timeout=cfg.navigation_timeout_ms)
        except Exception:
            pass

        assert_not_rate_limited(page)
        logger.log(f"Scanning page for actual ZIP link: {page.url}", combo)

        existing = wait_for_context_download(context, timeout_ms=1500)
        if existing is not None:
            return existing

        on_main_har_page = "sites.google.com/up.edu.mx/har-up" in page.url.lower()
        candidates = list(iter_download_candidates(page, prefer_camera2=on_main_har_page))

        per_candidate_timeout = min(cfg.followup_download_timeout_ms, 12_000)
        for cand in candidates:
            desc = candidate_description(cand)
            logger.log(f"Trying follow-up candidate: {desc}", combo)
            before_url = page.url
            try:
                with context.expect_event("download", timeout=per_candidate_timeout) as info:
                    click_locator(cand, timeout_ms=cfg.action_timeout_ms)
                return info.value
            except PlaywrightTimeoutError:
                new_pages = open_new_pages_since(context, known_page_ids)
                queue.extend(new_pages)
                known_page_ids.update(id(p) for p in new_pages)
                after_url = page.url if not page.is_closed() else before_url
                # If the click changed URL or opened a tab, immediately rescan new page state
                # instead of continuing through stale candidates from the previous page view.
                if (after_url != before_url) or new_pages:
                    if not page.is_closed():
                        queue.append(page)
                    break
                continue
            except Exception:
                continue

        new_pages = open_new_pages_since(context, known_page_ids)
        queue.extend(new_pages)
        known_page_ids.update(id(p) for p in new_pages)

    raise RetryableDownloadError("Could not find the real Camera2 download link after Camera2 click.")


def download_camera2_zip(context: BrowserContext, page: Page, host: FrameHost, cfg: Config, combo: Combo, logger: DualLogger) -> Path:
    href = get_camera2_href(host, base_url=page.url)
    if href:
        try:
            logger.log(f"Trying direct Camera2 download from href: {href}", combo)
            return download_camera2_direct_from_href(
                context=context,
                href=href,
                cfg=cfg,
                combo=combo,
                logger=logger,
            )
        except Exception as exc:
            logger.log(f"Direct Camera2 download failed, falling back to UI flow: {exc}", combo, level="WARN")

    camera2_candidates = find_camera2_candidates(host)
    if not camera2_candidates:
        camera2_candidates = [find_camera2_locator(host)]

    last_error: Optional[Exception] = None
    max_candidates = min(len(camera2_candidates), 8)

    for idx, camera2 in enumerate(camera2_candidates[:max_candidates], start=1):
        old_page_ids = {id(p) for p in context.pages}
        start_url = page.url
        logger.log(
            f"Clicking Camera2 candidate {idx}/{max_candidates}: {candidate_description(camera2)}",
            combo,
        )
        try:
            with context.expect_event("download", timeout=cfg.immediate_download_timeout_ms) as info:
                click_locator(camera2, timeout_ms=cfg.action_timeout_ms)
            logger.log("Camera2 triggered immediate download.", combo)
            return save_download_to_staging(info.value, cfg=cfg, combo=combo, logger=logger)
        except PlaywrightTimeoutError:
            logger.log("No immediate download event. Trying follow-up flow.", combo)
        except Exception as exc:
            last_error = exc
            logger.log(f"Camera2 click failed: {exc}", combo, level="WARN")
            continue

        new_pages = open_new_pages_since(context, old_page_ids)
        candidate_pages: list[Page] = []
        candidate_pages.extend(new_pages)
        if page.url != start_url:
            candidate_pages.append(page)
        if page not in candidate_pages:
            candidate_pages.append(page)

        try:
            download = resolve_followup_download(
                context=context,
                initial_pages=candidate_pages,
                cfg=cfg,
                combo=combo,
                logger=logger,
            )
            return save_download_to_staging(download, cfg=cfg, combo=combo, logger=logger)
        except Exception as exc:
            last_error = exc
            logger.log(f"Camera2 candidate follow-up failed: {exc}", combo, level="WARN")

    if last_error is not None:
        raise RetryableDownloadError(f"All Camera2 candidates failed: {last_error}") from last_error
    raise RetryableDownloadError("No Camera2 candidate produced a downloadable ZIP.")


def process_combo(context: BrowserContext, combo: Combo, cfg: Config, logger: DualLogger, stats: Stats) -> None:
    target_dir = combo.camera2_dir(cfg.root)
    trial_dir = combo.trial_dir(cfg.root)
    trial_dir.mkdir(parents=True, exist_ok=True)

    if folder_has_files(target_dir):
        logger.log("Skip: Camera2 folder already exists and contains files.", combo)
        stats.skipped += 1
        return

    if target_dir.exists():
        logger.log("Removing empty/partial Camera2 folder before retry.", combo)
        cleanup_path(target_dir)

    for attempt in range(1, cfg.max_attempts + 1):
        zip_path: Optional[Path] = None
        page: Optional[Page] = None
        try:
            logger.log(f"Attempt {attempt}/{cfg.max_attempts} started.", combo)
            page = context.new_page()
            host = select_combo(page=page, cfg=cfg, combo=combo, logger=logger)
            zip_path = download_camera2_zip(context=context, page=page, host=host, cfg=cfg, combo=combo, logger=logger)
            extract_zip_flatten(zip_path=zip_path, target_camera2_dir=target_dir, combo=combo, logger=logger)
            logger.log("Success: Camera2 downloaded, extracted, ZIP removed.", combo)
            stats.downloaded += 1
            return
        except Exception as exc:
            reason = normalize_exception_message(exc)
            logger.log(f"Attempt {attempt} failed: {reason}", combo, level="WARN")
            cleanup_path(target_dir)

            if attempt >= cfg.max_attempts:
                write_failed_csv(cfg.failed_csv, combo, reason)
                logger.log("Marked as failed and continuing.", combo, level="ERROR")
                stats.failed += 1
                return

            delay = compute_backoff(
                attempt=attempt,
                base=cfg.backoff_base,
                max_delay=cfg.backoff_max,
                jitter=cfg.backoff_jitter,
            )
            logger.log(f"Retrying in {delay:.1f}s (exponential backoff).", combo, level="WARN")
            time.sleep(delay)
        finally:
            if zip_path is not None:
                cleanup_path(zip_path)
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass
            close_all_context_pages(context)


def build_config_from_args(args: argparse.Namespace) -> Config:
    root = Path(args.root).expanduser()
    staging = Path(args.staging_dir).expanduser() if args.staging_dir else (root / "_staging_camera2")
    log_path = Path(args.log_path).expanduser() if args.log_path else (root / "download_log.txt")
    failed_csv = Path(args.failed_csv).expanduser() if args.failed_csv else (root / "failed_downloads.csv")

    root.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)

    return Config(
        start_url=args.start_url,
        root=root,
        staging_dir=staging,
        log_path=log_path,
        failed_csv=failed_csv,
        headless=not args.headful,
        max_attempts=args.max_attempts,
        backoff_base=args.backoff_base,
        backoff_max=args.backoff_max,
        backoff_jitter=args.backoff_jitter,
        discovery_timeout_s=args.discovery_timeout,
        navigation_timeout_ms=args.navigation_timeout,
        action_timeout_ms=args.action_timeout,
        immediate_download_timeout_ms=args.immediate_download_timeout,
        followup_download_timeout_ms=args.followup_download_timeout,
        max_subjects=args.max_subjects,
        max_combos=args.max_combos,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download and extract all HAR-UP Camera2 ZIPs.")
    ap.add_argument("--start-url", default=START_URL_DEFAULT, help="HAR-UP page URL")
    ap.add_argument("--root", default=str(ROOT_DEFAULT), help="Local UPFall dataset root")
    ap.add_argument("--headful", action="store_true", help="Run with visible browser window")
    ap.add_argument("--staging-dir", default="", help="Temporary staging directory for ZIP downloads")
    ap.add_argument("--log-path", default="", help="Path for append-mode text log")
    ap.add_argument("--failed-csv", default="", help="Path for append-mode failed CSV")

    ap.add_argument("--max-attempts", type=int, default=5, help="Retries per Subject/Activity/Trial")
    ap.add_argument("--backoff-base", type=float, default=5.0, help="Initial backoff seconds")
    ap.add_argument("--backoff-max", type=float, default=120.0, help="Max backoff seconds")
    ap.add_argument("--backoff-jitter", type=float, default=0.25, help="Backoff jitter fraction (0.25 = +-25%%)")

    ap.add_argument("--discovery-timeout", type=float, default=180.0, help="Seconds to wait for Subject widget")
    ap.add_argument("--navigation-timeout", type=int, default=120_000, help="Navigation timeout in ms")
    ap.add_argument("--action-timeout", type=int, default=20_000, help="Click action timeout in ms")
    ap.add_argument(
        "--immediate-download-timeout",
        type=int,
        default=20_000,
        help="Timeout in ms waiting for immediate download after Camera2 click",
    )
    ap.add_argument(
        "--followup-download-timeout",
        type=int,
        default=30_000,
        help="Timeout in ms for follow-up download clicks on intermediate pages",
    )

    ap.add_argument("--max-subjects", type=int, default=None, help="Optional limit for subject discovery (testing)")
    ap.add_argument("--max-combos", type=int, default=None, help="Optional limit for combos (testing)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = build_config_from_args(args)
    logger = DualLogger(cfg.log_path)
    stats = Stats()
    ensure_failed_csv_header(cfg.failed_csv)

    logger.log("Session started.")
    logger.log(f"Dataset root: {cfg.root}")
    logger.log(f"Staging dir: {cfg.staging_dir}")
    logger.log(f"Headless: {cfg.headless}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=cfg.headless)
        context = browser.new_context(accept_downloads=True)
        try:
            discovery_page = context.new_page()
            combos = discover_combinations(discovery_page, cfg=cfg, logger=logger)
            try:
                discovery_page.close()
            except Exception:
                pass
            close_all_context_pages(context)

            for idx, combo in enumerate(combos, start=1):
                logger.log(f"Processing item {idx}/{len(combos)}.", combo)
                process_combo(context=context, combo=combo, cfg=cfg, logger=logger, stats=stats)
        finally:
            close_all_context_pages(context)
            context.close()
            browser.close()

    logger.log(
        f"Finished. downloaded={stats.downloaded} skipped={stats.skipped} failed={stats.failed}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
