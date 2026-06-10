import argparse
import datetime as dt
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import arxiv
import matplotlib.pyplot as plt
from arxiv import UnexpectedEmptyPageError


Paper = dict[str, Any]
PaperStore = dict[str, dict[str, Paper]]

RESOURCE_TERMS = [
    "low-resource",
    "low resource",
    "few-shot",
    "few shot",
    "data-efficient",
    "data efficient",
    "sample-efficient",
    "sample efficient",
    "data-scarce",
    "data scarce",
    "low-data",
    "low data",
    "limited data",
    "data scarcity",
]

LOW_RESOURCE_SIGNALS = tuple(RESOURCE_TERMS)
GENERATION_SIGNALS = (
    "generation",
    "generative",
    "natural language generation",
    "text generation",
    "question generation",
    "summarization",
    "translation",
    "data-to-text",
    "dialogue",
)

KEEP_CATEGORIES = {"cs.CL", "cs.AI", "cs.LG", "cs.IR"}
BLOCK_CATEGORIES = {"cs.CV", "eess.AS", "cs.SD", "eess.SP", "q-bio.BM"}

ARXIV_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Settings:
    repo_owner: str = "bansky-cl"
    repo_name: str = "low-resource-generation-arxiv-daily-paper"
    topic: str = "low-resource-generation"
    json_file: Path = Path("docs/arxiv-daily.json")
    trend_file: Path = Path("imgs/trend.png")
    readme_file: Path = Path("README.md")
    readme_max_papers: int = 200
    recent_days: int = 30
    bootstrap_start_date: str = "2020-01-01"
    update_lookback_days: int = 365
    use_date_filter: bool = False
    bootstrap_max_results_per_query: int = 300
    update_max_results_per_query: int = 120
    arxiv_page_size: int = 100
    arxiv_delay_seconds: int = 3
    arxiv_num_retries: int = 5
    arxiv_run_retries: int = 3
    arxiv_backoff_seconds: int = 45

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            repo_owner=os.getenv("REPO_OWNER", cls.repo_owner),
            repo_name=os.getenv("REPO_NAME", cls.repo_name),
            readme_max_papers=int(os.getenv("README_MAX_PAPERS", str(cls.readme_max_papers))),
            recent_days=int(os.getenv("RECENT_DAYS", str(cls.recent_days))),
            bootstrap_start_date=os.getenv("ARXIV_BOOTSTRAP_START_DATE", cls.bootstrap_start_date),
            update_lookback_days=int(os.getenv("ARXIV_UPDATE_LOOKBACK_DAYS", str(cls.update_lookback_days))),
            use_date_filter=env_flag("ARXIV_USE_DATE_FILTER", cls.use_date_filter),
            bootstrap_max_results_per_query=int(
                os.getenv("ARXIV_BOOTSTRAP_MAX_RESULTS_PER_QUERY", str(cls.bootstrap_max_results_per_query))
            ),
            update_max_results_per_query=int(
                os.getenv("ARXIV_UPDATE_MAX_RESULTS_PER_QUERY", str(cls.update_max_results_per_query))
            ),
        )


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def build_queries() -> list[str]:
    # arXiv can return HTTP 500 for long OR queries combined with date ranges.
    # Short phrase queries plus local filtering are slower but much more stable.
    return [f'all:"{term}"' for term in RESOURCE_TERMS]


def add_date_filter(query: str, start_date: str | None) -> str:
    if not start_date:
        return query
    ymd = start_date.replace("-", "")
    if len(ymd) != 8 or not ymd.isdigit():
        raise ValueError(f"start_date must be YYYY-MM-DD, got {start_date!r}")
    return f"({query}) AND submittedDate:[{ymd}0000 TO *]"


def make_client(settings: Settings) -> arxiv.Client:
    return arxiv.Client(
        page_size=settings.arxiv_page_size,
        delay_seconds=settings.arxiv_delay_seconds,
        num_retries=settings.arxiv_num_retries,
    )


def iter_results_safe(client: arxiv.Client, search: arxiv.Search, settings: Settings):
    seen_ids: set[str] = set()
    for attempt in range(1, settings.arxiv_run_retries + 1):
        results = client.results(search)
        while True:
            try:
                result = next(results)
            except StopIteration:
                return
            except UnexpectedEmptyPageError as exc:
                print(f"[arXiv] empty page, stop paging: {exc}", flush=True)
                return
            except arxiv.HTTPError as exc:
                retryable = getattr(exc, "status", None) in ARXIV_RETRYABLE_STATUS
                if retryable and attempt < settings.arxiv_run_retries:
                    sleep_seconds = settings.arxiv_backoff_seconds * attempt
                    print(f"[arXiv] transient HTTP error: {exc}; retry after {sleep_seconds}s", flush=True)
                    time.sleep(sleep_seconds)
                    break
                print(f"[arXiv] HTTP error, stop this query: {exc}", flush=True)
                return

            result_key = getattr(result, "entry_id", None) or result.get_short_id()
            if result_key in seen_ids:
                continue
            seen_ids.add(result_key)
            yield result


def load_json(path: Path) -> PaperStore:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    data = json.loads(content)
    return data if isinstance(data, dict) else {}


def save_json(path: Path, data: PaperStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def is_bootstrap(data: PaperStore) -> bool:
    return not any(isinstance(papers, dict) and papers for papers in data.values())


def normalize_arxiv_id(short_id: str) -> str:
    return short_id.split("v")[0]


def extract_project_url(text: str) -> str | None:
    urls = re.findall(r"https?://[^\s{}<>\"']+", text)
    if not urls:
        return None
    preferred = [
        url for url in urls
        if any(host in url.lower() for host in ("github.com", "gitlab.com", "huggingface.co"))
    ]
    url = (preferred or urls)[-1]
    return url.rstrip(".,;:!?)\\]")


def should_keep(categories: list[str], title: str, abstract: str) -> bool:
    has_keep_category = any(category in KEEP_CATEGORIES for category in categories)
    has_block_category = any(category in BLOCK_CATEGORIES for category in categories)
    text = f"{title} {abstract}".lower()
    has_low_resource_signal = any(signal in text for signal in LOW_RESOURCE_SIGNALS)
    has_generation_signal = any(signal in text for signal in GENERATION_SIGNALS)
    if not (has_keep_category and has_low_resource_signal and has_generation_signal):
        return False
    return not (has_block_category and "cs.CL" not in categories)


def paper_from_result(result: arxiv.Result, settings: Settings) -> Paper | None:
    short_id = result.get_short_id()
    paper_id = normalize_arxiv_id(short_id)
    title = " ".join(result.title.split())
    abstract = " ".join(result.summary.split())
    categories = list(result.categories)

    if not should_keep(categories, title, abstract):
        return None

    code_url = extract_project_url(abstract)
    return {
        "id": paper_id,
        "version": short_id,
        "title": title,
        "authors": [str(author) for author in result.authors],
        "published": result.published.date().isoformat(),
        "updated": result.updated.date().isoformat(),
        "categories": categories,
        "abstract": abstract,
        "abs_url": result.entry_id,
        "pdf_url": result.pdf_url,
        "code_url": code_url,
    }


def fetch_papers(settings: Settings, start_date: str | None, max_results_per_query: int) -> dict[str, Paper]:
    client = make_client(settings)
    papers: dict[str, Paper] = {}
    for query in build_queries():
        search = arxiv.Search(
            query=add_date_filter(query, start_date),
            max_results=max_results_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        fetched = 0
        kept = 0
        for result in iter_results_safe(client, search, settings):
            fetched += 1
            paper = paper_from_result(result, settings)
            if not paper:
                continue
            kept += 1
            papers[paper["id"]] = paper
        print(f"[{settings.topic}] query={query!r}, fetched={fetched}, kept={kept}, since={start_date}", flush=True)
    return papers


def merge_papers(data: PaperStore, new_papers: dict[str, Paper], topic: str) -> PaperStore:
    topic_papers = data.setdefault(topic, {})
    for paper_id, paper in new_papers.items():
        existing = topic_papers.get(paper_id, {})
        topic_papers[paper_id] = {**existing, **paper}
    return data


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def all_unique_papers(data: PaperStore) -> list[Paper]:
    unique: dict[str, Paper] = {}
    for topic_papers in data.values():
        for paper_id, paper in topic_papers.items():
            if isinstance(paper, dict):
                unique[paper_id] = paper
    return sorted(unique.values(), key=lambda item: (item.get("published", ""), item.get("id", "")), reverse=True)


def json_to_trend(data: PaperStore, img_file: Path) -> None:
    counts = Counter()
    for paper in all_unique_papers(data):
        published = paper.get("published")
        if not published:
            continue
        try:
            published_date = parse_date(published)
        except ValueError:
            continue
        counts[f"{published_date.year:04d}-{published_date.month:02d}"] += 1

    img_file.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    if counts:
        months = sorted(counts)
        values = [counts[month] for month in months]
        plt.plot(months, values, marker="o", linewidth=1.8, label="Monthly count")
        plt.fill_between(months, values, alpha=0.12)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Papers")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No papers yet", ha="center", va="center", fontsize=16)
        plt.xticks([])
        plt.yticks([])
    plt.title("Low-Resource Generation Papers per Month")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(img_file, dpi=300)
    plt.close()
    print(f"trend saved to {img_file}", flush=True)


def md_escape(value: Any) -> str:
    text = str(value or "")
    return text.replace("\n", " ").replace("|", "\\|")


def paper_row(paper: Paper) -> str:
    date = md_escape(paper.get("published"))
    title = md_escape(paper.get("title"))
    categories = md_escape(", ".join(paper.get("categories", [])))
    version = md_escape(paper.get("version") or paper.get("id"))
    abs_url = paper.get("abs_url") or f"https://arxiv.org/abs/{version}"
    code_url = paper.get("code_url")
    code = f"**[code]({code_url})**" if code_url else "null"
    return f"|**{date}**|**{title}**|{categories}|[{version}]({abs_url})|{code}|"


def write_table(handle, papers: list[Paper]) -> None:
    handle.write("|Date|Title|Categories|PDF|Code|\n")
    handle.write("|---|---|---|---|---|\n")
    for paper in papers:
        handle.write(paper_row(paper) + "\n")
    handle.write("\n")


def json_to_md(data: PaperStore, settings: Settings) -> None:
    today = dt.date.today()
    cutoff = today - dt.timedelta(days=settings.recent_days)
    papers = all_unique_papers(data)
    recent = [paper for paper in papers if paper.get("published") and parse_date(paper["published"]) >= cutoff]
    older = [paper for paper in papers if paper.get("published") and parse_date(paper["published"]) < cutoff]

    recent_to_show = recent[: settings.readme_max_papers]
    remaining_slots = max(settings.readme_max_papers - len(recent_to_show), 0)
    older_to_show = older[:remaining_slots]
    hidden_count = max(len(papers) - len(recent_to_show) - len(older_to_show), 0)

    with settings.readme_file.open("w", encoding="utf-8") as handle:
        handle.write("[![Contributors][contributors-shield]][contributors-url]\n")
        handle.write("[![Forks][forks-shield]][forks-url]\n")
        handle.write("[![Stargazers][stars-shield]][stars-url]\n")
        handle.write("[![Issues][issues-shield]][issues-url]\n\n")

        handle.write("# Low-Resource Generation Arxiv Daily Paper\n\n")
        handle.write("This repository tracks low-resource generation related papers from arXiv.\n\n")
        handle.write(f"## Updated on {today.isoformat().replace('-', '.')}\n\n")
        handle.write("![Monthly Trend](imgs/trend.png)\n\n")
        handle.write("## Summary\n\n")
        handle.write(f"- Total papers in JSON: **{len(papers)}**\n")
        handle.write(f"- Recent {settings.recent_days} days: **{len(recent)}**\n")
        handle.write(f"- Older than {settings.recent_days} days: **{len(older)}**\n")
        handle.write(
            f"- README display limit: **{settings.readme_max_papers}** papers; "
            "extra papers stay in `docs/arxiv-daily.json`.\n\n"
        )

        handle.write(f"## Recent {settings.recent_days} Days\n\n")
        if recent_to_show:
            write_table(handle, recent_to_show)
        else:
            handle.write("No papers found in this window.\n\n")

        handle.write(f"## Older Than {settings.recent_days} Days\n\n")
        if older_to_show:
            write_table(handle, older_to_show)
        else:
            handle.write("No older papers to show within the README limit.\n\n")

        if hidden_count:
            handle.write(
                f"README omitted **{hidden_count}** older paper(s). "
                "See `docs/arxiv-daily.json` for the full archive.\n\n"
            )

        repo_url = f"https://github.com/{settings.repo_owner}/{settings.repo_name}"
        handle.write(
            f"[contributors-shield]: https://img.shields.io/github/contributors/"
            f"{settings.repo_owner}/{settings.repo_name}.svg?style=for-the-badge\n"
        )
        handle.write(f"[contributors-url]: {repo_url}/graphs/contributors\n")
        handle.write(
            f"[forks-shield]: https://img.shields.io/github/forks/"
            f"{settings.repo_owner}/{settings.repo_name}.svg?style=for-the-badge\n"
        )
        handle.write(f"[forks-url]: {repo_url}/network/members\n")
        handle.write(
            f"[stars-shield]: https://img.shields.io/github/stars/"
            f"{settings.repo_owner}/{settings.repo_name}.svg?style=for-the-badge\n"
        )
        handle.write(f"[stars-url]: {repo_url}/stargazers\n")
        handle.write(
            f"[issues-shield]: https://img.shields.io/github/issues/"
            f"{settings.repo_owner}/{settings.repo_name}.svg?style=for-the-badge\n"
        )
        handle.write(f"[issues-url]: {repo_url}/issues\n")
    print(f"README saved to {settings.readme_file}", flush=True)


def get_run_window(settings: Settings, data: PaperStore, override_max_results: int | None) -> tuple[str | None, int]:
    bootstrap = is_bootstrap(data)
    if bootstrap:
        start_date = settings.bootstrap_start_date if settings.use_date_filter else None
        max_results = settings.bootstrap_max_results_per_query
    else:
        start_date = None
        if settings.use_date_filter:
            start_date = (dt.date.today() - dt.timedelta(days=settings.update_lookback_days)).isoformat()
        max_results = settings.update_max_results_per_query
    if override_max_results is not None:
        max_results = override_max_results
    print(f"bootstrap={bootstrap}, start_date={start_date}, max_results_per_query={max_results}", flush=True)
    return start_date, max_results


def run_update(settings: Settings, max_results: int | None = None) -> None:
    data = load_json(settings.json_file)
    start_date, max_results_per_query = get_run_window(settings, data, max_results)
    new_papers = fetch_papers(settings, start_date=start_date, max_results_per_query=max_results_per_query)
    data = merge_papers(data, new_papers, settings.topic)
    save_json(settings.json_file, data)
    json_to_trend(data, settings.trend_file)
    json_to_md(data, settings)


def run_check(settings: Settings, max_results: int) -> None:
    papers = fetch_papers(settings, start_date=None, max_results_per_query=max_results)
    print(f"check kept={len(papers)}", flush=True)
    for paper in list(papers.values())[:10]:
        print(f"- {paper['id']} {paper['title']} [{', '.join(paper['categories'])}]", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch low-resource generation papers from arXiv.")
    parser.add_argument("--check-only", action="store_true", help="Fetch a small sample without writing files.")
    parser.add_argument("--max-results-per-query", type=int, help="Override arXiv max results for each query.")
    parser.add_argument("--use-date-filter", action="store_true", help="Enable submittedDate filters in arXiv queries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    if args.use_date_filter:
        settings = replace(settings, use_date_filter=True)

    if args.check_only:
        run_check(settings, max_results=args.max_results_per_query or 1)
    else:
        run_update(settings, max_results=args.max_results_per_query)


if __name__ == "__main__":
    main()
