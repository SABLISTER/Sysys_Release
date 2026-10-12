from __future__ import annotations

import argparse
import gzip
import html.parser
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree

DEFAULT_BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
DEFAULT_UPDATE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
DEFAULT_OUTPUT = Path("data/pubmed/pubmed_abstracts.jsonl")
DEFAULT_RAW_DIR = Path("data/pubmed/raw")


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def list_xml_gz_urls(source_url: str, pattern: str) -> list[str]:
    source_url = source_url.strip()
    if source_url.endswith(".xml.gz"):
        return [source_url]

    if not source_url.endswith("/"):
        source_url += "/"

    with urllib.request.urlopen(source_url, timeout=60) as response:
        content = response.read().decode("utf-8", errors="replace")

    parser = LinkParser()
    parser.feed(content)
    regex = re.compile(pattern)
    urls = []
    for link in parser.links:
        name = Path(urllib.parse.urlparse(link).path).name
        if name.endswith(".xml.gz") and regex.search(name):
            urls.append(urllib.parse.urljoin(source_url, link))
    return sorted(set(urls))


def download_file(url: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.urlparse(url).path).name
    target = raw_dir / filename
    partial = raw_dir / f"{filename}.part"

    if target.exists() and target.stat().st_size > 0:
        print(f"skip existing: {target}")
        return target

    print(f"download: {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(partial, target)
    return target


def text_content(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def first_text(element: ElementTree.Element | None, path: str) -> str:
    if element is None:
        return ""
    return text_content(element.find(path))


def parse_year(article: ElementTree.Element) -> int | None:
    candidates = [
        ".//Article/Journal/JournalIssue/PubDate/Year",
        ".//Article/ArticleDate/Year",
        ".//DateCompleted/Year",
        ".//DateRevised/Year",
    ]
    for path in candidates:
        raw = first_text(article, path)
        if raw.isdigit():
            return int(raw)
    medline_date = first_text(article, ".//Article/Journal/JournalIssue/PubDate/MedlineDate")
    match = re.search(r"\b(18|19|20)\d{2}\b", medline_date)
    return int(match.group()) if match else None


def parse_abstract(article: ElementTree.Element) -> str:
    abstract = article.find(".//Article/Abstract")
    if abstract is None:
        return ""
    parts = []
    for item in abstract.findall("AbstractText"):
        label = item.get("Label") or item.get("NlmCategory") or ""
        text = text_content(item)
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts).strip()


def parse_authors(article: ElementTree.Element) -> list[str]:
    authors = []
    for author in article.findall(".//Article/AuthorList/Author"):
        collective = first_text(author, "CollectiveName")
        if collective:
            authors.append(collective)
            continue
        last = first_text(author, "LastName")
        fore = first_text(author, "ForeName")
        initials = first_text(author, "Initials")
        name = ", ".join(part for part in (last, fore or initials) if part)
        if name:
            authors.append(name)
    return authors


def parse_article_ids(article: ElementTree.Element) -> dict[str, str]:
    ids = {}
    pmid = first_text(article, ".//MedlineCitation/PMID")
    if pmid:
        ids["pmid"] = pmid
    for item in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = (item.get("IdType") or "").strip().lower()
        value = text_content(item)
        if id_type and value:
            ids[id_type] = value
    return ids


def parse_publication_types(article: ElementTree.Element) -> list[str]:
    return [text_content(item) for item in article.findall(".//PublicationTypeList/PublicationType") if text_content(item)]


def parse_mesh_terms(article: ElementTree.Element) -> list[str]:
    terms = []
    for heading in article.findall(".//MeshHeadingList/MeshHeading"):
        descriptor = text_content(heading.find("DescriptorName"))
        if descriptor:
            terms.append(descriptor)
    return terms


def parse_pubmed_article(article: ElementTree.Element, source_file: str) -> dict:
    ids = parse_article_ids(article)
    journal = article.find(".//Article/Journal")
    return {
        "title": text_content(article.find(".//Article/ArticleTitle")),
        "abstract": parse_abstract(article),
        "authors": parse_authors(article),
        "year": parse_year(article),
        "doi": ids.get("doi", ""),
        "pmid": ids.get("pmid", ""),
        "pmcid": ids.get("pmc", ""),
        "journal": first_text(journal, "Title"),
        "journal_iso": first_text(journal, "ISOAbbreviation"),
        "publication_types": parse_publication_types(article),
        "mesh_terms": parse_mesh_terms(article),
        "source": "PubMed",
        "source_file": source_file,
    }


def iter_pubmed_records(path: Path, include_no_abstract: bool) -> Iterable[dict]:
    with gzip.open(path, "rb") as handle:
        context = ElementTree.iterparse(handle, events=("end",))
        for _, element in context:
            if element.tag != "PubmedArticle":
                continue
            record = parse_pubmed_article(element, path.name)
            if include_no_abstract or record.get("abstract"):
                yield record
            element.clear()


def write_jsonl(records: Iterable[dict], output_path: Path, progress_every: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
            if progress_every > 0 and count % progress_every == 0:
                print(f"parsed records: {count:,}")
    os.replace(tmp_path, output_path)
    return count


def collect_records(paths: list[Path], include_no_abstract: bool) -> Iterable[dict]:
    for path in paths:
        print(f"parse: {path}")
        yield from iter_pubmed_records(path, include_no_abstract=include_no_abstract)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default=DEFAULT_BASELINE_URL)
    parser.add_argument("--include-updates", action="store_true")
    parser.add_argument("--update-url", default=DEFAULT_UPDATE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--pattern", default=r"^pubmed\d+n\d{4}\.xml\.gz$")
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--include-no-abstract", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser


def resolve_downloads(args: argparse.Namespace) -> list[Path]:
    if args.skip_download:
        paths = sorted(args.raw_dir.glob("*.xml.gz"))
        if args.limit_files > 0:
            paths = paths[: args.limit_files]
        return paths

    urls = list_xml_gz_urls(args.source_url, args.pattern)
    if args.include_updates:
        urls.extend(list_xml_gz_urls(args.update_url, args.pattern))
    urls = sorted(set(urls))
    if args.limit_files > 0:
        urls = urls[: args.limit_files]

    if not urls:
        raise SystemExit("No PubMed .xml.gz files found. Check --source-url or --pattern.")

    return [download_file(url, args.raw_dir) for url in urls]


def main() -> int:
    args = build_arg_parser().parse_args()
    paths = resolve_downloads(args)
    if not paths:
        print(f"No raw .xml.gz files found in {args.raw_dir}", file=sys.stderr)
        return 1
    count = write_jsonl(
        collect_records(paths, include_no_abstract=args.include_no_abstract),
        args.output,
        progress_every=args.progress_every,
    )
    print(f"saved {count:,} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
