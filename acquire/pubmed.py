"""Async PubMed E-Utilities client."""
import asyncio
import logging
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed efetch XML into paper dicts."""
    papers = []
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        logger.error("PubMed XML parse error: %s", exc)
        return []

    for article in root.iter("PubmedArticle"):
        try:
            medline = article.find("MedlineCitation")
            if medline is None:
                continue

            pmid_el = medline.find("PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            art = medline.find("Article")
            if art is None:
                continue

            title_el = art.find("ArticleTitle")
            title = title_el.text if title_el is not None else ""

            abstract_el = art.find("Abstract")
            abstract = ""
            if abstract_el is not None:
                parts = []
                for at in abstract_el.findall("AbstractText"):
                    label = at.get("Label", "")
                    text = "".join(at.itertext())
                    if label:
                        parts.append(f"{label}: {text}")
                    else:
                        parts.append(text)
                abstract = " ".join(parts)

            authors = []
            author_list = art.find("AuthorList")
            if author_list is not None:
                for au in author_list.findall("Author"):
                    last = au.findtext("LastName", "")
                    fore = au.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last}, {fore}".strip(", "))

            year = 0
            journal = art.find("Journal")
            if journal is not None:
                ji = journal.find("JournalIssue")
                if ji is not None:
                    pd = ji.find("PubDate")
                    if pd is not None:
                        y_el = pd.find("Year")
                        if y_el is not None and y_el.text:
                            try:
                                year = int(y_el.text)
                            except ValueError:
                                pass

            doi = ""
            article_ids = article.find("PubmedData")
            if article_ids is not None:
                for aid in article_ids.iter("ArticleId"):
                    if aid.get("IdType") == "doi" and aid.text:
                        doi = aid.text
                        break

            papers.append({
                "title": title or "",
                "abstract": abstract or "",
                "authors": authors,
                "year": year,
                "doi": doi,
                "pmid": pmid,
                "source": "PubMed",
            })
        except Exception as exc:
            logger.warning("PubMed: error parsing article: %s", exc)
            continue

    return papers


class PubMedClient:
    """Async client for PubMed E-Utilities (esearch + efetch)."""

    def __init__(self, api_key: str = "", email: str = "", tool: str = "sysys"):
        self._params: dict[str, str] = {}
        if api_key:
            self._params["api_key"] = api_key
        if email:
            self._params["email"] = email
            self._params["tool"] = tool
        self._client = httpx.AsyncClient(timeout=30.0)
        self._delay = 0.1 if api_key else 0.34

    async def _request_with_retry(self, url: str, params: dict, retries: int = 3) -> httpx.Response:
        """GET with exponential-backoff retry for transient HTTP errors."""
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                await asyncio.sleep(self._delay * attempt)
                resp = await self._client.get(url, params=params)
                resp.raise_for_status()
                return resp
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < retries - 1:
                    logger.warning("PubMed request failed (%s), retrying...", exc)
                    continue
                raise last_exc
        raise last_exc or httpx.HTTPError("PubMed retries exhausted")

    async def _esearch(self, query: str, retmax: int = 10000) -> dict:
        """Run esearch and return the esearchresult dict."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(retmax),
            "usehistory": "y",
            "retmode": "json",
            **self._params,
        }
        logger.info("PubMed esearch: query='%s'", query[:120])
        resp = await self._request_with_retry(ESEARCH_URL, params)
        return resp.json().get("esearchresult", {})

    async def _efetch_batch(
        self, webenv: str, query_key: str, retstart: int, retmax: int = 200
    ) -> list[dict]:
        """Fetch one batch of PubMed articles via efetch XML."""
        params = {
            "db": "pubmed",
            "retmode": "xml",
            "rettype": "abstract",
            "WebEnv": webenv,
            "query_key": query_key,
            "retstart": str(retstart),
            "retmax": str(retmax),
            **self._params,
        }
        await asyncio.sleep(self._delay)
        resp = await self._request_with_retry(EFETCH_URL, params)
        return parse_pubmed_xml(resp.text)

    async def search(self, query: str, max_results: int = 5000) -> list[dict]:
        """Search PubMed and return parsed paper dicts."""
        if not query:
            return []

        esearch = await self._esearch(query, retmax=max_results)
        count = int(esearch.get("count", 0))
        webenv = esearch.get("webenv", "")
        query_key = esearch.get("querykey", "1")
        id_list = esearch.get("idlist", [])

        logger.info("PubMed esearch: %d hits, %d IDs, WebEnv=%s",
                     count, len(id_list), (webenv or "N/A")[:20])

        if not id_list and not webenv:
            return []

        papers: list[dict] = []
        batch_size = 200
        fetched = 0
        total = min(max_results, count)

        while fetched < total:
            batch = await self._efetch_batch(webenv, query_key, fetched, batch_size)
            if not batch:
                break
            papers.extend(batch)
            fetched += len(batch)
            logger.info("PubMed: fetched %d/%d", len(papers), total)

        logger.info("PubMed search complete: %d papers", len(papers))
        return papers

    async def close(self):
        await self._client.aclose()
