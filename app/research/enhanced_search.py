"""Enhanced web search tool with advanced content extraction and domain-specific strategies."""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

from ..exceptions import ResearchError
from ..tool.search.base import SearchItem, WebSearchEngine
from ..tool.web_search import WebSearch
from .models import ResearchFinding, ResearchSource, SourceType

logger = logging.getLogger(__name__)


class DomainStrategy:
    """Base class for domain-specific search strategies."""

    def __init__(self, name: str):
        self.name = name

    def expand_query(self, query: str) -> List[str]:
        """Expand query with domain-specific terms."""
        return [query]

    def filter_results(self, results: List[SearchItem]) -> List[SearchItem]:
        """Filter results based on domain relevance."""
        return results

    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract domain-specific content from HTML."""
        return {"content": self._extract_text(html)}

    def _extract_text(self, html: str) -> str:
        """Basic text extraction."""
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        return soup.get_text(separator="\n", strip=True)


class AcademicStrategy(DomainStrategy):
    """Search strategy for academic content."""

    def __init__(self):
        super().__init__("academic")
        self.academic_sites = [
            "scholar.google.com",
            "arxiv.org",
            "pubmed.ncbi.nlm.nih.gov",
            "researchgate.net",
            "jstor.org",
            "springer.com",
            "nature.com",
            "science.org",
            "cell.com",
            "plos.org",
            "acm.org",
            "ieee.org",
        ]

    def expand_query(self, query: str) -> List[str]:
        """Expand query with academic terms."""
        academic_terms = ["research", "study", "analysis", "paper", "journal"]
        expanded = [query]

        for term in academic_terms:
            if term not in query.lower():
                expanded.append(f"{query} {term}")

        # Add site-specific searches
        for site in self.academic_sites[:3]:  # Limit to top 3
            expanded.append(f"site:{site} {query}")

        return expanded

    def filter_results(self, results: List[SearchItem]) -> List[SearchItem]:
        """Filter for academic sources."""
        filtered = []
        for result in results:
            domain = urlparse(result.url).netloc.lower()
            if any(site in domain for site in self.academic_sites) or any(
                term in result.title.lower()
                for term in ["study", "research", "analysis", "paper"]
            ):
                filtered.append(result)
        return filtered

    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract academic content with metadata."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = ""
        title_tag = soup.find("title") or soup.find("h1")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Extract abstract
        abstract = ""
        abstract_selectors = [
            ".abstract",
            "#abstract",
            '[class*="abstract"]',
            ".summary",
            "#summary",
            '[class*="summary"]',
        ]
        for selector in abstract_selectors:
            abstract_elem = soup.select_one(selector)
            if abstract_elem:
                abstract = abstract_elem.get_text(strip=True)
                break

        # Extract authors
        authors = []
        author_selectors = [
            ".author",
            ".authors",
            '[class*="author"]',
            ".byline",
            '[class*="byline"]',
        ]
        for selector in author_selectors:
            author_elems = soup.select(selector)
            for elem in author_elems:
                authors.append(elem.get_text(strip=True))

        # Extract publication date
        pub_date = None
        date_selectors = [
            ".date",
            ".published",
            '[class*="date"]',
            ".publication-date",
            "[datetime]",
        ]
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                pub_date = date_elem.get_text(strip=True)
                break

        return {
            "content": self._extract_text(html),
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "publication_date": pub_date,
            "content_type": "academic",
        }


class NewsStrategy(DomainStrategy):
    """Search strategy for news content."""

    def __init__(self):
        super().__init__("news")
        self.news_sites = [
            "reuters.com",
            "ap.org",
            "bbc.com",
            "npr.org",
            "pbs.org",
            "wsj.com",
            "nytimes.com",
            "washingtonpost.com",
            "economist.com",
            "cnn.com",
            "bloomberg.com",
            "ft.com",
        ]

    def expand_query(self, query: str) -> List[str]:
        """Expand query with news-specific terms."""
        news_terms = ["news", "breaking", "latest", "report", "update"]
        expanded = [query]

        # Add temporal qualifiers
        expanded.extend(
            [
                f"{query} latest news",
                f"{query} recent developments",
                f"{query} breaking news",
            ]
        )

        # Add site-specific searches
        for site in self.news_sites[:3]:
            expanded.append(f"site:{site} {query}")

        return expanded

    def filter_results(self, results: List[SearchItem]) -> List[SearchItem]:
        """Filter for news sources."""
        filtered = []
        for result in results:
            domain = urlparse(result.url).netloc.lower()
            if any(site in domain for site in self.news_sites) or any(
                term in result.title.lower() for term in ["news", "breaking", "report"]
            ):
                filtered.append(result)
        return filtered

    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract news content with metadata."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract headline
        headline = ""
        headline_selectors = ["h1", ".headline", ".title", '[class*="headline"]']
        for selector in headline_selectors:
            headline_elem = soup.select_one(selector)
            if headline_elem:
                headline = headline_elem.get_text(strip=True)
                break

        # Extract byline/author
        byline = ""
        byline_selectors = [
            ".byline",
            ".author",
            '[class*="author"]',
            '[class*="byline"]',
        ]
        for selector in byline_selectors:
            byline_elem = soup.select_one(selector)
            if byline_elem:
                byline = byline_elem.get_text(strip=True)
                break

        # Extract publication date
        pub_date = ""
        date_selectors = [".date", ".published", "[datetime]", '[class*="date"]']
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                pub_date = date_elem.get_text(strip=True) or date_elem.get(
                    "datetime", ""
                )
                break

        # Extract article body
        article_content = ""
        article_selectors = [
            ".article-body",
            ".story-body",
            ".content",
            '[class*="article"]',
            ".post-content",
            ".entry-content",
        ]
        for selector in article_selectors:
            article_elem = soup.select_one(selector)
            if article_elem:
                article_content = article_elem.get_text(strip=True)
                break

        return {
            "content": article_content or self._extract_text(html),
            "headline": headline,
            "byline": byline,
            "publication_date": pub_date,
            "content_type": "news",
        }


class TechnicalStrategy(DomainStrategy):
    """Search strategy for technical documentation."""

    def __init__(self):
        super().__init__("technical")
        self.tech_sites = [
            "docs.python.org",
            "developer.mozilla.org",
            "stackoverflow.com",
            "github.com",
            "microsoft.com/docs",
            "aws.amazon.com/documentation",
            "kubernetes.io",
            "docker.com/docs",
            "tensorflow.org",
        ]

    def expand_query(self, query: str) -> List[str]:
        """Expand query with technical terms."""
        tech_terms = ["documentation", "tutorial", "guide", "API", "reference"]
        expanded = [query]

        for term in tech_terms:
            if term not in query.lower():
                expanded.append(f"{query} {term}")

        # Add site-specific searches
        for site in self.tech_sites[:3]:
            expanded.append(f"site:{site} {query}")

        return expanded

    def filter_results(self, results: List[SearchItem]) -> List[SearchItem]:
        """Filter for technical sources."""
        filtered = []
        for result in results:
            domain = urlparse(result.url).netloc.lower()
            if any(site in domain for site in self.tech_sites) or any(
                term in result.title.lower()
                for term in ["docs", "documentation", "api", "tutorial"]
            ):
                filtered.append(result)
        return filtered

    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract technical content with code examples."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract code blocks
        code_blocks = []
        for code_elem in soup.find_all(["code", "pre"]):
            code_text = code_elem.get_text(strip=True)
            if len(code_text) > 10:  # Filter out small inline code
                code_blocks.append(code_text)

        # Extract API endpoints
        api_patterns = [
            r"(GET|POST|PUT|DELETE|PATCH)\s+(/[^\s]*)",
            r"https?://[^\s]+/api/[^\s]*",
        ]
        api_endpoints = []
        text = soup.get_text()
        for pattern in api_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            api_endpoints.extend(matches)

        return {
            "content": self._extract_text(html),
            "code_blocks": code_blocks,
            "api_endpoints": api_endpoints,
            "content_type": "technical",
        }


class EnhancedWebSearchTool:
    """Enhanced web search tool with domain-specific strategies and intelligent query expansion."""

    def __init__(self):
        self.base_search = WebSearch()
        self.strategies = {
            "academic": AcademicStrategy(),
            "news": NewsStrategy(),
            "technical": TechnicalStrategy(),
            "general": DomainStrategy("general"),
        }
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def enhanced_search(
        self,
        query: str,
        domain: str = "general",
        num_results: int = 10,
        enable_query_expansion: bool = True,
        enable_content_extraction: bool = True,
        credibility_threshold: float = 0.3,
    ) -> List[ResearchFinding]:
        """
        Perform enhanced web search with domain-specific strategies.

        Args:
            query: The search query
            domain: Domain strategy to use (academic, news, technical, general)
            num_results: Number of results to return
            enable_query_expansion: Whether to expand the query
            enable_content_extraction: Whether to extract full content
            credibility_threshold: Minimum credibility threshold for results

        Returns:
            List of research findings with enhanced content
        """
        try:
            logger.info(f"Starting enhanced search for '{query}' in domain '{domain}'")

            strategy = self.strategies.get(domain, self.strategies["general"])

            # Expand query if enabled
            queries = [query]
            if enable_query_expansion:
                queries = await self._expand_query_intelligently(query, strategy)

            # Perform searches for all query variations
            all_results = []
            for search_query in queries[:3]:  # Limit to top 3 queries
                try:
                    search_response = await self.base_search.execute(
                        query=search_query,
                        num_results=num_results,
                        fetch_content=False,  # We'll do our own content extraction
                    )

                    if search_response.results:
                        # Convert to SearchItem format for strategy processing
                        search_items = [
                            SearchItem(
                                title=result.title,
                                url=result.url,
                                description=result.description,
                            )
                            for result in search_response.results
                        ]

                        # Apply domain-specific filtering
                        filtered_items = strategy.filter_results(search_items)
                        all_results.extend(filtered_items)

                except Exception as e:
                    logger.warning(
                        f"Search failed for query '{search_query}': {str(e)}"
                    )
                    continue

            # Remove duplicates
            unique_results = self._deduplicate_results(all_results)

            # Extract enhanced content if enabled
            findings = []
            if enable_content_extraction and self.session:
                findings = await self._extract_enhanced_content(
                    unique_results[:num_results], strategy, query
                )
            else:
                # Create basic findings without content extraction
                findings = await self._create_basic_findings(
                    unique_results[:num_results], query, domain
                )

            # Filter by credibility threshold
            filtered_findings = [
                f
                for f in findings
                if f.source.credibility_score >= credibility_threshold
            ]

            logger.info(f"Enhanced search completed: {len(filtered_findings)} findings")
            return filtered_findings

        except Exception as e:
            logger.error(f"Enhanced search failed: {str(e)}")
            raise ResearchError(f"Enhanced search failed: {str(e)}") from e

    async def _expand_query_intelligently(
        self, query: str, strategy: DomainStrategy
    ) -> List[str]:
        """Intelligently expand query using various techniques."""
        expanded_queries = set([query])

        # Domain-specific expansion
        domain_expanded = strategy.expand_query(query)
        expanded_queries.update(domain_expanded)

        # Synonym expansion
        synonyms = await self._get_query_synonyms(query)
        for synonym in synonyms:
            expanded_queries.add(f"{query} {synonym}")

        # Related terms expansion
        related_terms = await self._get_related_terms(query)
        for term in related_terms:
            expanded_queries.add(f"{query} {term}")

        # Question variations
        question_variations = self._generate_question_variations(query)
        expanded_queries.update(question_variations)

        return list(expanded_queries)

    async def _get_query_synonyms(self, query: str) -> List[str]:
        """Get synonyms for query terms (simplified implementation)."""
        # This is a simplified implementation
        # In production, you might use WordNet, word embeddings, or external APIs
        synonym_map = {
            "research": ["study", "investigation", "analysis"],
            "analysis": ["examination", "evaluation", "assessment"],
            "development": ["creation", "building", "construction"],
            "technology": ["tech", "innovation", "advancement"],
            "method": ["approach", "technique", "procedure"],
            "system": ["framework", "platform", "infrastructure"],
        }

        synonyms = []
        query_words = query.lower().split()

        for word in query_words:
            if word in synonym_map:
                synonyms.extend(synonym_map[word])

        return synonyms[:3]  # Limit to top 3 synonyms

    async def _get_related_terms(self, query: str) -> List[str]:
        """Get related terms for the query."""
        # Simplified related terms based on common patterns
        related_terms = []

        if "python" in query.lower():
            related_terms.extend(["programming", "coding", "development"])
        elif "machine learning" in query.lower():
            related_terms.extend(["AI", "artificial intelligence", "ML"])
        elif "web" in query.lower():
            related_terms.extend(["internet", "online", "website"])

        return related_terms[:2]  # Limit to top 2 related terms

    def _generate_question_variations(self, query: str) -> List[str]:
        """Generate question variations of the query."""
        variations = []

        # Add question words if not present
        question_starters = ["what is", "how to", "why", "when", "where"]

        for starter in question_starters:
            if starter not in query.lower():
                variations.append(f"{starter} {query}")

        return variations[:2]  # Limit to top 2 variations

    def _deduplicate_results(self, results: List[SearchItem]) -> List[SearchItem]:
        """Remove duplicate search results."""
        seen_urls = set()
        unique_results = []

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results

    async def _extract_enhanced_content(
        self, results: List[SearchItem], strategy: DomainStrategy, query: str
    ) -> List[ResearchFinding]:
        """Extract enhanced content from search results."""
        findings = []

        for i, result in enumerate(results):
            try:
                # Fetch content
                content_data = await self._fetch_and_extract_content(
                    result.url, strategy
                )

                if not content_data or not content_data.get("content"):
                    continue

                # Create research source
                source = await self._create_research_source(result, content_data)

                # Extract key points
                key_points = self._extract_key_points(content_data["content"])

                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(
                    content_data["content"], query
                )

                # Create research finding
                finding = ResearchFinding(
                    content=content_data["content"][:5000],  # Limit content size
                    source=source,
                    query=query,
                    relevance_score=relevance_score,
                    confidence_score=source.credibility_score,
                    key_points=key_points,
                    metadata={
                        **content_data,
                        "search_position": i + 1,
                        "extraction_timestamp": datetime.utcnow().isoformat(),
                    },
                )

                findings.append(finding)

            except Exception as e:
                logger.warning(f"Failed to extract content from {result.url}: {str(e)}")
                continue

        return findings

    async def _create_basic_findings(
        self, results: List[SearchItem], query: str, domain: str
    ) -> List[ResearchFinding]:
        """Create basic findings without content extraction."""
        findings = []

        for i, result in enumerate(results):
            # Create basic research source
            source = ResearchSource(
                name=urlparse(result.url).netloc,
                url=result.url,
                source_type=self._determine_source_type(result.url, domain),
                credibility_score=self._estimate_credibility(result.url),
                metadata={"title": result.title, "description": result.description},
            )

            # Use description as content if available
            content = result.description or result.title or "No content available"

            # Calculate basic relevance score
            relevance_score = self._calculate_relevance_score(content, query)

            finding = ResearchFinding(
                content=content,
                source=source,
                query=query,
                relevance_score=relevance_score,
                confidence_score=source.credibility_score,
                key_points=[result.title] if result.title else [],
                metadata={"search_position": i + 1},
            )

            findings.append(finding)

        return findings

    async def _fetch_and_extract_content(
        self, url: str, strategy: DomainStrategy
    ) -> Optional[Dict[str, Any]]:
        """Fetch and extract content from a URL using domain strategy."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None

                html = await response.text()
                return strategy.extract_content(html, url)

        except Exception as e:
            logger.warning(f"Failed to fetch content from {url}: {str(e)}")
            return None

    async def _create_research_source(
        self, result: SearchItem, content_data: Dict[str, Any]
    ) -> ResearchSource:
        """Create a research source from search result and content data."""
        domain = urlparse(result.url).netloc

        return ResearchSource(
            name=domain,
            url=result.url,
            source_type=self._determine_source_type(
                result.url, content_data.get("content_type", "general")
            ),
            credibility_score=self._estimate_credibility(result.url),
            freshness_score=self._calculate_freshness_score(content_data),
            metadata={
                "title": result.title,
                "description": result.description,
                **{k: v for k, v in content_data.items() if k != "content"},
            },
        )

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        if not content:
            return []

        # Split into sentences
        sentences = re.split(r"[.!?]+", content)

        # Filter sentences by length and importance indicators
        key_sentences = []
        importance_indicators = [
            "important",
            "key",
            "significant",
            "crucial",
            "essential",
            "main",
            "primary",
            "major",
            "critical",
            "fundamental",
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length
                # Check for importance indicators
                if any(
                    indicator in sentence.lower() for indicator in importance_indicators
                ):
                    key_sentences.append(sentence)
                # Or if it's at the beginning (likely important)
                elif sentences.index(sentence + ".") < 3:
                    key_sentences.append(sentence)

        return key_sentences[:5]  # Return top 5 key points

    def _calculate_relevance_score(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query."""
        if not content or not query:
            return 0.0

        content_lower = content.lower()
        query_words = query.lower().split()

        # Count exact matches
        exact_matches = sum(1 for word in query_words if word in content_lower)

        # Calculate basic relevance score
        relevance = exact_matches / len(query_words) if query_words else 0.0

        # Boost score if query appears as phrase
        if query.lower() in content_lower:
            relevance += 0.2

        # Boost score based on word frequency
        total_words = len(content.split())
        if total_words > 0:
            word_frequency_boost = min(0.3, exact_matches / total_words * 10)
            relevance += word_frequency_boost

        return min(1.0, relevance)

    def _determine_source_type(self, url: str, domain: str) -> SourceType:
        """Determine the source type based on URL and domain."""
        domain_lower = urlparse(url).netloc.lower()

        # Academic sources
        academic_indicators = [
            "edu",
            "scholar",
            "arxiv",
            "pubmed",
            "researchgate",
            "jstor",
            "springer",
            "nature",
            "science",
            "cell",
            "plos",
            "acm",
            "ieee",
        ]
        if any(indicator in domain_lower for indicator in academic_indicators):
            return SourceType.ACADEMIC

        # News sources
        news_indicators = [
            "news",
            "reuters",
            "ap.org",
            "bbc",
            "npr",
            "pbs",
            "wsj",
            "nytimes",
            "washingtonpost",
            "economist",
            "cnn",
            "bloomberg",
            "ft",
        ]
        if any(indicator in domain_lower for indicator in news_indicators):
            return SourceType.NEWS

        # Government sources
        gov_indicators = ["gov", "mil", "org"]
        if any(domain_lower.endswith(f".{indicator}") for indicator in gov_indicators):
            return SourceType.GOVERNMENT

        # Technical/Documentation sources
        tech_indicators = [
            "docs",
            "documentation",
            "stackoverflow",
            "github",
            "microsoft",
            "aws",
            "kubernetes",
            "docker",
            "tensorflow",
        ]
        if any(indicator in domain_lower for indicator in tech_indicators):
            return SourceType.TECHNICAL

        return SourceType.WEB

    def _estimate_credibility(self, url: str) -> float:
        """Estimate credibility score based on URL characteristics."""
        domain = urlparse(url).netloc.lower()

        # High credibility domains
        high_credibility = [
            "edu",
            "gov",
            "mil",
            "scholar.google",
            "arxiv.org",
            "pubmed",
            "nature.com",
            "science.org",
            "reuters.com",
            "ap.org",
            "bbc.com",
        ]

        # Medium credibility domains
        medium_credibility = [
            "org",
            "wikipedia",
            "stackoverflow",
            "github",
            "microsoft",
            "aws",
            "wsj",
            "nytimes",
            "washingtonpost",
            "economist",
        ]

        # Check for high credibility
        if any(indicator in domain for indicator in high_credibility):
            return 0.9

        # Check for medium credibility
        if any(indicator in domain for indicator in medium_credibility):
            return 0.7

        # Check for suspicious indicators
        suspicious_indicators = ["blogspot", "wordpress", "tumblr", "geocities"]
        if any(indicator in domain for indicator in suspicious_indicators):
            return 0.3

        # Default credibility for unknown domains
        return 0.5

    def _calculate_freshness_score(self, content_data: Dict[str, Any]) -> float:
        """Calculate freshness score based on publication date."""
        pub_date_str = content_data.get("publication_date")
        if not pub_date_str:
            return 0.5  # Default score if no date available

        try:
            # Try to parse various date formats
            date_formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%B %d, %Y",
                "%b %d, %Y",
                "%d %B %Y",
                "%d %b %Y",
            ]

            pub_date = None
            for fmt in date_formats:
                try:
                    pub_date = datetime.strptime(pub_date_str[: len(fmt)], fmt)
                    break
                except ValueError:
                    continue

            if not pub_date:
                return 0.5  # Default if parsing fails

            # Calculate days since publication
            days_old = (datetime.now() - pub_date).days

            # Freshness scoring: newer is better
            if days_old <= 7:
                return 1.0  # Very fresh
            elif days_old <= 30:
                return 0.8  # Fresh
            elif days_old <= 90:
                return 0.6  # Moderately fresh
            elif days_old <= 365:
                return 0.4  # Somewhat old
            else:
                return 0.2  # Old

        except Exception as e:
            logger.warning(f"Failed to parse date '{pub_date_str}': {str(e)}")
            return 0.5

    async def search_with_ranking(
        self,
        query: str,
        domains: List[str] = None,
        num_results: int = 10,
        ranking_criteria: Dict[str, float] = None,
    ) -> List[ResearchFinding]:
        """
        Search with custom ranking criteria.

        Args:
            query: Search query
            domains: List of domain strategies to use
            num_results: Number of results to return
            ranking_criteria: Weights for ranking (relevance, credibility, freshness)

        Returns:
            Ranked list of research findings
        """
        if domains is None:
            domains = ["general"]

        if ranking_criteria is None:
            ranking_criteria = {"relevance": 0.5, "credibility": 0.3, "freshness": 0.2}

        all_findings = []

        # Search across all specified domains
        for domain in domains:
            try:
                findings = await self.enhanced_search(
                    query=query,
                    domain=domain,
                    num_results=num_results,
                    enable_query_expansion=True,
                    enable_content_extraction=True,
                )
                all_findings.extend(findings)
            except Exception as e:
                logger.warning(f"Search failed for domain '{domain}': {str(e)}")
                continue

        # Remove duplicates based on URL
        unique_findings = {}
        for finding in all_findings:
            url = finding.source.url
            if (
                url not in unique_findings
                or finding.relevance_score > unique_findings[url].relevance_score
            ):
                unique_findings[url] = finding

        findings_list = list(unique_findings.values())

        # Calculate composite scores and rank
        for finding in findings_list:
            composite_score = (
                finding.relevance_score * ranking_criteria.get("relevance", 0.5)
                + finding.source.credibility_score
                * ranking_criteria.get("credibility", 0.3)
                + finding.source.freshness_score
                * ranking_criteria.get("freshness", 0.2)
            )
            finding.metadata["composite_score"] = composite_score

        # Sort by composite score
        findings_list.sort(
            key=lambda x: x.metadata.get("composite_score", 0), reverse=True
        )

        return findings_list[:num_results]
