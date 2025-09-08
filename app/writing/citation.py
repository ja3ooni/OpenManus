"""
Citation management system with support for multiple citation styles.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

from ..logger import get_logger
from .models import Citation, CitationStyle

logger = get_logger(__name__)


class CitationManager:
    """
    Manages citations and references with support for multiple citation styles.

    Provides automatic citation generation from research sources, citation validation,
    formatting consistency checks, and bibliography generation.
    """

    def __init__(self):
        """Initialize the citation manager."""
        self.citations: Dict[str, Citation] = {}
        self.citation_counter = 0
        self.style_formatters = {
            CitationStyle.APA: self._format_apa,
            CitationStyle.MLA: self._format_mla,
            CitationStyle.CHICAGO: self._format_chicago,
            CitationStyle.IEEE: self._format_ieee,
            CitationStyle.HARVARD: self._format_harvard,
        }

    async def generate_citations(
        self, sources: List[Dict], style: CitationStyle = CitationStyle.APA
    ) -> List[Citation]:
        """
        Generate citations from research sources.

        Args:
            sources: List of source dictionaries with metadata
            style: Citation style to use

        Returns:
            List of formatted citations
        """
        logger.info(
            f"Generating citations in {style.value} style for {len(sources)} sources"
        )

        citations = []
        for source in sources:
            try:
                citation = await self._create_citation_from_source(source)
                if citation:
                    self.citations[citation.id] = citation
                    citations.append(citation)
            except Exception as e:
                logger.error(f"Failed to create citation from source: {str(e)}")
                continue

        return citations

    async def validate_citations(
        self, citations: List[Citation]
    ) -> Dict[str, List[str]]:
        """
        Validate citations for completeness and formatting consistency.

        Args:
            citations: List of citations to validate

        Returns:
            Dictionary with validation results and issues
        """
        logger.info(f"Validating {len(citations)} citations")

        validation_results = {"valid": [], "warnings": [], "errors": []}

        for citation in citations:
            issues = await self._validate_single_citation(citation)

            if not issues:
                validation_results["valid"].append(citation.id)
            else:
                for issue in issues:
                    if issue["severity"] == "error":
                        validation_results["errors"].append(
                            f"{citation.id}: {issue['message']}"
                        )
                    else:
                        validation_results["warnings"].append(
                            f"{citation.id}: {issue['message']}"
                        )

        return validation_results

    async def format_citation(self, citation: Citation, style: CitationStyle) -> str:
        """
        Format a single citation in the specified style.

        Args:
            citation: Citation to format
            style: Citation style to use

        Returns:
            Formatted citation string
        """
        formatter = self.style_formatters.get(style)
        if not formatter:
            raise ValueError(f"Unsupported citation style: {style}")

        return formatter(citation)

    async def generate_bibliography(
        self,
        citations: List[Citation],
        style: CitationStyle = CitationStyle.APA,
        sort_alphabetically: bool = True,
    ) -> str:
        """
        Generate a complete bibliography from citations.

        Args:
            citations: List of citations to include
            style: Citation style to use
            sort_alphabetically: Whether to sort citations alphabetically

        Returns:
            Formatted bibliography string
        """
        logger.info(
            f"Generating bibliography with {len(citations)} citations in {style.value} style"
        )

        if not citations:
            return ""

        # Sort citations if requested
        if sort_alphabetically:
            citations = sorted(
                citations, key=lambda c: c.authors[0] if c.authors else c.title
            )

        # Format each citation
        formatted_citations = []
        for citation in citations:
            try:
                formatted = await self.format_citation(citation, style)
                formatted_citations.append(formatted)
            except Exception as e:
                logger.error(f"Failed to format citation {citation.id}: {str(e)}")
                continue

        # Create bibliography header based on style
        if style == CitationStyle.APA:
            header = "References\n\n"
        elif style == CitationStyle.MLA:
            header = "Works Cited\n\n"
        else:
            header = "Bibliography\n\n"

        return header + "\n\n".join(formatted_citations)

    async def add_citation(self, citation: Citation) -> str:
        """
        Add a citation to the manager.

        Args:
            citation: Citation to add

        Returns:
            Citation ID
        """
        self.citations[citation.id] = citation
        return citation.id

    async def get_citation(self, citation_id: str) -> Optional[Citation]:
        """
        Retrieve a citation by ID.

        Args:
            citation_id: ID of the citation to retrieve

        Returns:
            Citation if found, None otherwise
        """
        return self.citations.get(citation_id)

    async def update_citation(
        self, citation_id: str, updated_citation: Citation
    ) -> bool:
        """
        Update an existing citation.

        Args:
            citation_id: ID of the citation to update
            updated_citation: Updated citation data

        Returns:
            True if updated successfully, False otherwise
        """
        if citation_id in self.citations:
            self.citations[citation_id] = updated_citation
            return True
        return False

    async def remove_citation(self, citation_id: str) -> bool:
        """
        Remove a citation from the manager.

        Args:
            citation_id: ID of the citation to remove

        Returns:
            True if removed successfully, False otherwise
        """
        if citation_id in self.citations:
            del self.citations[citation_id]
            return True
        return False

    async def _create_citation_from_source(self, source: Dict) -> Optional[Citation]:
        """Create a citation from a source dictionary."""
        try:
            self.citation_counter += 1
            citation_id = f"cite_{self.citation_counter}"

            # Extract common fields
            title = source.get("title", "")
            authors = source.get("authors", [])
            if isinstance(authors, str):
                authors = [authors]

            url = source.get("url", "")
            publication_date = source.get("publication_date")
            if isinstance(publication_date, str):
                try:
                    publication_date = datetime.fromisoformat(
                        publication_date.replace("Z", "+00:00")
                    )
                except:
                    publication_date = None

            # Determine source type
            source_type = source.get("type", self._infer_source_type(source))

            citation = Citation(
                id=citation_id,
                source_type=source_type,
                title=title,
                authors=authors,
                publication_date=publication_date,
                url=url,
                doi=source.get("doi"),
                isbn=source.get("isbn"),
                publisher=source.get("publisher"),
                journal=source.get("journal"),
                volume=source.get("volume"),
                issue=source.get("issue"),
                pages=source.get("pages"),
                access_date=datetime.now() if url else None,
            )

            return citation

        except Exception as e:
            logger.error(f"Failed to create citation from source: {str(e)}")
            return None

    def _infer_source_type(self, source: Dict) -> str:
        """Infer source type from source metadata."""
        if source.get("url"):
            domain = urlparse(source["url"]).netloc.lower()
            if any(
                academic in domain
                for academic in ["scholar.google", "jstor", "pubmed", "arxiv"]
            ):
                return "journal"
            elif any(
                news in domain
                for news in ["news", "times", "post", "guardian", "reuters"]
            ):
                return "news"
            else:
                return "website"
        elif source.get("journal"):
            return "journal"
        elif source.get("isbn"):
            return "book"
        else:
            return "misc"

    async def _validate_single_citation(self, citation: Citation) -> List[Dict]:
        """Validate a single citation and return issues."""
        issues = []

        # Check required fields
        if not citation.title:
            issues.append({"severity": "error", "message": "Missing title"})

        if not citation.authors:
            issues.append({"severity": "warning", "message": "Missing authors"})

        # Validate URLs
        if citation.url:
            if not self._is_valid_url(citation.url):
                issues.append({"severity": "error", "message": "Invalid URL format"})

        # Validate DOI
        if citation.doi:
            if not self._is_valid_doi(citation.doi):
                issues.append({"severity": "warning", "message": "Invalid DOI format"})

        # Check for missing publication date
        if not citation.publication_date and citation.source_type in [
            "journal",
            "book",
        ]:
            issues.append(
                {"severity": "warning", "message": "Missing publication date"}
            )

        return issues

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _is_valid_doi(self, doi: str) -> bool:
        """Check if DOI is valid."""
        doi_pattern = r"^10\.\d{4,}\/[-._;()\/:a-zA-Z0-9]+$"
        return bool(re.match(doi_pattern, doi))

    def _format_apa(self, citation: Citation) -> str:
        """Format citation in APA style."""
        parts = []

        # Authors
        if citation.authors:
            if len(citation.authors) == 1:
                author_str = citation.authors[0]
            elif len(citation.authors) <= 7:
                author_str = (
                    ", ".join(citation.authors[:-1]) + f", & {citation.authors[-1]}"
                )
            else:
                author_str = (
                    ", ".join(citation.authors[:6]) + ", ... " + citation.authors[-1]
                )
            parts.append(author_str)
        else:
            parts.append("Unknown Author")

        # Year
        if citation.publication_date:
            parts.append(f"({citation.publication_date.year})")
        else:
            parts.append("(n.d.)")

        # Title
        if citation.source_type in ["journal", "book"]:
            parts.append(f"{citation.title}.")
        else:
            parts.append(citation.title)

        # Source-specific formatting
        if citation.source_type == "journal" and citation.journal:
            journal_part = f"*{citation.journal}*"
            if citation.volume:
                journal_part += f", {citation.volume}"
                if citation.issue:
                    journal_part += f"({citation.issue})"
            if citation.pages:
                journal_part += f", {citation.pages}"
            parts.append(journal_part + ".")

        elif citation.source_type == "book" and citation.publisher:
            parts.append(f"{citation.publisher}.")

        elif citation.source_type == "website" and citation.url:
            if citation.access_date:
                parts.append(
                    f"Retrieved {citation.access_date.strftime('%B %d, %Y')}, from {citation.url}"
                )
            else:
                parts.append(f"Retrieved from {citation.url}")

        # DOI
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}")

        return " ".join(parts)

    def _format_mla(self, citation: Citation) -> str:
        """Format citation in MLA style."""
        parts = []

        # Author
        if citation.authors:
            author = citation.authors[0]
            if "," in author:
                parts.append(f"{author}.")
            else:
                name_parts = author.split()
                if len(name_parts) >= 2:
                    parts.append(f"{name_parts[-1]}, {' '.join(name_parts[:-1])}.")
                else:
                    parts.append(f"{author}.")
        else:
            # Start with title if no author
            pass

        # Title
        if citation.source_type in ["journal", "website"]:
            parts.append(f'"{citation.title}."')
        else:
            parts.append(f"*{citation.title}*.")

        # Container (journal, website, etc.)
        if citation.journal:
            parts.append(f"*{citation.journal}*,")

        # Publication details
        if citation.volume:
            parts.append(f"vol. {citation.volume},")
        if citation.issue:
            parts.append(f"no. {citation.issue},")
        if citation.publication_date:
            parts.append(f"{citation.publication_date.year},")
        if citation.pages:
            parts.append(f"pp. {citation.pages}.")

        # Web source
        if citation.url:
            parts.append("Web.")
            if citation.access_date:
                parts.append(citation.access_date.strftime("%d %b %Y") + ".")

        return " ".join(parts)

    def _format_chicago(self, citation: Citation) -> str:
        """Format citation in Chicago style."""
        parts = []

        # Author
        if citation.authors:
            author = citation.authors[0]
            parts.append(f"{author}.")

        # Title
        if citation.source_type == "journal":
            parts.append(f'"{citation.title}."')
        else:
            parts.append(f"*{citation.title}*.")

        # Publication info
        if citation.journal:
            parts.append(f"*{citation.journal}*")
            if citation.volume:
                parts.append(f"{citation.volume}")
                if citation.issue:
                    parts.append(f", no. {citation.issue}")
            if citation.publication_date:
                parts.append(f"({citation.publication_date.year}):")
            if citation.pages:
                parts.append(f"{citation.pages}.")

        elif citation.publisher:
            if citation.publication_date:
                parts.append(f"{citation.publisher}, {citation.publication_date.year}.")

        # URL/DOI
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}.")
        elif citation.url:
            parts.append(
                f"Accessed {citation.access_date.strftime('%B %d, %Y') if citation.access_date else 'date'}. {citation.url}."
            )

        return " ".join(parts)

    def _format_ieee(self, citation: Citation) -> str:
        """Format citation in IEEE style."""
        parts = []

        # Authors
        if citation.authors:
            if len(citation.authors) == 1:
                parts.append(f"{citation.authors[0]},")
            else:
                author_str = (
                    ", ".join(citation.authors[:-1]) + f", and {citation.authors[-1]},"
                )
                parts.append(author_str)

        # Title
        parts.append(f'"{citation.title},"')

        # Source
        if citation.journal:
            parts.append(f"*{citation.journal}*,")
            if citation.volume:
                parts.append(f"vol. {citation.volume},")
            if citation.issue:
                parts.append(f"no. {citation.issue},")
            if citation.pages:
                parts.append(f"pp. {citation.pages},")

        # Date
        if citation.publication_date:
            parts.append(f"{citation.publication_date.strftime('%b. %Y')}.")

        # DOI/URL
        if citation.doi:
            parts.append(f"doi: {citation.doi}")
        elif citation.url:
            parts.append(f"[Online]. Available: {citation.url}")

        return " ".join(parts)

    def _format_harvard(self, citation: Citation) -> str:
        """Format citation in Harvard style."""
        parts = []

        # Author and year
        if citation.authors:
            author = citation.authors[0]
            year = (
                citation.publication_date.year if citation.publication_date else "n.d."
            )
            parts.append(f"{author} {year},")

        # Title
        if citation.source_type == "journal":
            parts.append(f"'{citation.title}',")
        else:
            parts.append(f"*{citation.title}*,")

        # Source details
        if citation.journal:
            parts.append(f"*{citation.journal}*,")
            if citation.volume:
                parts.append(f"vol. {citation.volume},")
            if citation.issue:
                parts.append(f"no. {citation.issue},")
            if citation.pages:
                parts.append(f"pp. {citation.pages}.")

        elif citation.publisher:
            parts.append(f"{citation.publisher}.")

        # URL
        if citation.url:
            parts.append(f"Available at: {citation.url}")
            if citation.access_date:
                parts.append(
                    f"(Accessed: {citation.access_date.strftime('%d %B %Y')})."
                )

        return " ".join(parts)
