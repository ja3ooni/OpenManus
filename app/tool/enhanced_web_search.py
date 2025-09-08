"""Enhanced web search tool that extends the base web search with advanced features."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..exceptions import ResearchError
from ..research.enhanced_search import EnhancedWebSearchTool
from ..research.models import ResearchFinding
from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class EnhancedSearchResult(BaseModel):
    """Enhanced search result with additional metadata."""

    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    description: str = Field(description="Description or snippet")
    content_preview: str = Field(description="Preview of extracted content")
    relevance_score: float = Field(description="Relevance score (0-1)")
    credibility_score: float = Field(description="Source credibility score (0-1)")
    freshness_score: float = Field(description="Content freshness score (0-1)")
    source_type: str = Field(
        description="Type of source (academic, news, technical, etc.)"
    )
    key_points: List[str] = Field(
        default_factory=list, description="Key points extracted from content"
    )
    domain: str = Field(description="Domain strategy used")


class EnhancedSearchResponse(ToolResult):
    """Response from enhanced web search."""

    query: str = Field(description="Original search query")
    results: List[EnhancedSearchResult] = Field(
        default_factory=list, description="Enhanced search results"
    )
    total_found: int = Field(description="Total number of results found")
    domains_searched: List[str] = Field(
        default_factory=list, description="Domain strategies used"
    )
    query_expansions: List[str] = Field(
        default_factory=list, description="Query expansions used"
    )

    def model_post_init(self, __context: Any) -> None:
        """Populate output field after model initialization."""
        if self.error:
            return

        result_text = [f"Enhanced search results for '{self.query}':\n"]

        if self.query_expansions:
            result_text.append(
                f"Query expansions used: {', '.join(self.query_expansions[:3])}\n"
            )

        if self.domains_searched:
            result_text.append(
                f"Domains searched: {', '.join(self.domains_searched)}\n"
            )

        for i, result in enumerate(self.results, 1):
            result_text.extend(
                [
                    f"\n{i}. {result.title}",
                    f"   URL: {result.url}",
                    f"   Source Type: {result.source_type}",
                    f"   Scores: Relevance={result.relevance_score:.2f}, Credibility={result.credibility_score:.2f}, Freshness={result.freshness_score:.2f}",
                    f"   Description: {result.description}",
                ]
            )

            if result.content_preview:
                preview = (
                    result.content_preview[:300] + "..."
                    if len(result.content_preview) > 300
                    else result.content_preview
                )
                result_text.append(f"   Content Preview: {preview}")

            if result.key_points:
                result_text.append(f"   Key Points:")
                for point in result.key_points[:3]:  # Show top 3 key points
                    result_text.append(f"     • {point}")

        result_text.append(f"\nTotal results found: {self.total_found}")

        self.output = "\n".join(result_text)


class EnhancedWebSearch(BaseTool):
    """Enhanced web search tool with domain-specific strategies and intelligent ranking."""

    name: str = "enhanced_web_search"
    description: str = """Advanced web search tool with intelligent query expansion, domain-specific strategies,
    and comprehensive content analysis. This tool provides enhanced search results with relevance scoring,
    credibility assessment, and key point extraction. It supports multiple domain strategies including
    academic, news, technical, and general web content."""

    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to execute.",
            },
            "domain": {
                "type": "string",
                "description": "(optional) Domain strategy to use: 'academic', 'news', 'technical', or 'general'. Default is 'general'.",
                "enum": ["academic", "news", "technical", "general"],
                "default": "general",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) Number of results to return. Default is 10.",
                "default": 10,
                "minimum": 1,
                "maximum": 20,
            },
            "enable_query_expansion": {
                "type": "boolean",
                "description": "(optional) Whether to expand the query with related terms. Default is true.",
                "default": True,
            },
            "enable_content_extraction": {
                "type": "boolean",
                "description": "(optional) Whether to extract full content from result pages. Default is true.",
                "default": True,
            },
            "credibility_threshold": {
                "type": "number",
                "description": "(optional) Minimum credibility threshold (0-1). Default is 0.3.",
                "default": 0.3,
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "ranking_criteria": {
                "type": "object",
                "description": "(optional) Custom ranking weights for relevance, credibility, and freshness.",
                "properties": {
                    "relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "credibility": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "freshness": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "default": {"relevance": 0.5, "credibility": 0.3, "freshness": 0.2},
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        super().__init__()
        self.enhanced_search_tool = None

    async def execute(
        self,
        query: str,
        domain: str = "general",
        num_results: int = 10,
        enable_query_expansion: bool = True,
        enable_content_extraction: bool = True,
        credibility_threshold: float = 0.3,
        ranking_criteria: Optional[Dict[str, float]] = None,
    ) -> EnhancedSearchResponse:
        """
        Execute enhanced web search with advanced features.

        Args:
            query: The search query to execute
            domain: Domain strategy to use (academic, news, technical, general)
            num_results: Number of results to return
            enable_query_expansion: Whether to expand the query
            enable_content_extraction: Whether to extract full content
            credibility_threshold: Minimum credibility threshold
            ranking_criteria: Custom ranking weights

        Returns:
            Enhanced search response with detailed results
        """
        try:
            logger.info(
                f"Starting enhanced web search for '{query}' in domain '{domain}'"
            )

            # Initialize enhanced search tool with async context
            async with EnhancedWebSearchTool() as search_tool:
                # Perform enhanced search
                findings = await search_tool.enhanced_search(
                    query=query,
                    domain=domain,
                    num_results=num_results,
                    enable_query_expansion=enable_query_expansion,
                    enable_content_extraction=enable_content_extraction,
                    credibility_threshold=credibility_threshold,
                )

                # Convert findings to enhanced search results
                results = []
                query_expansions = []

                for finding in findings:
                    result = EnhancedSearchResult(
                        title=finding.source.metadata.get("title", "No title"),
                        url=finding.source.url,
                        description=finding.source.metadata.get("description", ""),
                        content_preview=(
                            finding.content[:500] if finding.content else ""
                        ),
                        relevance_score=finding.relevance_score,
                        credibility_score=finding.source.credibility_score,
                        freshness_score=finding.source.freshness_score,
                        source_type=finding.source.source_type.value,
                        key_points=finding.key_points,
                        domain=domain,
                    )
                    results.append(result)

                # If ranking criteria provided, re-rank results
                if ranking_criteria:
                    results = await self._rerank_results(results, ranking_criteria)

                return EnhancedSearchResponse(
                    status="success",
                    query=query,
                    results=results,
                    total_found=len(results),
                    domains_searched=[domain],
                    query_expansions=query_expansions,
                )

        except ResearchError as e:
            logger.error(f"Research error in enhanced search: {str(e)}")
            return EnhancedSearchResponse(
                query=query,
                error=f"Research error: {str(e)}",
                results=[],
                total_found=0,
                domains_searched=[],
                query_expansions=[],
            )
        except Exception as e:
            logger.error(f"Unexpected error in enhanced search: {str(e)}")
            return EnhancedSearchResponse(
                query=query,
                error=f"Search failed: {str(e)}",
                results=[],
                total_found=0,
                domains_searched=[],
                query_expansions=[],
            )

    async def _rerank_results(
        self, results: List[EnhancedSearchResult], ranking_criteria: Dict[str, float]
    ) -> List[EnhancedSearchResult]:
        """Re-rank results based on custom criteria."""
        # Normalize weights to sum to 1
        total_weight = sum(ranking_criteria.values())
        if total_weight > 0:
            normalized_criteria = {
                k: v / total_weight for k, v in ranking_criteria.items()
            }
        else:
            normalized_criteria = {
                "relevance": 0.5,
                "credibility": 0.3,
                "freshness": 0.2,
            }

        # Calculate composite scores
        for result in results:
            composite_score = (
                result.relevance_score * normalized_criteria.get("relevance", 0.5)
                + result.credibility_score * normalized_criteria.get("credibility", 0.3)
                + result.freshness_score * normalized_criteria.get("freshness", 0.2)
            )
            # Store composite score for sorting
            result.composite_score = composite_score

        # Sort by composite score
        results.sort(key=lambda x: getattr(x, "composite_score", 0), reverse=True)

        # Remove the temporary composite_score attribute
        for result in results:
            if hasattr(result, "composite_score"):
                delattr(result, "composite_score")

        return results

    async def multi_domain_search(
        self,
        query: str,
        domains: List[str] = None,
        num_results: int = 10,
        ranking_criteria: Optional[Dict[str, float]] = None,
    ) -> EnhancedSearchResponse:
        """
        Search across multiple domains and combine results.

        Args:
            query: The search query
            domains: List of domains to search (default: all domains)
            num_results: Number of results to return
            ranking_criteria: Custom ranking weights

        Returns:
            Combined enhanced search response
        """
        if domains is None:
            domains = ["general", "academic", "news", "technical"]

        try:
            logger.info(
                f"Starting multi-domain search for '{query}' across domains: {domains}"
            )

            async with EnhancedWebSearchTool() as search_tool:
                # Use the search_with_ranking method for multi-domain search
                findings = await search_tool.search_with_ranking(
                    query=query,
                    domains=domains,
                    num_results=num_results,
                    ranking_criteria=ranking_criteria,
                )

                # Convert findings to enhanced search results
                results = []
                for finding in findings:
                    result = EnhancedSearchResult(
                        title=finding.source.metadata.get("title", "No title"),
                        url=finding.source.url,
                        description=finding.source.metadata.get("description", ""),
                        content_preview=(
                            finding.content[:500] if finding.content else ""
                        ),
                        relevance_score=finding.relevance_score,
                        credibility_score=finding.source.credibility_score,
                        freshness_score=finding.source.freshness_score,
                        source_type=finding.source.source_type.value,
                        key_points=finding.key_points,
                        domain="multi-domain",
                    )
                    results.append(result)

                return EnhancedSearchResponse(
                    status="success",
                    query=query,
                    results=results,
                    total_found=len(results),
                    domains_searched=domains,
                    query_expansions=[],  # Query expansions are handled internally
                )

        except Exception as e:
            logger.error(f"Multi-domain search failed: {str(e)}")
            return EnhancedSearchResponse(
                query=query,
                error=f"Multi-domain search failed: {str(e)}",
                results=[],
                total_found=0,
                domains_searched=domains,
                query_expansions=[],
            )


# Example usage and testing
if __name__ == "__main__":

    async def test_enhanced_search():
        """Test the enhanced web search tool."""
        tool = EnhancedWebSearch()

        # Test basic search
        print("Testing basic enhanced search...")
        result = await tool.execute(
            query="machine learning algorithms",
            domain="academic",
            num_results=5,
        )
        print(result.output)
        print("\n" + "=" * 50 + "\n")

        # Test multi-domain search
        print("Testing multi-domain search...")
        result = await tool.multi_domain_search(
            query="artificial intelligence ethics",
            domains=["academic", "news"],
            num_results=5,
        )
        print(result.output)

    # Run the test
    asyncio.run(test_enhanced_search())
