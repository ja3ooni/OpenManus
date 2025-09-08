"""Tests for the enhanced web search tool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.exceptions import ResearchError
from app.research.models import ResearchFinding, ResearchSource, SourceType
from app.tool.enhanced_web_search import EnhancedSearchResponse, EnhancedWebSearch


class TestEnhancedWebSearch:
    """Test suite for enhanced web search tool."""

    @pytest.fixture
    def enhanced_search_tool(self):
        """Create enhanced web search tool instance."""
        return EnhancedWebSearch()

    @pytest.fixture
    def mock_research_finding(self):
        """Create mock research finding."""
        source = ResearchSource(
            name="example.com",
            url="https://example.com/article",
            source_type=SourceType.WEB,
            credibility_score=0.7,
            freshness_score=0.8,
            metadata={
                "title": "Test Article",
                "description": "Test description",
            },
        )

        return ResearchFinding(
            content="This is test content about machine learning algorithms.",
            source=source,
            query="machine learning",
            relevance_score=0.9,
            confidence_score=0.8,
            key_points=["Machine learning is important", "Algorithms are useful"],
            metadata={"search_position": 1},
        )

    @pytest.mark.asyncio
    async def test_basic_enhanced_search(
        self, enhanced_search_tool, mock_research_finding
    ):
        """Test basic enhanced search functionality."""
        with patch(
            "app.tool.enhanced_web_search.EnhancedWebSearchTool"
        ) as mock_tool_class:
            # Setup mock
            mock_tool = AsyncMock()
            mock_tool.__aenter__ = AsyncMock(return_value=mock_tool)
            mock_tool.__aexit__ = AsyncMock(return_value=None)
            mock_tool.enhanced_search = AsyncMock(return_value=[mock_research_finding])
            mock_tool_class.return_value = mock_tool

            # Execute search
            result = await enhanced_search_tool.execute(
                query="machine learning",
                domain="academic",
                num_results=5,
            )

            # Verify result
            assert isinstance(result, EnhancedSearchResponse)
            assert result.status == "success"
            assert result.query == "machine learning"
            assert len(result.results) == 1
            assert result.results[0].title == "Test Article"
            assert result.results[0].url == "https://example.com/article"
            assert result.results[0].relevance_score == 0.9
            assert result.results[0].credibility_score == 0.7
            assert result.results[0].freshness_score == 0.8
            assert result.results[0].source_type == "web"
            assert len(result.results[0].key_points) == 2

            # Verify mock was called correctly
            mock_tool.enhanced_search.assert_called_once_with(
                query="machine learning",
                domain="academic",
                num_results=5,
                enable_query_expansion=True,
                enable_content_extraction=True,
                credibility_threshold=0.3,
            )

    @pytest.mark.asyncio
    async def test_enhanced_search_with_custom_parameters(
        self, enhanced_search_tool, mock_research_finding
    ):
        """Test enhanced search with custom parameters."""
        with patch(
            "app.tool.enhanced_web_search.EnhancedWebSearchTool"
        ) as mock_tool_class:
            # Setup mock
            mock_tool = AsyncMock()
            mock_tool.__aenter__ = AsyncMock(return_value=mock_tool)
            mock_tool.__aexit__ = AsyncMock(return_value=None)
            mock_tool.enhanced_search = AsyncMock(return_value=[mock_research_finding])
            mock_tool_class.return_value = mock_tool

            # Execute search with custom parameters
            result = await enhanced_search_tool.execute(
                query="python programming",
                domain="technical",
                num_results=3,
                enable_query_expansion=False,
                enable_content_extraction=False,
                credibility_threshold=0.5,
            )

            # Verify result
            assert result.status == "success"
            assert result.query == "python programming"

            # Verify mock was called with custom parameters
            mock_tool.enhanced_search.assert_called_once_with(
                query="python programming",
                domain="technical",
                num_results=3,
                enable_query_expansion=False,
                enable_content_extraction=False,
                credibility_threshold=0.5,
            )

    @pytest.mark.asyncio
    async def test_enhanced_search_with_ranking_criteria(self, enhanced_search_tool):
        """Test enhanced search with custom ranking criteria."""
        # Create multiple mock findings with different scores
        findings = []
        for i in range(3):
            source = ResearchSource(
                name=f"example{i}.com",
                url=f"https://example{i}.com/article",
                source_type=SourceType.WEB,
                credibility_score=0.5 + i * 0.2,  # 0.5, 0.7, 0.9
                freshness_score=0.9 - i * 0.2,  # 0.9, 0.7, 0.5
                metadata={"title": f"Article {i}", "description": f"Description {i}"},
            )

            finding = ResearchFinding(
                content=f"Content {i}",
                source=source,
                query="test query",
                relevance_score=0.6 + i * 0.1,  # 0.6, 0.7, 0.8
                confidence_score=0.7,
                key_points=[f"Point {i}"],
                metadata={"search_position": i + 1},
            )
            findings.append(finding)

        with patch(
            "app.tool.enhanced_web_search.EnhancedWebSearchTool"
        ) as mock_tool_class:
            # Setup mock
            mock_tool = AsyncMock()
            mock_tool.__aenter__ = AsyncMock(return_value=mock_tool)
            mock_tool.__aexit__ = AsyncMock(return_value=None)
            mock_tool.enhanced_search = AsyncMock(return_value=findings)
            mock_tool_class.return_value = mock_tool

            # Execute search with custom ranking criteria (prioritize freshness)
            ranking_criteria = {"relevance": 0.2, "credibility": 0.2, "freshness": 0.6}
            result = await enhanced_search_tool.execute(
                query="test query",
                ranking_criteria=ranking_criteria,
            )

            # Verify results are re-ranked (first result should have highest freshness)
            assert result.status == "success"
            assert len(result.results) == 3
            # The first finding (index 0) has the highest freshness score (0.9)
            assert result.results[0].title == "Article 0"

    @pytest.mark.asyncio
    async def test_multi_domain_search(
        self, enhanced_search_tool, mock_research_finding
    ):
        """Test multi-domain search functionality."""
        with patch(
            "app.tool.enhanced_web_search.EnhancedWebSearchTool"
        ) as mock_tool_class:
            # Setup mock
            mock_tool = AsyncMock()
            mock_tool.__aenter__ = AsyncMock(return_value=mock_tool)
            mock_tool.__aexit__ = AsyncMock(return_value=None)
            mock_tool.search_with_ranking = AsyncMock(
                return_value=[mock_research_finding]
            )
            mock_tool_class.return_value = mock_tool

            # Execute multi-domain search
            result = await enhanced_search_tool.multi_domain_search(
                query="artificial intelligence",
                domains=["academic", "news"],
                num_results=5,
            )

            # Verify result
            assert result.status == "success"
            assert result.query == "artificial intelligence"
            assert result.domains_searched == ["academic", "news"]
            assert len(result.results) == 1
            assert result.results[0].domain == "multi-domain"

            # Verify mock was called correctly
            mock_tool.search_with_ranking.assert_called_once_with(
                query="artificial intelligence",
                domains=["academic", "news"],
                num_results=5,
                ranking_criteria=None,
            )

    @pytest.mark.asyncio
    async def test_multi_domain_search_default_domains(
        self, enhanced_search_tool, mock_research_finding
    ):
        """Test multi-domain search with default domains."""
        with patch(
            "app.tool.enhanced_web_search.EnhancedWebSearchTool"
        ) as mock_tool_class:
            # Setup mock
            mock_tool = AsyncMock()
            mock_tool.__aenter__ = AsyncMock(return_value=mock_tool)
            mock_tool.__aexit__ = AsyncMock(return_value=None)
            mock_tool.search_with_ranking = AsyncMock(
                return_value=[mock_research_finding]
            )
            mock_tool_class.return_value = mock_tool

            # Execute multi-domain search without specifying domains
            result = await enhanced_search_tool.multi_domain_search(
                query="test query",
                num_results=3,
            )

            # Verify default domains were used
            assert result.domains_searched == [
                "general",
                "academic",
                "news",
                "technical",
            ]

            # Verify mock was called with default domains
            mock_tool.search_with_ranking.assert_called_once_with(
                query="test query",
                domains=["general", "academic", "news", "technical"],
                num_results=3,
                ranking_criteria=None,
            )

    @pytest.mark.asyncio
    async def test_enhanced_search_research_error(self, enhanced_search_tool):
        """Test enhanced search handling of research errors."""
        with patch(
            "app.tool.enhanced_web_search.EnhancedWebSearchTool"
        ) as mock_tool_class:
            # Setup mock to raise ResearchError
            mock_tool = AsyncMock()
            mock_tool.__aenter__ = AsyncMock(return_value=mock_tool)
            mock_tool.__aexit__ = AsyncMock(return_value=None)
            mock_tool.enhanced_search = AsyncMock(
                side_effect=ResearchError("Search failed")
            )
            mock_tool_class.return_value = mock_tool

            # Execute search
            result = await enhanced_search_tool.execute(query="test query")

            # Verify error handling
            assert result.error == "Research error: Search failed"
            assert result.status is None  # Error response doesn't set status
            assert len(result.results) == 0
            assert result.total_found == 0

    @pytest.mark.asyncio
    async def test_enhanced_search_unexpected_error(self, enhanced_search_tool):
        """Test enhanced search handling of unexpected errors."""
        with patch(
            "app.tool.enhanced_web_search.EnhancedWebSearchTool"
        ) as mock_tool_class:
            # Setup mock to raise unexpected error
            mock_tool = AsyncMock()
            mock_tool.__aenter__ = AsyncMock(return_value=mock_tool)
            mock_tool.__aexit__ = AsyncMock(return_value=None)
            mock_tool.enhanced_search = AsyncMock(
                side_effect=ValueError("Unexpected error")
            )
            mock_tool_class.return_value = mock_tool

            # Execute search
            result = await enhanced_search_tool.execute(query="test query")

            # Verify error handling
            assert result.error == "Search failed: Unexpected error"
            assert len(result.results) == 0
            assert result.total_found == 0

    @pytest.mark.asyncio
    async def test_multi_domain_search_error(self, enhanced_search_tool):
        """Test multi-domain search error handling."""
        with patch(
            "app.tool.enhanced_web_search.EnhancedWebSearchTool"
        ) as mock_tool_class:
            # Setup mock to raise error
            mock_tool = AsyncMock()
            mock_tool.__aenter__ = AsyncMock(return_value=mock_tool)
            mock_tool.__aexit__ = AsyncMock(return_value=None)
            mock_tool.search_with_ranking = AsyncMock(
                side_effect=Exception("Search failed")
            )
            mock_tool_class.return_value = mock_tool

            # Execute multi-domain search
            result = await enhanced_search_tool.multi_domain_search(query="test query")

            # Verify error handling
            assert result.error == "Multi-domain search failed: Search failed"
            assert len(result.results) == 0
            assert result.total_found == 0

    def test_tool_parameters(self, enhanced_search_tool):
        """Test tool parameter definitions."""
        params = enhanced_search_tool.parameters

        # Verify required parameters
        assert "query" in params["required"]

        # Verify parameter properties
        properties = params["properties"]
        assert "query" in properties
        assert "domain" in properties
        assert "num_results" in properties
        assert "enable_query_expansion" in properties
        assert "enable_content_extraction" in properties
        assert "credibility_threshold" in properties
        assert "ranking_criteria" in properties

        # Verify domain enum values
        assert properties["domain"]["enum"] == [
            "academic",
            "news",
            "technical",
            "general",
        ]

        # Verify default values
        assert properties["domain"]["default"] == "general"
        assert properties["num_results"]["default"] == 10
        assert properties["enable_query_expansion"]["default"] is True
        assert properties["enable_content_extraction"]["default"] is True
        assert properties["credibility_threshold"]["default"] == 0.3

    def test_tool_metadata(self, enhanced_search_tool):
        """Test tool metadata."""
        assert enhanced_search_tool.name == "enhanced_web_search"
        assert "Advanced web search tool" in enhanced_search_tool.description
        assert "domain-specific strategies" in enhanced_search_tool.description
        assert "relevance scoring" in enhanced_search_tool.description

    @pytest.mark.asyncio
    async def test_rerank_results(self, enhanced_search_tool):
        """Test result re-ranking functionality."""
        from app.tool.enhanced_web_search import EnhancedSearchResult

        # Create test results with different scores
        results = [
            EnhancedSearchResult(
                title="Result 1",
                url="https://example1.com",
                description="Description 1",
                content_preview="Content 1",
                relevance_score=0.8,
                credibility_score=0.6,
                freshness_score=0.4,
                source_type="web",
                key_points=[],
                domain="general",
            ),
            EnhancedSearchResult(
                title="Result 2",
                url="https://example2.com",
                description="Description 2",
                content_preview="Content 2",
                relevance_score=0.6,
                credibility_score=0.9,
                freshness_score=0.8,
                source_type="academic",
                key_points=[],
                domain="academic",
            ),
        ]

        # Re-rank with credibility priority
        ranking_criteria = {"relevance": 0.1, "credibility": 0.8, "freshness": 0.1}
        reranked = await enhanced_search_tool._rerank_results(results, ranking_criteria)

        # Result 2 should be first (higher credibility)
        assert reranked[0].title == "Result 2"
        assert reranked[1].title == "Result 1"

    @pytest.mark.asyncio
    async def test_empty_results(self, enhanced_search_tool):
        """Test handling of empty search results."""
        with patch(
            "app.tool.enhanced_web_search.EnhancedWebSearchTool"
        ) as mock_tool_class:
            # Setup mock to return empty results
            mock_tool = AsyncMock()
            mock_tool.__aenter__ = AsyncMock(return_value=mock_tool)
            mock_tool.__aexit__ = AsyncMock(return_value=None)
            mock_tool.enhanced_search = AsyncMock(return_value=[])
            mock_tool_class.return_value = mock_tool

            # Execute search
            result = await enhanced_search_tool.execute(query="test query")

            # Verify empty results are handled correctly
            assert result.status == "success"
            assert len(result.results) == 0
            assert result.total_found == 0
            assert "Enhanced search results for 'test query'" in result.output
            assert "Total results found: 0" in result.output


# Integration tests (require actual network access)
class TestEnhancedWebSearchIntegration:
    """Integration tests for enhanced web search (requires network access)."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_search_academic(self):
        """Test real academic search (requires network)."""
        tool = EnhancedWebSearch()

        result = await tool.execute(
            query="machine learning neural networks",
            domain="academic",
            num_results=3,
            enable_content_extraction=False,  # Faster for testing
        )

        # Basic validation
        assert result.status == "success" or result.error is not None
        if result.status == "success":
            assert result.query == "machine learning neural networks"
            assert result.domains_searched == ["academic"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_multi_domain_search(self):
        """Test real multi-domain search (requires network)."""
        tool = EnhancedWebSearch()

        result = await tool.multi_domain_search(
            query="artificial intelligence ethics",
            domains=["academic", "news"],
            num_results=2,
        )

        # Basic validation
        assert result.status == "success" or result.error is not None
        if result.status == "success":
            assert result.query == "artificial intelligence ethics"
            assert set(result.domains_searched) == {"academic", "news"}


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
