#!/usr/bin/env python3
"""Simple test script for enhanced web search tool."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

# Add current directory to path
sys.path.insert(0, ".")


async def test_enhanced_search_tool():
    """Test the enhanced web search tool with mocked dependencies."""
    print("Testing Enhanced Web Search Tool...")

    try:
        # Import the models first
        from app.research.models import ResearchFinding, ResearchSource, SourceType

        print("✓ Research models imported successfully")

        # Create mock research finding
        source = ResearchSource(
            name="example.com",
            url="https://example.com/article",
            source_type=SourceType.WEB,
            credibility_score=0.8,
            freshness_score=0.7,
            metadata={
                "title": "Test Article About Machine Learning",
                "description": "A comprehensive guide to machine learning algorithms",
            },
        )

        finding = ResearchFinding(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            source=source,
            query="machine learning",
            relevance_score=0.9,
            confidence_score=0.8,
            key_points=[
                "Machine learning uses algorithms to learn from data",
                "It's a subset of artificial intelligence",
                "Applications include image recognition and natural language processing",
            ],
            metadata={"search_position": 1},
        )
        print("✓ Mock research finding created successfully")

        # Test the enhanced web search tool (import directly to avoid browser_use dependency)
        # Mock the problematic imports first
        import sys
        from unittest.mock import MagicMock

        # Mock the enhanced search tool to avoid aiohttp dependency
        mock_enhanced_search = MagicMock()
        sys.modules["app.research.enhanced_search"] = mock_enhanced_search

        from app.tool.enhanced_web_search import (
            EnhancedSearchResponse,
            EnhancedWebSearch,
        )

        print("✓ Enhanced web search tool imported successfully")

        # Create tool instance
        tool = EnhancedWebSearch()
        print(f"✓ Tool created: {tool.name}")
        print(f"✓ Tool description: {tool.description[:100]}...")

        # Test tool parameters
        params = tool.parameters
        assert "query" in params["required"]
        assert "domain" in params["properties"]
        assert params["properties"]["domain"]["enum"] == [
            "academic",
            "news",
            "technical",
            "general",
        ]
        print("✓ Tool parameters validated")

        # Test response model
        response = EnhancedSearchResponse(
            status="success",
            query="test query",
            results=[],
            total_found=0,
            domains_searched=["general"],
            query_expansions=["test query expanded"],
        )
        print("✓ Enhanced search response model works")

        # Test with mock data
        from app.tool.enhanced_web_search import EnhancedSearchResult

        result = EnhancedSearchResult(
            title=finding.source.metadata["title"],
            url=finding.source.url,
            description=finding.source.metadata["description"],
            content_preview=finding.content[:200],
            relevance_score=finding.relevance_score,
            credibility_score=finding.source.credibility_score,
            freshness_score=finding.source.freshness_score,
            source_type=finding.source.source_type.value,
            key_points=finding.key_points,
            domain="general",
        )

        response_with_results = EnhancedSearchResponse(
            status="success",
            query="machine learning",
            results=[result],
            total_found=1,
            domains_searched=["general"],
            query_expansions=[],
        )

        # Test output generation
        output = response_with_results.output
        assert "Enhanced search results for 'machine learning'" in output
        assert "Test Article About Machine Learning" in output
        assert "Relevance=0.90" in output
        assert "Total results found: 1" in output
        print("✓ Response output generation works correctly")

        print("\n" + "=" * 50)
        print("SAMPLE OUTPUT:")
        print("=" * 50)
        print(output)
        print("=" * 50)

        print("\n✅ All tests passed! Enhanced Web Search Tool is working correctly.")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_enhanced_search_tool())
    sys.exit(0 if success else 1)
