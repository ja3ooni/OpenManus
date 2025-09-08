"""Research orchestrator for coordinating multi-source information gathering."""

import asyncio
import logging
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..exceptions import ResearchError
from .models import (
    ConflictType,
    CrossReference,
    InformationConflict,
    ResearchDepth,
    ResearchFinding,
    ResearchQuery,
    ResearchResult,
    ResearchSource,
    SourceType,
)

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Orchestrates multi-source research with parallel execution and result aggregation."""

    def __init__(
        self,
        max_concurrent_searches: int = 5,
        default_timeout: int = 120,
        enable_deduplication: bool = True,
    ):
        """Initialize the research orchestrator."""
        self.max_concurrent_searches = max_concurrent_searches
        self.default_timeout = default_timeout
        self.enable_deduplication = enable_deduplication
        self._search_tools: Dict[SourceType, Any] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_searches)

    def register_search_tool(self, source_type: SourceType, tool: Any) -> None:
        """Register a search tool for a specific source type."""
        self._search_tools[source_type] = tool
        logger.info(f"Registered search tool for {source_type}")

    async def conduct_research(
        self,
        query: ResearchQuery,
        timeout: Optional[int] = None,
    ) -> ResearchResult:
        """Conduct comprehensive research using multiple sources."""
        start_time = datetime.utcnow()
        timeout = timeout or self.default_timeout

        try:
            logger.info(f"Starting research for query: {query.query}")

            # Determine sources to use
            sources = await self._determine_sources(query)

            # Conduct parallel searches
            findings = await self._conduct_parallel_searches(query, sources, timeout)

            # Apply deduplication if enabled
            if self.enable_deduplication:
                findings = await self._deduplicate_findings(findings)

            # Perform cross-referencing if enabled
            cross_references = []
            if query.enable_cross_referencing:
                cross_references = await self._cross_reference_findings(findings)

            # Detect conflicts if enabled
            conflicts = []
            if query.enable_conflict_detection:
                conflicts = await self._detect_conflicts(findings)

            # Generate summary and insights
            summary = await self._generate_summary(query.query, findings)
            key_insights = await self._extract_key_insights(findings)

            # Calculate scores
            confidence_score = self._calculate_confidence_score(findings)
            completeness_score = self._calculate_completeness_score(
                query, findings, sources
            )

            # Create result
            duration = (datetime.utcnow() - start_time).total_seconds()

            result = ResearchResult(
                query=query.query,
                findings=findings,
                cross_references=cross_references,
                conflicts=conflicts,
                summary=summary,
                key_insights=key_insights,
                confidence_score=confidence_score,
                completeness_score=completeness_score,
                sources_used=sources,
                research_depth=query.depth,
                duration_seconds=duration,
                metadata={
                    "total_sources_attempted": len(sources),
                    "successful_sources": len(set(f.source.id for f in findings)),
                    "deduplication_enabled": self.enable_deduplication,
                },
            )

            logger.info(
                f"Research completed in {duration:.2f}s with {len(findings)} findings"
            )
            return result

        except asyncio.TimeoutError:
            raise ResearchError(f"Research timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            raise ResearchError(f"Research failed: {str(e)}") from e

    async def _determine_sources(self, query: ResearchQuery) -> List[ResearchSource]:
        """Determine which sources to use for the research query."""
        sources = []
        source_types = query.source_types or list(self._search_tools.keys())

        for source_type in source_types:
            if source_type in self._search_tools:
                source = ResearchSource(
                    name=f"{source_type.value}_search",
                    source_type=source_type,
                    domain=query.domain_focus,
                    credibility_score=self._get_default_credibility(source_type),
                    metadata={"tool_available": True},
                )
                sources.append(source)

        if query.max_sources > 0:
            sources = sources[: query.max_sources]

        logger.info(f"Selected {len(sources)} sources for research")
        return sources

    async def _conduct_parallel_searches(
        self, query: ResearchQuery, sources: List[ResearchSource], timeout: int
    ) -> List[ResearchFinding]:
        """Conduct searches across multiple sources in parallel."""
        search_tasks = []

        for source in sources:
            if source.source_type in self._search_tools:
                task = asyncio.create_task(self._search_source(query, source, timeout))
                search_tasks.append(task)

        if not search_tasks:
            logger.warning("No search tools available")
            return []

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*search_tasks, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Parallel searches timed out after {timeout} seconds")
            for task in search_tasks:
                if not task.done():
                    task.cancel()
            raise

        findings = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Search failed for source {sources[i].name}: {str(result)}"
                )
            elif isinstance(result, list):
                findings.extend(result)

        logger.info(f"Collected {len(findings)} findings from parallel searches")
        return findings

    async def _search_source(
        self, query: ResearchQuery, source: ResearchSource, timeout: int
    ) -> List[ResearchFinding]:
        """Search a single source for information."""
        async with self._semaphore:
            try:
                tool = self._search_tools[source.source_type]

                search_params = {
                    "query": query.query,
                    "max_results": min(query.max_findings, 20),
                    "language": query.languages[0] if query.languages else "en",
                }

                if query.time_range_days:
                    search_params["time_range"] = query.time_range_days

                search_timeout = min(timeout // 2, 60)
                results = await asyncio.wait_for(
                    tool.search(**search_params), timeout=search_timeout
                )

                findings = []
                for result in results:
                    finding = self._convert_to_finding(result, source, query.query)
                    if finding and finding.relevance_score >= 0.3:
                        findings.append(finding)

                logger.debug(f"Found {len(findings)} findings from {source.name}")
                return findings

            except asyncio.TimeoutError:
                logger.warning(f"Search timed out for source {source.name}")
                return []
            except Exception as e:
                logger.error(f"Search failed for source {source.name}: {str(e)}")
                return []

    def _convert_to_finding(
        self, result: Dict[str, Any], source: ResearchSource, query: str
    ) -> Optional[ResearchFinding]:
        """Convert a search result to a ResearchFinding."""
        try:
            content = result.get("content", result.get("text", ""))
            if not content:
                return None

            relevance_score = self._calculate_relevance_score(content, query)
            key_points = self._extract_key_points(content)

            finding = ResearchFinding(
                content=content,
                source=source,
                query=query,
                relevance_score=relevance_score,
                confidence_score=source.credibility_score,
                key_points=key_points,
                context=result.get("context"),
                metadata={
                    "url": result.get("url"),
                    "title": result.get("title"),
                    "published_date": result.get("published_date"),
                    "author": result.get("author"),
                },
            )

            return finding

        except Exception as e:
            logger.error(f"Failed to convert result to finding: {str(e)}")
            return None

    async def _deduplicate_findings(
        self, findings: List[ResearchFinding]
    ) -> List[ResearchFinding]:
        """Remove duplicate findings based on content similarity."""
        if not findings:
            return findings

        deduplicated = []
        seen_content = set()

        for finding in findings:
            content_hash = hash(finding.content.lower().strip())
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                deduplicated.append(finding)

        logger.info(f"Deduplication: {len(findings)} -> {len(deduplicated)} findings")
        return deduplicated

    async def _cross_reference_findings(
        self, findings: List[ResearchFinding]
    ) -> List[CrossReference]:
        """Identify cross-references between findings."""
        cross_references = []

        for i, finding1 in enumerate(findings):
            for j, finding2 in enumerate(findings[i + 1 :], i + 1):
                similarity = self._calculate_content_similarity(
                    finding1.content, finding2.content
                )

                if similarity > 0.6:
                    cross_ref = CrossReference(
                        finding_ids=[finding1.id, finding2.id],
                        relationship_type=(
                            "supports" if similarity > 0.8 else "relates_to"
                        ),
                        strength=similarity,
                        confidence=min(
                            finding1.confidence_score, finding2.confidence_score
                        ),
                        description=f"Content similarity: {similarity:.2f}",
                    )
                    cross_references.append(cross_ref)

        logger.info(f"Found {len(cross_references)} cross-references")
        return cross_references

    async def _detect_conflicts(
        self, findings: List[ResearchFinding]
    ) -> List[InformationConflict]:
        """Detect conflicts between findings."""
        conflicts = []
        contradiction_pairs = [
            ("increase", "decrease"),
            ("rise", "fall"),
            ("positive", "negative"),
            ("true", "false"),
            ("yes", "no"),
            ("support", "oppose"),
        ]

        for i, finding1 in enumerate(findings):
            for j, finding2 in enumerate(findings[i + 1 :], i + 1):
                content1_lower = finding1.content.lower()
                content2_lower = finding2.content.lower()

                for term1, term2 in contradiction_pairs:
                    if (term1 in content1_lower and term2 in content2_lower) or (
                        term2 in content1_lower and term1 in content2_lower
                    ):

                        conflict = InformationConflict(
                            conflicting_findings=[finding1.id, finding2.id],
                            conflict_type=ConflictType.DIRECT_CONTRADICTION,
                            severity=0.7,
                            description=f"Contradictory terms detected: {term1} vs {term2}",
                            resolution_suggestions=[
                                "Check source credibility",
                                "Look for temporal differences",
                                "Consider different contexts",
                            ],
                        )
                        conflicts.append(conflict)
                        break

        logger.info(f"Detected {len(conflicts)} potential conflicts")
        return conflicts

    async def _generate_summary(
        self, query: str, findings: List[ResearchFinding]
    ) -> str:
        """Generate a summary of the research findings."""
        if not findings:
            return f"No findings were discovered for the query: {query}"

        high_confidence_findings = [f for f in findings if f.confidence_score > 0.7]

        summary_parts = [
            f"Research Summary for: {query}",
            f"Total findings: {len(findings)}",
            f"High confidence findings: {len(high_confidence_findings)}",
            f"Sources used: {len(set(f.source.name for f in findings))}",
        ]

        top_findings = sorted(
            findings, key=lambda x: x.relevance_score * x.confidence_score, reverse=True
        )[:3]

        if top_findings:
            summary_parts.append("\nKey findings:")
            for i, finding in enumerate(top_findings, 1):
                summary_parts.append(f"{i}. {finding.content[:200]}...")

        return "\n".join(summary_parts)

    async def _extract_key_insights(self, findings: List[ResearchFinding]) -> List[str]:
        """Extract key insights from the findings."""
        insights = []

        if not findings:
            return insights

        all_key_points = []
        for finding in findings:
            all_key_points.extend(finding.key_points)

        point_counts = Counter(all_key_points)

        for point, count in point_counts.most_common(5):
            if count > 1:
                insights.append(f"{point} (mentioned {count} times)")

        return insights

    def _calculate_confidence_score(self, findings: List[ResearchFinding]) -> float:
        """Calculate overall confidence score for the research."""
        if not findings:
            return 0.0

        total_weight = 0
        weighted_sum = 0

        for finding in findings:
            weight = finding.relevance_score
            weighted_sum += finding.confidence_score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_completeness_score(
        self,
        query: ResearchQuery,
        findings: List[ResearchFinding],
        sources: List[ResearchSource],
    ) -> float:
        """Calculate completeness score for the research."""
        if not sources:
            return 0.0

        source_coverage = len(set(f.source.id for f in findings)) / len(sources)
        finding_count_factor = min(len(findings) / query.max_findings, 1.0)
        depth_factor = {
            ResearchDepth.QUICK: 0.5,
            ResearchDepth.STANDARD: 0.7,
            ResearchDepth.COMPREHENSIVE: 0.9,
            ResearchDepth.DEEP: 1.0,
        }.get(query.depth, 0.7)

        return (source_coverage + finding_count_factor + depth_factor) / 3

    def _get_default_credibility(self, source_type: SourceType) -> float:
        """Get default credibility score for a source type."""
        credibility_map = {
            SourceType.ACADEMIC: 0.9,
            SourceType.GOVERNMENT: 0.8,
            SourceType.NEWS: 0.7,
            SourceType.TECHNICAL: 0.8,
            SourceType.WEB: 0.5,
            SourceType.SOCIAL: 0.3,
            SourceType.COMMERCIAL: 0.4,
        }
        return credibility_map.get(source_type, 0.5)

    def _calculate_relevance_score(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(content_words))
        return min(overlap / len(query_words), 1.0)

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        sentences = content.split(".")
        key_points = []

        for sentence in sentences[:3]:
            sentence = sentence.strip()
            if len(sentence) > 20:
                key_points.append(sentence)

        return key_points

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0
