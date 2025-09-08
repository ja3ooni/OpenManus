"""Source validation and credibility scoring system."""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp

from ..exceptions import ResearchError
from .models import CredibilityLevel, ResearchSource, SourceType

logger = logging.getLogger(__name__)


class SourceValidator:
    """Validates and scores information sources for credibility and freshness."""

    def __init__(
        self,
        enable_domain_authority: bool = True,
        enable_freshness_check: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """Initialize the source validator."""
        self.enable_domain_authority = enable_domain_authority
        self.enable_freshness_check = enable_freshness_check
        self.cache_ttl_hours = cache_ttl_hours
        self._domain_cache: Dict[str, Dict[str, Any]] = {}
        self._reputation_cache: Dict[str, float] = {}

        # Known high-credibility domains
        self.high_credibility_domains = {
            # Academic and research
            "arxiv.org",
            "pubmed.ncbi.nlm.nih.gov",
            "scholar.google.com",
            "researchgate.net",
            "jstor.org",
            "springer.com",
            "nature.com",
            "science.org",
            "cell.com",
            "plos.org",
            "acm.org",
            "ieee.org",
            # Government and official
            "gov",
            "edu",
            "who.int",
            "cdc.gov",
            "fda.gov",
            "nih.gov",
            "nasa.gov",
            "noaa.gov",
            "census.gov",
            "sec.gov",
            "ftc.gov",
            # Reputable news and media
            "reuters.com",
            "ap.org",
            "bbc.com",
            "npr.org",
            "pbs.org",
            "wsj.com",
            "nytimes.com",
            "washingtonpost.com",
            "economist.com",
            # Technical documentation
            "docs.python.org",
            "developer.mozilla.org",
            "stackoverflow.com",
            "github.com",
            "microsoft.com/docs",
            "aws.amazon.com/documentation",
        }

        # Known low-credibility patterns
        self.low_credibility_patterns = {
            r".*\.blogspot\.com$",
            r".*\.wordpress\.com$",
            r".*\.medium\.com$",
            r".*\.substack\.com$",
            r".*\.tumblr\.com$",
            r".*clickbait.*",
            r".*fake.*news.*",
            r".*conspiracy.*",
        }

        # Bias indicators
        self.bias_indicators = {
            "high_bias": [
                "shocking",
                "unbelievable",
                "secret",
                "they don't want you to know",
                "mainstream media",
                "fake news",
                "conspiracy",
                "cover-up",
                "exclusive",
                "breaking",
                "urgent",
                "must read",
                "viral",
            ],
            "moderate_bias": [
                "allegedly",
                "reportedly",
                "sources say",
                "insider claims",
                "rumored",
                "speculation",
                "controversial",
                "debate",
            ],
        }

    async def validate_source_credibility(
        self, source: ResearchSource
    ) -> ResearchSource:
        """Validate and score source credibility."""
        try:
            logger.debug(f"Validating credibility for source: {source.name}")

            # Calculate domain authority score
            authority_score = await self._calculate_domain_authority(source)

            # Calculate reputation score
            reputation_score = await self._calculate_reputation_score(source)

            # Calculate bias score
            bias_score = await self._calculate_bias_score(source)

            # Combine scores for overall credibility
            credibility_score = self._combine_credibility_scores(
                authority_score, reputation_score, bias_score, source.source_type
            )

            # Determine credibility level
            credibility_level = self._determine_credibility_level(credibility_score)

            # Update source with calculated scores
            source.credibility_score = credibility_score
            source.credibility_level = credibility_level
            source.authority_score = authority_score
            source.bias_score = bias_score

            logger.debug(
                f"Source {source.name} credibility: {credibility_score:.2f} "
                f"({credibility_level})"
            )

            return source

        except Exception as e:
            logger.error(f"Failed to validate source credibility: {str(e)}")
            # Return source with default scores on error
            source.credibility_score = 0.5
            source.credibility_level = CredibilityLevel.UNKNOWN
            return source

    async def check_information_freshness(
        self, source: ResearchSource, content: Optional[str] = None
    ) -> ResearchSource:
        """Check and score information freshness."""
        try:
            logger.debug(f"Checking freshness for source: {source.name}")

            freshness_score = 0.5  # Default

            # Check last updated timestamp
            if source.last_updated:
                days_old = (datetime.utcnow() - source.last_updated).days
                freshness_score = self._calculate_temporal_score(days_old)

            # Check URL for date patterns if available
            elif source.url:
                url_freshness = await self._extract_date_from_url(source.url)
                if url_freshness:
                    freshness_score = url_freshness

            # Analyze content for temporal indicators
            if content and self.enable_freshness_check:
                content_freshness = self._analyze_content_freshness(content)
                freshness_score = max(freshness_score, content_freshness)

            source.freshness_score = freshness_score

            logger.debug(f"Source {source.name} freshness: {freshness_score:.2f}")
            return source

        except Exception as e:
            logger.error(f"Failed to check information freshness: {str(e)}")
            source.freshness_score = 0.5
            return source

    async def rank_sources(
        self, sources: List[ResearchSource], query: str
    ) -> List[ResearchSource]:
        """Rank sources based on credibility, freshness, and relevance."""
        try:
            logger.info(f"Ranking {len(sources)} sources")

            # Calculate composite scores for each source
            scored_sources = []
            for source in sources:
                composite_score = self._calculate_composite_score(source, query)
                scored_sources.append((source, composite_score))

            # Sort by composite score (descending)
            scored_sources.sort(key=lambda x: x[1], reverse=True)

            # Return ranked sources
            ranked_sources = [source for source, _ in scored_sources]

            logger.info(f"Sources ranked by composite score")
            return ranked_sources

        except Exception as e:
            logger.error(f"Failed to rank sources: {str(e)}")
            return sources

    async def filter_sources(
        self,
        sources: List[ResearchSource],
        min_credibility: float = 0.3,
        max_bias: float = 0.8,
        min_freshness: float = 0.2,
    ) -> List[ResearchSource]:
        """Filter sources based on quality thresholds."""
        try:
            logger.info(f"Filtering {len(sources)} sources")

            filtered_sources = []
            for source in sources:
                if (
                    source.credibility_score >= min_credibility
                    and source.bias_score <= max_bias
                    and source.freshness_score >= min_freshness
                ):
                    filtered_sources.append(source)

            logger.info(
                f"Filtered to {len(filtered_sources)} sources "
                f"(removed {len(sources) - len(filtered_sources)})"
            )

            return filtered_sources

        except Exception as e:
            logger.error(f"Failed to filter sources: {str(e)}")
            return sources

    async def _calculate_domain_authority(self, source: ResearchSource) -> float:
        """Calculate domain authority score."""
        if not source.url or not self.enable_domain_authority:
            return self._get_default_authority_by_type(source.source_type)

        try:
            domain = urlparse(source.url).netloc.lower()

            # Check cache first
            cache_key = f"authority_{domain}"
            if cache_key in self._domain_cache:
                cache_entry = self._domain_cache[cache_key]
                if self._is_cache_valid(cache_entry["timestamp"]):
                    return cache_entry["score"]

            # Check against known high-credibility domains
            for trusted_domain in self.high_credibility_domains:
                if domain.endswith(trusted_domain) or domain == trusted_domain:
                    score = 0.9
                    self._cache_domain_score(cache_key, score)
                    return score

            # Check for government/educational domains
            if domain.endswith(".gov") or domain.endswith(".edu"):
                score = 0.85
                self._cache_domain_score(cache_key, score)
                return score

            # Check for academic domains
            if any(
                academic in domain
                for academic in ["university", "college", "institute"]
            ):
                score = 0.8
                self._cache_domain_score(cache_key, score)
                return score

            # Check against low-credibility patterns
            for pattern in self.low_credibility_patterns:
                if re.match(pattern, domain):
                    score = 0.3
                    self._cache_domain_score(cache_key, score)
                    return score

            # Default score for unknown domains
            score = 0.5
            self._cache_domain_score(cache_key, score)
            return score

        except Exception as e:
            logger.warning(f"Failed to calculate domain authority: {str(e)}")
            return self._get_default_authority_by_type(source.source_type)

    async def _calculate_reputation_score(self, source: ResearchSource) -> float:
        """Calculate source reputation score."""
        try:
            # Use source type as basis for reputation
            base_score = self._get_default_reputation_by_type(source.source_type)

            # Adjust based on metadata if available
            if source.metadata:
                # Check for quality indicators
                if source.metadata.get("peer_reviewed"):
                    base_score += 0.2
                if source.metadata.get("citations_count", 0) > 100:
                    base_score += 0.1
                if source.metadata.get("author_credentials"):
                    base_score += 0.1

                # Check for quality detractors
                if source.metadata.get("user_generated"):
                    base_score -= 0.2
                if source.metadata.get("sponsored_content"):
                    base_score -= 0.1

            return min(max(base_score, 0.0), 1.0)

        except Exception as e:
            logger.warning(f"Failed to calculate reputation score: {str(e)}")
            return 0.5

    async def _calculate_bias_score(self, source: ResearchSource) -> float:
        """Calculate bias score (0 = unbiased, 1 = highly biased)."""
        try:
            bias_score = 0.5  # Default neutral bias

            # Check source name and URL for bias indicators
            text_to_check = f"{source.name} {source.url or ''}"

            high_bias_count = sum(
                1
                for indicator in self.bias_indicators["high_bias"]
                if indicator.lower() in text_to_check.lower()
            )

            moderate_bias_count = sum(
                1
                for indicator in self.bias_indicators["moderate_bias"]
                if indicator.lower() in text_to_check.lower()
            )

            # Adjust bias score based on indicators
            bias_score += high_bias_count * 0.2
            bias_score += moderate_bias_count * 0.1

            # Adjust based on source type
            type_bias_adjustment = {
                SourceType.ACADEMIC: -0.2,
                SourceType.GOVERNMENT: -0.1,
                SourceType.TECHNICAL: -0.1,
                SourceType.NEWS: 0.0,
                SourceType.COMMERCIAL: 0.1,
                SourceType.SOCIAL: 0.3,
                SourceType.WEB: 0.1,
            }

            bias_score += type_bias_adjustment.get(source.source_type, 0.0)

            return min(max(bias_score, 0.0), 1.0)

        except Exception as e:
            logger.warning(f"Failed to calculate bias score: {str(e)}")
            return 0.5

    def _combine_credibility_scores(
        self,
        authority_score: float,
        reputation_score: float,
        bias_score: float,
        source_type: SourceType,
    ) -> float:
        """Combine individual scores into overall credibility score."""
        # Invert bias score (lower bias = higher credibility)
        unbiased_score = 1.0 - bias_score

        # Weight the scores based on source type
        weights = self._get_score_weights(source_type)

        credibility_score = (
            authority_score * weights["authority"]
            + reputation_score * weights["reputation"]
            + unbiased_score * weights["bias"]
        )

        return min(max(credibility_score, 0.0), 1.0)

    def _get_score_weights(self, source_type: SourceType) -> Dict[str, float]:
        """Get scoring weights based on source type."""
        weight_map = {
            SourceType.ACADEMIC: {"authority": 0.4, "reputation": 0.4, "bias": 0.2},
            SourceType.GOVERNMENT: {"authority": 0.5, "reputation": 0.3, "bias": 0.2},
            SourceType.TECHNICAL: {"authority": 0.4, "reputation": 0.4, "bias": 0.2},
            SourceType.NEWS: {"authority": 0.3, "reputation": 0.4, "bias": 0.3},
            SourceType.COMMERCIAL: {"authority": 0.3, "reputation": 0.3, "bias": 0.4},
            SourceType.SOCIAL: {"authority": 0.2, "reputation": 0.3, "bias": 0.5},
            SourceType.WEB: {"authority": 0.3, "reputation": 0.3, "bias": 0.4},
        }

        return weight_map.get(
            source_type, {"authority": 0.33, "reputation": 0.33, "bias": 0.34}
        )

    def _determine_credibility_level(
        self, credibility_score: float
    ) -> CredibilityLevel:
        """Determine credibility level from score."""
        if credibility_score >= 0.9:
            return CredibilityLevel.VERY_HIGH
        elif credibility_score >= 0.7:
            return CredibilityLevel.HIGH
        elif credibility_score >= 0.5:
            return CredibilityLevel.MEDIUM
        elif credibility_score >= 0.3:
            return CredibilityLevel.LOW
        else:
            return CredibilityLevel.VERY_LOW

    def _calculate_temporal_score(self, days_old: int) -> float:
        """Calculate freshness score based on age in days."""
        if days_old <= 1:
            return 1.0
        elif days_old <= 7:
            return 0.9
        elif days_old <= 30:
            return 0.8
        elif days_old <= 90:
            return 0.6
        elif days_old <= 365:
            return 0.4
        elif days_old <= 730:
            return 0.2
        else:
            return 0.1

    async def _extract_date_from_url(self, url: str) -> Optional[float]:
        """Extract date information from URL patterns."""
        try:
            # Common date patterns in URLs
            date_patterns = [
                r"/(\d{4})/(\d{1,2})/(\d{1,2})/",  # /2024/01/15/
                r"/(\d{4})-(\d{1,2})-(\d{1,2})/",  # /2024-01-15/
                r"(\d{4})(\d{2})(\d{2})",  # 20240115
            ]

            for pattern in date_patterns:
                match = re.search(pattern, url)
                if match:
                    try:
                        year, month, day = map(int, match.groups())
                        date = datetime(year, month, day)
                        days_old = (datetime.utcnow() - date).days
                        return self._calculate_temporal_score(days_old)
                    except ValueError:
                        continue

            return None

        except Exception as e:
            logger.debug(f"Failed to extract date from URL: {str(e)}")
            return None

    def _analyze_content_freshness(self, content: str) -> float:
        """Analyze content for temporal freshness indicators."""
        try:
            content_lower = content.lower()

            # Recent time indicators
            recent_indicators = [
                "today",
                "yesterday",
                "this week",
                "this month",
                "recently",
                "latest",
                "current",
                "now",
                "just released",
                "breaking",
            ]

            # Old time indicators
            old_indicators = [
                "last year",
                "years ago",
                "decades ago",
                "historically",
                "in the past",
                "previously",
                "former",
                "old",
                "vintage",
            ]

            recent_count = sum(
                1 for indicator in recent_indicators if indicator in content_lower
            )
            old_count = sum(
                1 for indicator in old_indicators if indicator in content_lower
            )

            # Calculate freshness based on temporal indicators
            if recent_count > old_count:
                return min(0.8 + (recent_count * 0.05), 1.0)
            elif old_count > recent_count:
                return max(0.3 - (old_count * 0.05), 0.1)
            else:
                return 0.5

        except Exception as e:
            logger.debug(f"Failed to analyze content freshness: {str(e)}")
            return 0.5

    def _calculate_composite_score(self, source: ResearchSource, query: str) -> float:
        """Calculate composite score for ranking."""
        # Base score from credibility and freshness
        base_score = (source.credibility_score * 0.6) + (source.freshness_score * 0.4)

        # Adjust for query relevance if domain matches
        if source.domain and query:
            query_words = set(query.lower().split())
            domain_words = set(source.domain.lower().split())
            relevance = (
                len(query_words.intersection(domain_words)) / len(query_words)
                if query_words
                else 0
            )
            base_score += relevance * 0.1

        return min(base_score, 1.0)

    def _get_default_authority_by_type(self, source_type: SourceType) -> float:
        """Get default authority score by source type."""
        authority_map = {
            SourceType.ACADEMIC: 0.9,
            SourceType.GOVERNMENT: 0.8,
            SourceType.TECHNICAL: 0.7,
            SourceType.NEWS: 0.6,
            SourceType.COMMERCIAL: 0.4,
            SourceType.WEB: 0.3,
            SourceType.SOCIAL: 0.2,
        }
        return authority_map.get(source_type, 0.5)

    def _get_default_reputation_by_type(self, source_type: SourceType) -> float:
        """Get default reputation score by source type."""
        reputation_map = {
            SourceType.ACADEMIC: 0.8,
            SourceType.GOVERNMENT: 0.7,
            SourceType.TECHNICAL: 0.6,
            SourceType.NEWS: 0.6,
            SourceType.COMMERCIAL: 0.4,
            SourceType.WEB: 0.3,
            SourceType.SOCIAL: 0.2,
        }
        return reputation_map.get(source_type, 0.5)

    def _cache_domain_score(self, cache_key: str, score: float) -> None:
        """Cache domain authority score."""
        self._domain_cache[cache_key] = {
            "score": score,
            "timestamp": datetime.utcnow(),
        }

    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cache entry is still valid."""
        return (datetime.utcnow() - timestamp).total_seconds() < (
            self.cache_ttl_hours * 3600
        )
