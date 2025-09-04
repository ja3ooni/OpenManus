"""Data models for the research system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class ResearchDepth(str, Enum):
    """Research depth levels."""

    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"


class SourceType(str, Enum):
    """Types of research sources."""

    WEB = "web"
    ACADEMIC = "academic"
    NEWS = "news"
    TECHNICAL = "technical"
    SOCIAL = "social"
    GOVERNMENT = "government"
    COMMERCIAL = "commercial"


class CredibilityLevel(str, Enum):
    """Source credibility levels."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"
    UNKNOWN = "unknown"


class ConflictType(str, Enum):
    """Types of information conflicts."""

    DIRECT_CONTRADICTION = "direct_contradiction"
    PARTIAL_DISAGREEMENT = "partial_disagreement"
    DIFFERENT_PERSPECTIVE = "different_perspective"
    TEMPORAL_DIFFERENCE = "temporal_difference"
    METHODOLOGICAL_DIFFERENCE = "methodological_difference"


class ResearchSource(BaseModel):
    """Represents a source of information for research."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description="Name or identifier of the source")
    url: Optional[str] = Field(default=None, description="URL of the source")
    source_type: SourceType = Field(description="Type of source")
    domain: Optional[str] = Field(default=None, description="Domain or subject area")
    credibility_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Credibility score (0-1)"
    )
    credibility_level: CredibilityLevel = Field(default=CredibilityLevel.UNKNOWN)
    freshness_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Freshness score (0-1)"
    )
    authority_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Domain authority score (0-1)"
    )
    bias_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Bias score (0=unbiased, 1=highly biased)",
    )
    language: str = Field(default="en", description="Language of the source")
    country: Optional[str] = Field(default=None, description="Country of origin")
    last_updated: Optional[datetime] = Field(
        default=None, description="When the source was last updated"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ResearchFinding(BaseModel):
    """Represents a finding from research."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(description="The actual finding content")
    source: ResearchSource = Field(description="Source of this finding")
    query: str = Field(description="Original query that led to this finding")
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Relevance to the query (0-1)"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence in the finding (0-1)"
    )
    key_points: List[str] = Field(
        default_factory=list, description="Key points extracted from the finding"
    )
    supporting_evidence: List[str] = Field(
        default_factory=list, description="Supporting evidence"
    )
    contradicting_evidence: List[str] = Field(
        default_factory=list, description="Contradicting evidence"
    )
    context: Optional[str] = Field(default=None, description="Additional context")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the finding was discovered"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class CrossReference(BaseModel):
    """Represents a cross-reference between findings."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    finding_ids: List[str] = Field(description="IDs of related findings")
    relationship_type: str = Field(
        description="Type of relationship (supports, contradicts, expands, etc.)"
    )
    strength: float = Field(
        ge=0.0, le=1.0, description="Strength of the relationship (0-1)"
    )
    description: Optional[str] = Field(
        default=None, description="Description of the relationship"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the relationship (0-1)"
    )


class InformationConflict(BaseModel):
    """Represents a conflict between different pieces of information."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    conflicting_findings: List[str] = Field(description="IDs of conflicting findings")
    conflict_type: ConflictType = Field(description="Type of conflict")
    severity: float = Field(
        ge=0.0, le=1.0, description="Severity of the conflict (0-1)"
    )
    description: str = Field(description="Description of the conflict")
    resolution_suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for resolving the conflict"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ResearchResult(BaseModel):
    """Comprehensive result of a research operation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str = Field(description="Original research query")
    findings: List[ResearchFinding] = Field(
        default_factory=list, description="All findings from the research"
    )
    cross_references: List[CrossReference] = Field(
        default_factory=list, description="Cross-references between findings"
    )
    conflicts: List[InformationConflict] = Field(
        default_factory=list, description="Identified conflicts"
    )
    summary: Optional[str] = Field(default=None, description="Summary of the research")
    key_insights: List[str] = Field(
        default_factory=list, description="Key insights from the research"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in the research results"
    )
    completeness_score: float = Field(
        ge=0.0, le=1.0, description="How complete the research is"
    )
    sources_used: List[ResearchSource] = Field(
        default_factory=list, description="All sources used in the research"
    )
    research_depth: ResearchDepth = Field(description="Depth of research performed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the research was completed"
    )
    duration_seconds: Optional[float] = Field(
        default=None, description="How long the research took"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def get_high_confidence_findings(
        self, threshold: float = 0.7
    ) -> List[ResearchFinding]:
        """Get findings with confidence above threshold."""
        return [f for f in self.findings if f.confidence_score >= threshold]

    def get_findings_by_source_type(
        self, source_type: SourceType
    ) -> List[ResearchFinding]:
        """Get findings from a specific source type."""
        return [f for f in self.findings if f.source.source_type == source_type]

    def get_conflicting_findings(self) -> List[ResearchFinding]:
        """Get findings that are involved in conflicts."""
        conflicting_ids = set()
        for conflict in self.conflicts:
            conflicting_ids.update(conflict.conflicting_findings)
        return [f for f in self.findings if f.id in conflicting_ids]


class ResearchQuery(BaseModel):
    """Represents a research query with parameters."""

    query: str = Field(description="The research query")
    depth: ResearchDepth = Field(
        default=ResearchDepth.STANDARD, description="Depth of research"
    )
    max_sources: int = Field(default=10, description="Maximum number of sources to use")
    max_findings: int = Field(
        default=50, description="Maximum number of findings to collect"
    )
    source_types: List[SourceType] = Field(
        default_factory=list, description="Preferred source types"
    )
    languages: List[str] = Field(
        default_factory=lambda: ["en"], description="Preferred languages"
    )
    time_range_days: Optional[int] = Field(
        default=None, description="Time range in days for recent information"
    )
    domain_focus: Optional[str] = Field(
        default=None, description="Specific domain to focus on"
    )
    enable_cross_referencing: bool = Field(
        default=True, description="Whether to perform cross-referencing"
    )
    enable_conflict_detection: bool = Field(
        default=True, description="Whether to detect conflicts"
    )
    min_credibility_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum credibility threshold"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional query parameters"
    )
