"""
Data models for the writing system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class WritingTone(Enum):
    """Available writing tones."""

    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    PERSUASIVE = "persuasive"
    INFORMATIVE = "informative"
    CONVERSATIONAL = "conversational"


class TechnicalLevel(Enum):
    """Technical complexity levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContentLength(Enum):
    """Content length specifications."""

    SHORT = "short"  # < 500 words
    MEDIUM = "medium"  # 500-2000 words
    LONG = "long"  # 2000-5000 words
    EXTENDED = "extended"  # > 5000 words


class ContentFormat(Enum):
    """Available content formats."""

    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    STRUCTURED_TEXT = "structured_text"
    ACADEMIC_PAPER = "academic_paper"
    TECHNICAL_REPORT = "technical_report"
    BLOG_POST = "blog_post"
    DOCUMENTATION = "documentation"
    PRESENTATION = "presentation"


class CitationStyle(Enum):
    """Citation styles supported."""

    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"


@dataclass
class WritingStyle:
    """Defines the writing style parameters."""

    tone: WritingTone
    formality_level: int  # 1-10 scale
    vocabulary_complexity: int  # 1-10 scale
    sentence_structure: str  # "simple", "complex", "varied"
    perspective: str  # "first_person", "second_person", "third_person"
    active_voice_preference: float  # 0.0-1.0

    @classmethod
    def professional(cls) -> "WritingStyle":
        """Create a professional writing style."""
        return cls(
            tone=WritingTone.PROFESSIONAL,
            formality_level=8,
            vocabulary_complexity=7,
            sentence_structure="varied",
            perspective="third_person",
            active_voice_preference=0.8,
        )

    @classmethod
    def academic(cls) -> "WritingStyle":
        """Create an academic writing style."""
        return cls(
            tone=WritingTone.ACADEMIC,
            formality_level=9,
            vocabulary_complexity=8,
            sentence_structure="complex",
            perspective="third_person",
            active_voice_preference=0.6,
        )

    @classmethod
    def casual(cls) -> "WritingStyle":
        """Create a casual writing style."""
        return cls(
            tone=WritingTone.CASUAL,
            formality_level=4,
            vocabulary_complexity=5,
            sentence_structure="simple",
            perspective="second_person",
            active_voice_preference=0.9,
        )


@dataclass
class StyleGuide:
    """Style guide specifications."""

    name: str
    rules: Dict[str, Any]
    preferred_terms: Dict[str, str]
    avoided_terms: List[str]
    formatting_rules: Dict[str, str]


@dataclass
class WritingRequirements:
    """Comprehensive writing requirements."""

    target_audience: str
    tone: WritingTone
    length: ContentLength
    format: ContentFormat
    style_guide: Optional[StyleGuide]
    citations_required: bool
    citation_style: Optional[CitationStyle]
    technical_level: TechnicalLevel
    keywords: List[str]
    outline_required: bool
    executive_summary: bool

    # Content structure requirements
    include_introduction: bool = True
    include_conclusion: bool = True
    max_heading_levels: int = 4

    # Quality requirements
    min_readability_score: Optional[float] = None
    max_passive_voice_percentage: Optional[float] = None

    # SEO requirements
    meta_description: Optional[str] = None
    target_keywords: List[str] = None


@dataclass
class ContentMetadata:
    """Metadata for generated content."""

    title: str
    author: Optional[str]
    created_at: datetime
    word_count: int
    character_count: int
    reading_time_minutes: int
    readability_score: float
    technical_level: TechnicalLevel
    keywords: List[str]
    outline: List[str]


@dataclass
class GeneratedContent:
    """Container for generated content with metadata."""

    content: str
    metadata: ContentMetadata
    citations: List["Citation"]
    quality_score: float
    readability_score: float
    structure_analysis: "StructureAnalysis"
    suggestions: List[str]


@dataclass
class StructureAnalysis:
    """Analysis of content structure."""

    heading_hierarchy: List[Dict[str, Any]]
    paragraph_count: int
    sentence_count: int
    avg_sentence_length: float
    avg_paragraph_length: float
    transition_quality: float
    coherence_score: float
    logical_flow_score: float


@dataclass
class Citation:
    """Citation information."""

    id: str
    source_type: str  # "book", "article", "website", "journal", etc.
    title: str
    authors: List[str]
    publication_date: Optional[datetime]
    url: Optional[str]
    doi: Optional[str]
    isbn: Optional[str]
    publisher: Optional[str]
    journal: Optional[str]
    volume: Optional[str]
    issue: Optional[str]
    pages: Optional[str]
    access_date: Optional[datetime]

    def format_apa(self) -> str:
        """Format citation in APA style."""
        # Basic APA formatting implementation
        authors_str = ", ".join(self.authors) if self.authors else "Unknown Author"
        year = f"({self.publication_date.year})" if self.publication_date else "(n.d.)"

        if self.source_type == "website" and self.url:
            return f"{authors_str} {year}. {self.title}. Retrieved from {self.url}"
        elif self.source_type == "journal" and self.journal:
            return f"{authors_str} {year}. {self.title}. {self.journal}, {self.volume}({self.issue}), {self.pages}."
        else:
            return f"{authors_str} {year}. {self.title}."

    def format_mla(self) -> str:
        """Format citation in MLA style."""
        # Basic MLA formatting implementation
        if not self.authors:
            return f'"{self.title}." Web.'

        author = self.authors[0]
        if self.source_type == "website" and self.url:
            return f'{author}. "{self.title}." Web. {self.access_date.strftime("%d %b %Y") if self.access_date else ""}.'
        else:
            return f'{author}. "{self.title}."'


@dataclass
class EditingSuggestion:
    """Suggestion for content improvement."""

    type: str  # "grammar", "style", "structure", "clarity"
    severity: str  # "low", "medium", "high"
    location: Dict[str, int]  # line, column, length
    original_text: str
    suggested_text: str
    explanation: str
    confidence: float


@dataclass
class ReadabilityMetrics:
    """Readability analysis metrics."""

    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog_index: float
    coleman_liau_index: float
    automated_readability_index: float
    avg_sentence_length: float
    avg_syllables_per_word: float
    complex_words_percentage: float
