"""
Core writing engine for advanced content generation.
"""

import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..llm import LLMClient
from ..logger import get_logger
from .models import (
    ContentFormat,
    ContentMetadata,
    EditingSuggestion,
    GeneratedContent,
    ReadabilityMetrics,
    StructureAnalysis,
    TechnicalLevel,
    WritingRequirements,
    WritingStyle,
    WritingTone,
)

logger = get_logger(__name__)


class WritingEngine:
    """
    Advanced content generation engine with style management and format control.

    Provides style-aware content generation with tone and audience adaptation,
    content structure analysis and optimization.
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize the writing engine."""
        self.llm_client = llm_client
        self.style_templates = self._load_style_templates()
        self.format_templates = self._load_format_templates()

    async def generate_content(
        self,
        prompt: str,
        style: WritingStyle,
        format: ContentFormat,
        requirements: WritingRequirements,
    ) -> GeneratedContent:
        """
        Generate content based on style and format requirements.

        Args:
            prompt: The content generation prompt
            style: Writing style parameters
            format: Target content format
            requirements: Comprehensive writing requirements

        Returns:
            Generated content with metadata and analysis
        """
        logger.info(
            f"Generating content with style: {style.tone.value}, format: {format.value}"
        )

        try:
            # Build comprehensive generation prompt
            generation_prompt = await self._build_generation_prompt(
                prompt, style, format, requirements
            )

            # Generate content using LLM
            raw_content = await self._generate_raw_content(generation_prompt)

            # Post-process and structure content
            structured_content = await self._structure_content(
                raw_content, format, requirements
            )

            # Analyze content structure and quality
            structure_analysis = await self._analyze_structure(structured_content)
            readability_metrics = await self._calculate_readability(structured_content)

            # Create metadata
            metadata = self._create_metadata(
                structured_content, requirements, readability_metrics
            )

            # Calculate quality score
            quality_score = await self._calculate_quality_score(
                structured_content, style, requirements, structure_analysis
            )

            # Generate improvement suggestions
            suggestions = await self._generate_suggestions(
                structured_content, style, requirements, structure_analysis
            )

            return GeneratedContent(
                content=structured_content,
                metadata=metadata,
                citations=[],  # Will be populated by citation manager
                quality_score=quality_score,
                readability_score=readability_metrics.flesch_reading_ease,
                structure_analysis=structure_analysis,
                suggestions=suggestions,
            )

        except Exception as e:
            logger.error(f"Content generation failed: {str(e)}")
            raise

    async def edit_content(
        self,
        content: str,
        edit_type: str,
        style: WritingStyle,
        requirements: Optional[WritingRequirements] = None,
    ) -> GeneratedContent:
        """
        Edit and improve existing content.

        Args:
            content: Original content to edit
            edit_type: Type of editing ("grammar", "style", "structure", "clarity")
            style: Target writing style
            requirements: Optional writing requirements

        Returns:
            Edited content with improvements
        """
        logger.info(f"Editing content with type: {edit_type}")

        try:
            # Build editing prompt
            edit_prompt = await self._build_edit_prompt(
                content, edit_type, style, requirements
            )

            # Generate edited content
            edited_content = await self._generate_raw_content(edit_prompt)

            # Analyze improvements
            structure_analysis = await self._analyze_structure(edited_content)
            readability_metrics = await self._calculate_readability(edited_content)

            # Create metadata
            metadata = self._create_metadata(
                edited_content,
                requirements
                or WritingRequirements(
                    target_audience="general",
                    tone=style.tone,
                    length=ContentFormat.MEDIUM,
                    format=ContentFormat.MARKDOWN,
                    citations_required=False,
                    technical_level=TechnicalLevel.INTERMEDIATE,
                    keywords=[],
                    outline_required=False,
                    executive_summary=False,
                ),
                readability_metrics,
            )

            # Calculate quality score
            quality_score = await self._calculate_quality_score(
                edited_content, style, requirements, structure_analysis
            )

            return GeneratedContent(
                content=edited_content,
                metadata=metadata,
                citations=[],
                quality_score=quality_score,
                readability_score=readability_metrics.flesch_reading_ease,
                structure_analysis=structure_analysis,
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Content editing failed: {str(e)}")
            raise

    async def analyze_content_structure(self, content: str) -> StructureAnalysis:
        """
        Analyze the structure of existing content.

        Args:
            content: Content to analyze

        Returns:
            Detailed structure analysis
        """
        return await self._analyze_structure(content)

    async def optimize_content_structure(
        self, content: str, requirements: WritingRequirements
    ) -> GeneratedContent:
        """
        Optimize content structure based on requirements.

        Args:
            content: Original content
            requirements: Structure requirements

        Returns:
            Optimized content
        """
        logger.info("Optimizing content structure")

        try:
            # Analyze current structure
            current_analysis = await self._analyze_structure(content)

            # Build optimization prompt
            optimization_prompt = await self._build_optimization_prompt(
                content, current_analysis, requirements
            )

            # Generate optimized content
            optimized_content = await self._generate_raw_content(optimization_prompt)

            # Analyze optimized structure
            new_analysis = await self._analyze_structure(optimized_content)
            readability_metrics = await self._calculate_readability(optimized_content)

            # Create metadata
            metadata = self._create_metadata(
                optimized_content, requirements, readability_metrics
            )

            # Calculate quality score
            quality_score = await self._calculate_quality_score(
                optimized_content,
                WritingStyle.professional(),
                requirements,
                new_analysis,
            )

            return GeneratedContent(
                content=optimized_content,
                metadata=metadata,
                citations=[],
                quality_score=quality_score,
                readability_score=readability_metrics.flesch_reading_ease,
                structure_analysis=new_analysis,
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Content optimization failed: {str(e)}")
            raise

    def _load_style_templates(self) -> Dict[str, str]:
        """Load writing style templates."""
        return {
            "professional": """
            Write in a professional, business-appropriate tone. Use:
            - Clear, concise language
            - Active voice when possible
            - Formal vocabulary
            - Third-person perspective
            - Well-structured paragraphs
            """,
            "academic": """
            Write in an academic style suitable for scholarly work. Use:
            - Formal, precise language
            - Complex sentence structures
            - Technical vocabulary when appropriate
            - Third-person perspective
            - Evidence-based arguments
            """,
            "casual": """
            Write in a casual, conversational tone. Use:
            - Simple, everyday language
            - Contractions and informal expressions
            - Second-person perspective
            - Short, readable sentences
            - Engaging, friendly tone
            """,
            "technical": """
            Write in a technical style for expert audiences. Use:
            - Precise technical terminology
            - Detailed explanations
            - Step-by-step instructions
            - Code examples when relevant
            - Clear structure with headings
            """,
        }

    def _load_format_templates(self) -> Dict[str, str]:
        """Load content format templates."""
        return {
            "markdown": """
            Format the content using Markdown syntax:
            - Use # for main headings, ## for subheadings
            - Use **bold** and *italic* for emphasis
            - Use bullet points and numbered lists
            - Include code blocks with ```
            - Use links [text](url) format
            """,
            "academic_paper": """
            Structure as an academic paper:
            - Abstract (if required)
            - Introduction with thesis
            - Literature review (if applicable)
            - Methodology/Analysis
            - Results/Discussion
            - Conclusion
            - References
            """,
            "technical_report": """
            Structure as a technical report:
            - Executive Summary
            - Introduction and Objectives
            - Technical Details
            - Implementation
            - Results and Analysis
            - Recommendations
            - Appendices (if needed)
            """,
            "blog_post": """
            Structure as an engaging blog post:
            - Compelling headline
            - Hook in introduction
            - Clear subheadings
            - Short paragraphs
            - Call-to-action conclusion
            - SEO-friendly structure
            """,
        }

    async def _build_generation_prompt(
        self,
        prompt: str,
        style: WritingStyle,
        format: ContentFormat,
        requirements: WritingRequirements,
    ) -> str:
        """Build comprehensive generation prompt."""
        style_template = self.style_templates.get(style.tone.value, "")
        format_template = self.format_templates.get(format.value, "")

        generation_prompt = f"""
        Generate high-quality content based on the following requirements:

        CONTENT REQUEST: {prompt}

        WRITING STYLE:
        {style_template}
        - Formality level: {style.formality_level}/10
        - Vocabulary complexity: {style.vocabulary_complexity}/10
        - Sentence structure: {style.sentence_structure}
        - Perspective: {style.perspective}
        - Active voice preference: {style.active_voice_preference * 100}%

        FORMAT REQUIREMENTS:
        {format_template}

        CONTENT REQUIREMENTS:
        - Target audience: {requirements.target_audience}
        - Technical level: {requirements.technical_level.value}
        - Length: {requirements.length.value}
        - Keywords to include: {', '.join(requirements.keywords)}
        - Include introduction: {requirements.include_introduction}
        - Include conclusion: {requirements.include_conclusion}
        - Maximum heading levels: {requirements.max_heading_levels}
        """

        if requirements.outline_required:
            generation_prompt += "\n- Provide a clear outline structure"

        if requirements.executive_summary:
            generation_prompt += "\n- Include an executive summary"

        if requirements.citations_required:
            generation_prompt += f"\n- Include citations in {requirements.citation_style.value if requirements.citation_style else 'APA'} format"

        generation_prompt += "\n\nGenerate comprehensive, well-structured content that meets all requirements."

        return generation_prompt

    async def _build_edit_prompt(
        self,
        content: str,
        edit_type: str,
        style: WritingStyle,
        requirements: Optional[WritingRequirements],
    ) -> str:
        """Build editing prompt."""
        style_template = self.style_templates.get(style.tone.value, "")

        edit_prompt = f"""
        Edit and improve the following content focusing on {edit_type}:

        ORIGINAL CONTENT:
        {content}

        EDITING FOCUS: {edit_type}

        TARGET STYLE:
        {style_template}

        IMPROVEMENTS TO MAKE:
        """

        if edit_type == "grammar":
            edit_prompt += """
            - Fix grammatical errors
            - Improve sentence structure
            - Correct punctuation and spelling
            - Ensure proper verb tenses
            """
        elif edit_type == "style":
            edit_prompt += f"""
            - Adjust tone to {style.tone.value}
            - Improve vocabulary appropriateness
            - Enhance readability
            - Ensure consistent voice and perspective
            """
        elif edit_type == "structure":
            edit_prompt += """
            - Improve logical flow
            - Enhance paragraph transitions
            - Optimize heading hierarchy
            - Strengthen introduction and conclusion
            """
        elif edit_type == "clarity":
            edit_prompt += """
            - Simplify complex sentences
            - Remove redundancy
            - Improve word choice
            - Enhance overall clarity
            """

        edit_prompt += "\n\nProvide the improved version of the content."

        return edit_prompt

    async def _build_optimization_prompt(
        self,
        content: str,
        analysis: StructureAnalysis,
        requirements: WritingRequirements,
    ) -> str:
        """Build structure optimization prompt."""
        return f"""
        Optimize the structure of the following content:

        ORIGINAL CONTENT:
        {content}

        CURRENT STRUCTURE ANALYSIS:
        - Heading hierarchy: {len(analysis.heading_hierarchy)} levels
        - Paragraph count: {analysis.paragraph_count}
        - Average sentence length: {analysis.avg_sentence_length:.1f} words
        - Coherence score: {analysis.coherence_score:.2f}
        - Logical flow score: {analysis.logical_flow_score:.2f}

        OPTIMIZATION REQUIREMENTS:
        - Target audience: {requirements.target_audience}
        - Maximum heading levels: {requirements.max_heading_levels}
        - Include introduction: {requirements.include_introduction}
        - Include conclusion: {requirements.include_conclusion}

        OPTIMIZATION GOALS:
        - Improve logical flow and coherence
        - Optimize heading hierarchy
        - Enhance paragraph structure
        - Strengthen transitions between sections
        - Ensure clear introduction and conclusion

        Provide the optimized version with improved structure.
        """

    async def _generate_raw_content(self, prompt: str) -> str:
        """Generate raw content using LLM."""
        try:
            response = await self.llm_client.generate_response(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"LLM content generation failed: {str(e)}")
            raise

    async def _structure_content(
        self, content: str, format: ContentFormat, requirements: WritingRequirements
    ) -> str:
        """Apply format-specific structuring to content."""
        # Basic structuring - can be enhanced with format-specific processing
        structured = content

        # Ensure proper heading hierarchy for markdown
        if format == ContentFormat.MARKDOWN:
            structured = self._fix_markdown_headings(
                structured, requirements.max_heading_levels
            )

        return structured

    def _fix_markdown_headings(self, content: str, max_levels: int) -> str:
        """Fix markdown heading hierarchy."""
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            if line.strip().startswith("#"):
                # Count heading level
                level = len(line) - len(line.lstrip("#"))
                if level > max_levels:
                    # Reduce to maximum level
                    line = "#" * max_levels + line[level:]
            fixed_lines.append(line)

        return "\n".join(fixed_lines)

    async def _analyze_structure(self, content: str) -> StructureAnalysis:
        """Analyze content structure."""
        lines = content.split("\n")
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Extract heading hierarchy
        heading_hierarchy = []
        for line in lines:
            if line.strip().startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("#").strip()
                heading_hierarchy.append(
                    {"level": level, "text": text, "line": lines.index(line)}
                )

        # Calculate metrics
        avg_sentence_length = (
            sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        )
        avg_paragraph_length = (
            sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            if paragraphs
            else 0
        )

        # Simple coherence and flow scoring (can be enhanced with NLP)
        coherence_score = min(1.0, len(heading_hierarchy) / max(1, len(paragraphs) / 3))
        logical_flow_score = 0.8  # Placeholder - would need more sophisticated analysis
        transition_quality = 0.7  # Placeholder - would analyze transition words/phrases

        return StructureAnalysis(
            heading_hierarchy=heading_hierarchy,
            paragraph_count=len(paragraphs),
            sentence_count=len(sentences),
            avg_sentence_length=avg_sentence_length,
            avg_paragraph_length=avg_paragraph_length,
            transition_quality=transition_quality,
            coherence_score=coherence_score,
            logical_flow_score=logical_flow_score,
        )

    async def _calculate_readability(self, content: str) -> ReadabilityMetrics:
        """Calculate readability metrics."""
        words = content.split()
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_count = len(words)
        sentence_count = len(sentences)

        if sentence_count == 0 or word_count == 0:
            return ReadabilityMetrics(
                flesch_reading_ease=0.0,
                flesch_kincaid_grade=0.0,
                gunning_fog_index=0.0,
                coleman_liau_index=0.0,
                automated_readability_index=0.0,
                avg_sentence_length=0.0,
                avg_syllables_per_word=0.0,
                complex_words_percentage=0.0,
            )

        avg_sentence_length = word_count / sentence_count

        # Simple syllable counting (can be enhanced with proper syllable counter)
        def count_syllables(word):
            word = word.lower()
            vowels = "aeiouy"
            syllable_count = 0
            prev_was_vowel = False

            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False

            if word.endswith("e"):
                syllable_count -= 1

            return max(1, syllable_count)

        total_syllables = sum(count_syllables(word) for word in words)
        avg_syllables_per_word = total_syllables / word_count

        # Complex words (3+ syllables)
        complex_words = sum(1 for word in words if count_syllables(word) >= 3)
        complex_words_percentage = (complex_words / word_count) * 100

        # Flesch Reading Ease
        flesch_reading_ease = (
            206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        )

        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (
            (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        )

        # Gunning Fog Index
        gunning_fog_index = 0.4 * (avg_sentence_length + complex_words_percentage)

        # Coleman-Liau Index (simplified)
        avg_letters_per_100_words = (
            sum(len(word) for word in words) / word_count
        ) * 100
        avg_sentences_per_100_words = (sentence_count / word_count) * 100
        coleman_liau_index = (
            (0.0588 * avg_letters_per_100_words)
            - (0.296 * avg_sentences_per_100_words)
            - 15.8
        )

        # Automated Readability Index
        avg_chars_per_word = sum(len(word) for word in words) / word_count
        automated_readability_index = (
            (4.71 * avg_chars_per_word) + (0.5 * avg_sentence_length) - 21.43
        )

        return ReadabilityMetrics(
            flesch_reading_ease=max(0, min(100, flesch_reading_ease)),
            flesch_kincaid_grade=max(0, flesch_kincaid_grade),
            gunning_fog_index=max(0, gunning_fog_index),
            coleman_liau_index=max(0, coleman_liau_index),
            automated_readability_index=max(0, automated_readability_index),
            avg_sentence_length=avg_sentence_length,
            avg_syllables_per_word=avg_syllables_per_word,
            complex_words_percentage=complex_words_percentage,
        )

    def _create_metadata(
        self,
        content: str,
        requirements: WritingRequirements,
        readability: ReadabilityMetrics,
    ) -> ContentMetadata:
        """Create content metadata."""
        words = content.split()
        word_count = len(words)
        character_count = len(content)

        # Estimate reading time (average 200 words per minute)
        reading_time_minutes = max(1, word_count // 200)

        # Extract outline from headings
        outline = []
        for line in content.split("\n"):
            if line.strip().startswith("#"):
                outline.append(line.strip())

        return ContentMetadata(
            title=outline[0].lstrip("#").strip() if outline else "Generated Content",
            author=None,
            created_at=datetime.now(),
            word_count=word_count,
            character_count=character_count,
            reading_time_minutes=reading_time_minutes,
            readability_score=readability.flesch_reading_ease,
            technical_level=requirements.technical_level,
            keywords=requirements.keywords,
            outline=outline,
        )

    async def _calculate_quality_score(
        self,
        content: str,
        style: WritingStyle,
        requirements: Optional[WritingRequirements],
        structure: StructureAnalysis,
    ) -> float:
        """Calculate overall content quality score."""
        scores = []

        # Structure quality (0-1)
        structure_score = (
            structure.coherence_score
            + structure.logical_flow_score
            + structure.transition_quality
        ) / 3
        scores.append(structure_score)

        # Length appropriateness (0-1)
        word_count = len(content.split())
        if requirements:
            target_ranges = {
                "short": (100, 500),
                "medium": (500, 2000),
                "long": (2000, 5000),
                "extended": (5000, 10000),
            }
            target_range = target_ranges.get(requirements.length.value, (500, 2000))
            if target_range[0] <= word_count <= target_range[1]:
                length_score = 1.0
            else:
                # Penalize for being too short or too long
                if word_count < target_range[0]:
                    length_score = word_count / target_range[0]
                else:
                    length_score = max(0.5, target_range[1] / word_count)
            scores.append(length_score)

        # Content completeness (0-1)
        completeness_score = 0.8  # Placeholder - would check for required sections
        scores.append(completeness_score)

        # Style consistency (0-1)
        style_score = 0.8  # Placeholder - would analyze style consistency
        scores.append(style_score)

        return sum(scores) / len(scores)

    async def _generate_suggestions(
        self,
        content: str,
        style: WritingStyle,
        requirements: WritingRequirements,
        structure: StructureAnalysis,
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        # Structure suggestions
        if structure.coherence_score < 0.7:
            suggestions.append("Consider improving the logical flow between sections")

        if structure.avg_sentence_length > 25:
            suggestions.append(
                "Consider breaking down long sentences for better readability"
            )

        if len(structure.heading_hierarchy) < 2 and structure.paragraph_count > 5:
            suggestions.append("Add more headings to improve content organization")

        # Style suggestions
        if style.active_voice_preference > 0.8:
            suggestions.append("Consider using more active voice constructions")

        # Requirements-based suggestions
        if requirements.citations_required and "citation" not in content.lower():
            suggestions.append("Add citations to support your claims")

        if requirements.executive_summary and "summary" not in content.lower():
            suggestions.append("Consider adding an executive summary")

        return suggestions
