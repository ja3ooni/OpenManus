"""
Content editing and improvement tools.
"""

import re
from typing import Dict, List, Optional, Tuple

from ..llm import LLMClient
from ..logger import get_logger
from .models import (
    EditingSuggestion,
    GeneratedContent,
    ReadabilityMetrics,
    StructureAnalysis,
    WritingRequirements,
    WritingStyle,
)

logger = get_logger(__name__)


class ContentEditor:
    """
    Advanced content editing and improvement system.

    Provides grammar checking, style improvement suggestions, readability analysis,
    content structure validation, and plagiarism detection capabilities.
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize the content editor."""
        self.llm_client = llm_client
        self.grammar_rules = self._load_grammar_rules()
        self.style_rules = self._load_style_rules()
        self.readability_thresholds = self._load_readability_thresholds()

    async def check_grammar(self, content: str) -> List[EditingSuggestion]:
        """
        Check content for grammatical errors and provide suggestions.

        Args:
            content: Content to check

        Returns:
            List of grammar suggestions
        """
        logger.info("Performing grammar check")

        suggestions = []

        # Basic grammar checks
        suggestions.extend(await self._check_basic_grammar(content))

        # Advanced grammar check using LLM
        suggestions.extend(await self._check_advanced_grammar(content))

        return suggestions

    async def improve_style(
        self,
        content: str,
        target_style: WritingStyle,
        requirements: Optional[WritingRequirements] = None,
    ) -> List[EditingSuggestion]:
        """
        Analyze content style and provide improvement suggestions.

        Args:
            content: Content to analyze
            target_style: Target writing style
            requirements: Optional writing requirements

        Returns:
            List of style improvement suggestions
        """
        logger.info(f"Analyzing style for {target_style.tone.value} tone")

        suggestions = []

        # Check tone consistency
        suggestions.extend(await self._check_tone_consistency(content, target_style))

        # Check vocabulary appropriateness
        suggestions.extend(await self._check_vocabulary(content, target_style))

        # Check sentence structure
        suggestions.extend(await self._check_sentence_structure(content, target_style))

        # Check voice (active/passive)
        suggestions.extend(await self._check_voice(content, target_style))

        return suggestions

    async def analyze_readability(self, content: str) -> ReadabilityMetrics:
        """
        Analyze content readability and provide metrics.

        Args:
            content: Content to analyze

        Returns:
            Comprehensive readability metrics
        """
        logger.info("Analyzing content readability")

        # Calculate basic metrics
        words = content.split()
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not words or not sentences:
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

        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / sentence_count

        # Calculate syllables and complex words
        total_syllables = 0
        complex_words = 0

        for word in words:
            syllables = self._count_syllables(word)
            total_syllables += syllables
            if syllables >= 3:
                complex_words += 1

        avg_syllables_per_word = total_syllables / word_count
        complex_words_percentage = (complex_words / word_count) * 100

        # Calculate readability scores
        flesch_reading_ease = (
            206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        )
        flesch_kincaid_grade = (
            (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        )
        gunning_fog_index = 0.4 * (avg_sentence_length + complex_words_percentage)

        # Coleman-Liau Index
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

    async def optimize_readability(
        self, content: str, target_grade_level: Optional[float] = None
    ) -> GeneratedContent:
        """
        Optimize content for better readability.

        Args:
            content: Content to optimize
            target_grade_level: Target reading grade level (optional)

        Returns:
            Optimized content with improved readability
        """
        logger.info("Optimizing content readability")

        current_metrics = await self.analyze_readability(content)
        target_level = target_grade_level or 8.0  # Default to 8th grade level

        optimization_prompt = f"""
        Improve the readability of the following content to approximately a {target_level} grade reading level:

        CURRENT METRICS:
        - Flesch-Kincaid Grade: {current_metrics.flesch_kincaid_grade:.1f}
        - Average sentence length: {current_metrics.avg_sentence_length:.1f} words
        - Complex words: {current_metrics.complex_words_percentage:.1f}%

        CONTENT TO OPTIMIZE:
        {content}

        OPTIMIZATION GUIDELINES:
        - Simplify complex sentences (aim for {target_level * 2:.0f} words per sentence)
        - Replace complex words with simpler alternatives
        - Break long paragraphs into shorter ones
        - Use active voice when possible
        - Maintain the original meaning and key information

        Provide the optimized version:
        """

        try:
            optimized_content = await self.llm_client.generate_response(
                messages=[{"role": "user", "content": optimization_prompt}],
                max_tokens=4000,
                temperature=0.3,
            )

            # Analyze optimized content
            new_metrics = await self.analyze_readability(optimized_content)

            # Create metadata (simplified version)
            from datetime import datetime

            from .models import ContentMetadata, TechnicalLevel

            metadata = ContentMetadata(
                title="Optimized Content",
                author=None,
                created_at=datetime.now(),
                word_count=len(optimized_content.split()),
                character_count=len(optimized_content),
                reading_time_minutes=max(1, len(optimized_content.split()) // 200),
                readability_score=new_metrics.flesch_reading_ease,
                technical_level=TechnicalLevel.INTERMEDIATE,
                keywords=[],
                outline=[],
            )

            return GeneratedContent(
                content=optimized_content,
                metadata=metadata,
                citations=[],
                quality_score=0.8,  # Placeholder
                readability_score=new_metrics.flesch_reading_ease,
                structure_analysis=StructureAnalysis(
                    heading_hierarchy=[],
                    paragraph_count=len(
                        [p for p in optimized_content.split("\n\n") if p.strip()]
                    ),
                    sentence_count=len(re.split(r"[.!?]+", optimized_content)),
                    avg_sentence_length=new_metrics.avg_sentence_length,
                    avg_paragraph_length=0.0,  # Placeholder
                    transition_quality=0.7,  # Placeholder
                    coherence_score=0.8,  # Placeholder
                    logical_flow_score=0.8,  # Placeholder
                ),
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Readability optimization failed: {str(e)}")
            raise

    async def validate_structure(
        self, content: str, requirements: WritingRequirements
    ) -> List[EditingSuggestion]:
        """
        Validate content structure against requirements.

        Args:
            content: Content to validate
            requirements: Structure requirements

        Returns:
            List of structure improvement suggestions
        """
        logger.info("Validating content structure")

        suggestions = []
        lines = content.split("\n")

        # Check for required sections
        if requirements.include_introduction:
            if not self._has_introduction(content):
                suggestions.append(
                    EditingSuggestion(
                        type="structure",
                        severity="high",
                        location={"line": 1, "column": 1, "length": 0},
                        original_text="",
                        suggested_text="Add an introduction section",
                        explanation="Content should include an introduction",
                        confidence=0.9,
                    )
                )

        if requirements.include_conclusion:
            if not self._has_conclusion(content):
                suggestions.append(
                    EditingSuggestion(
                        type="structure",
                        severity="high",
                        location={"line": len(lines), "column": 1, "length": 0},
                        original_text="",
                        suggested_text="Add a conclusion section",
                        explanation="Content should include a conclusion",
                        confidence=0.9,
                    )
                )

        # Check heading hierarchy
        headings = self._extract_headings(content)
        if len(headings) > requirements.max_heading_levels:
            suggestions.append(
                EditingSuggestion(
                    type="structure",
                    severity="medium",
                    location={"line": 0, "column": 0, "length": 0},
                    original_text="",
                    suggested_text=f"Reduce heading levels to {requirements.max_heading_levels}",
                    explanation=f"Content has {len(headings)} heading levels, maximum allowed is {requirements.max_heading_levels}",
                    confidence=0.8,
                )
            )

        # Check for keywords
        if requirements.keywords:
            missing_keywords = self._check_keywords(content, requirements.keywords)
            if missing_keywords:
                suggestions.append(
                    EditingSuggestion(
                        type="structure",
                        severity="medium",
                        location={"line": 0, "column": 0, "length": 0},
                        original_text="",
                        suggested_text=f"Include keywords: {', '.join(missing_keywords)}",
                        explanation="Content should include specified keywords",
                        confidence=0.7,
                    )
                )

        return suggestions

    async def check_originality(self, content: str) -> Dict[str, any]:
        """
        Basic plagiarism detection and originality checking.

        Args:
            content: Content to check

        Returns:
            Originality analysis results
        """
        logger.info("Checking content originality")

        # This is a simplified implementation
        # In production, you would integrate with plagiarism detection APIs

        results = {
            "originality_score": 0.85,  # Placeholder
            "potential_matches": [],
            "suspicious_phrases": [],
            "recommendations": [],
        }

        # Check for common phrases that might indicate copying
        suspicious_patterns = [
            r"according to (the )?research",
            r"studies have shown",
            r"it is widely known",
            r"as mentioned (earlier|before|above)",
        ]

        for pattern in suspicious_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                results["suspicious_phrases"].append(
                    {
                        "phrase": match.group(),
                        "position": match.start(),
                        "reason": "Common academic phrase - ensure proper attribution",
                    }
                )

        # Provide recommendations
        if len(results["suspicious_phrases"]) > 5:
            results["recommendations"].append(
                "Consider paraphrasing common phrases to improve originality"
            )

        if results["originality_score"] < 0.8:
            results["recommendations"].append(
                "Content may need significant revision for originality"
            )

        return results

    def _load_grammar_rules(self) -> Dict[str, str]:
        """Load basic grammar rules."""
        return {
            "double_space": r"  +",
            "comma_space": r",(?!\s)",
            "period_space": r"\.(?!\s|$)",
            "apostrophe": r"(\w)'(\w)",
            "its_vs_its": r"\bits\s",
        }

    def _load_style_rules(self) -> Dict[str, Dict]:
        """Load style checking rules."""
        return {
            "passive_voice": {
                "pattern": r"\b(was|were|is|are|been|being)\s+\w+ed\b",
                "message": "Consider using active voice",
            },
            "weak_words": {
                "words": ["very", "really", "quite", "rather", "somewhat"],
                "message": "Consider using stronger, more specific words",
            },
            "redundant_phrases": {
                "phrases": [
                    "in order to",
                    "due to the fact that",
                    "at this point in time",
                ],
                "message": "Consider using more concise alternatives",
            },
        }

    def _load_readability_thresholds(self) -> Dict[str, float]:
        """Load readability score thresholds."""
        return {
            "flesch_reading_ease": {
                "very_easy": 90,
                "easy": 80,
                "fairly_easy": 70,
                "standard": 60,
                "fairly_difficult": 50,
                "difficult": 30,
                "very_difficult": 0,
            }
        }

    async def _check_basic_grammar(self, content: str) -> List[EditingSuggestion]:
        """Check basic grammar rules."""
        suggestions = []

        for rule_name, pattern in self.grammar_rules.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                suggestion = EditingSuggestion(
                    type="grammar",
                    severity="medium",
                    location={
                        "line": content[: match.start()].count("\n") + 1,
                        "column": match.start() - content.rfind("\n", 0, match.start()),
                        "length": len(match.group()),
                    },
                    original_text=match.group(),
                    suggested_text=self._get_grammar_suggestion(
                        rule_name, match.group()
                    ),
                    explanation=f"Grammar issue: {rule_name}",
                    confidence=0.8,
                )
                suggestions.append(suggestion)

        return suggestions

    async def _check_advanced_grammar(self, content: str) -> List[EditingSuggestion]:
        """Use LLM for advanced grammar checking."""
        try:
            grammar_prompt = f"""
            Check the following text for grammatical errors and provide specific suggestions:

            TEXT:
            {content[:2000]}  # Limit to avoid token limits

            For each error found, provide:
            1. The incorrect text
            2. The corrected version
            3. A brief explanation

            Format as: ERROR: [incorrect] -> CORRECTION: [correct] (REASON: [explanation])
            """

            response = await self.llm_client.generate_response(
                messages=[{"role": "user", "content": grammar_prompt}],
                max_tokens=1000,
                temperature=0.1,
            )

            # Parse LLM response and create suggestions
            suggestions = self._parse_grammar_response(response, content)
            return suggestions

        except Exception as e:
            logger.error(f"Advanced grammar check failed: {str(e)}")
            return []

    def _parse_grammar_response(
        self, response: str, content: str
    ) -> List[EditingSuggestion]:
        """Parse LLM grammar response into suggestions."""
        suggestions = []

        # Simple parsing - in production, use more robust parsing
        lines = response.split("\n")
        for line in lines:
            if "ERROR:" in line and "CORRECTION:" in line:
                try:
                    parts = line.split(" -> ")
                    if len(parts) >= 2:
                        error_part = parts[0].replace("ERROR:", "").strip()
                        correction_part = (
                            parts[1]
                            .split("(REASON:")[0]
                            .replace("CORRECTION:", "")
                            .strip()
                        )
                        reason = (
                            parts[1].split("(REASON:")[1].replace(")", "").strip()
                            if "(REASON:" in parts[1]
                            else "Grammar correction"
                        )

                        # Find position in content
                        pos = content.find(error_part)
                        if pos != -1:
                            suggestion = EditingSuggestion(
                                type="grammar",
                                severity="medium",
                                location={
                                    "line": content[:pos].count("\n") + 1,
                                    "column": pos - content.rfind("\n", 0, pos),
                                    "length": len(error_part),
                                },
                                original_text=error_part,
                                suggested_text=correction_part,
                                explanation=reason,
                                confidence=0.7,
                            )
                            suggestions.append(suggestion)
                except:
                    continue

        return suggestions

    async def _check_tone_consistency(
        self, content: str, style: WritingStyle
    ) -> List[EditingSuggestion]:
        """Check tone consistency."""
        suggestions = []

        # Simple tone checking - can be enhanced with NLP
        formal_indicators = ["furthermore", "therefore", "consequently", "moreover"]
        informal_indicators = ["gonna", "wanna", "kinda", "sorta", "yeah"]

        if style.formality_level >= 7:  # Formal writing
            for indicator in informal_indicators:
                if indicator in content.lower():
                    pos = content.lower().find(indicator)
                    suggestions.append(
                        EditingSuggestion(
                            type="style",
                            severity="medium",
                            location={
                                "line": content[:pos].count("\n") + 1,
                                "column": pos - content.rfind("\n", 0, pos),
                                "length": len(indicator),
                            },
                            original_text=indicator,
                            suggested_text="[more formal alternative]",
                            explanation="Use more formal language for professional tone",
                            confidence=0.8,
                        )
                    )

        return suggestions

    async def _check_vocabulary(
        self, content: str, style: WritingStyle
    ) -> List[EditingSuggestion]:
        """Check vocabulary appropriateness."""
        suggestions = []

        # Check for weak words if high vocabulary complexity is desired
        if style.vocabulary_complexity >= 7:
            weak_words = self.style_rules["weak_words"]["words"]
            for word in weak_words:
                pattern = r"\b" + re.escape(word) + r"\b"
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    suggestions.append(
                        EditingSuggestion(
                            type="style",
                            severity="low",
                            location={
                                "line": content[: match.start()].count("\n") + 1,
                                "column": match.start()
                                - content.rfind("\n", 0, match.start()),
                                "length": len(match.group()),
                            },
                            original_text=match.group(),
                            suggested_text="[stronger alternative]",
                            explanation=self.style_rules["weak_words"]["message"],
                            confidence=0.6,
                        )
                    )

        return suggestions

    async def _check_sentence_structure(
        self, content: str, style: WritingStyle
    ) -> List[EditingSuggestion]:
        """Check sentence structure."""
        suggestions = []

        sentences = re.split(r"[.!?]+", content)
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                word_count = len(sentence.split())

                # Check for overly long sentences
                if word_count > 30 and style.sentence_structure != "complex":
                    suggestions.append(
                        EditingSuggestion(
                            type="style",
                            severity="medium",
                            location={"line": 0, "column": 0, "length": len(sentence)},
                            original_text=sentence.strip(),
                            suggested_text="[break into shorter sentences]",
                            explanation="Consider breaking long sentences for better readability",
                            confidence=0.7,
                        )
                    )

                # Check for overly short sentences in formal writing
                elif word_count < 8 and style.formality_level >= 7:
                    suggestions.append(
                        EditingSuggestion(
                            type="style",
                            severity="low",
                            location={"line": 0, "column": 0, "length": len(sentence)},
                            original_text=sentence.strip(),
                            suggested_text="[expand or combine with adjacent sentence]",
                            explanation="Consider expanding short sentences in formal writing",
                            confidence=0.6,
                        )
                    )

        return suggestions

    async def _check_voice(
        self, content: str, style: WritingStyle
    ) -> List[EditingSuggestion]:
        """Check active/passive voice usage."""
        suggestions = []

        if style.active_voice_preference > 0.7:
            passive_pattern = self.style_rules["passive_voice"]["pattern"]
            matches = re.finditer(passive_pattern, content, re.IGNORECASE)

            for match in matches:
                suggestions.append(
                    EditingSuggestion(
                        type="style",
                        severity="medium",
                        location={
                            "line": content[: match.start()].count("\n") + 1,
                            "column": match.start()
                            - content.rfind("\n", 0, match.start()),
                            "length": len(match.group()),
                        },
                        original_text=match.group(),
                        suggested_text="[active voice alternative]",
                        explanation=self.style_rules["passive_voice"]["message"],
                        confidence=0.7,
                    )
                )

        return suggestions

    def _get_grammar_suggestion(self, rule_name: str, original: str) -> str:
        """Get grammar correction suggestion."""
        corrections = {
            "double_space": " ",
            "comma_space": ", ",
            "period_space": ". ",
        }
        return corrections.get(rule_name, original)

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower().strip()
        if not word:
            return 0

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

        # Handle silent 'e'
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def _has_introduction(self, content: str) -> bool:
        """Check if content has an introduction."""
        intro_indicators = ["introduction", "overview", "background", "summary"]
        first_paragraph = content.split("\n\n")[0].lower()
        return any(indicator in first_paragraph for indicator in intro_indicators)

    def _has_conclusion(self, content: str) -> bool:
        """Check if content has a conclusion."""
        conclusion_indicators = [
            "conclusion",
            "summary",
            "in summary",
            "to conclude",
            "finally",
        ]
        last_paragraph = content.split("\n\n")[-1].lower()
        return any(indicator in last_paragraph for indicator in conclusion_indicators)

    def _extract_headings(self, content: str) -> List[str]:
        """Extract headings from content."""
        headings = []
        for line in content.split("\n"):
            if line.strip().startswith("#"):
                headings.append(line.strip())
        return headings

    def _check_keywords(self, content: str, keywords: List[str]) -> List[str]:
        """Check for missing keywords."""
        content_lower = content.lower()
        missing = []
        for keyword in keywords:
            if keyword.lower() not in content_lower:
                missing.append(keyword)
        return missing
