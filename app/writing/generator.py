"""
Document generation tools for various formats and types.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..llm import LLMClient
from ..logger import get_logger
from .models import (
    Citation,
    ContentFormat,
    GeneratedContent,
    WritingRequirements,
    WritingStyle,
)

logger = get_logger(__name__)


class DocumentGenerator:
    """
    Advanced document generation system for various formats and types.

    Provides report generation with executive summaries, technical documentation
    with code examples, presentation generation, and multi-format export capabilities.
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize the document generator."""
        self.llm_client = llm_client
        self.templates = self._load_templates()
        self.export_handlers = self._setup_export_handlers()

    async def generate_report(
        self,
        title: str,
        data: Dict[str, Any],
        requirements: WritingRequirements,
        include_executive_summary: bool = True,
        include_recommendations: bool = True,
    ) -> GeneratedContent:
        """
        Generate a comprehensive report with executive summary and key findings.

        Args:
            title: Report title
            data: Data and findings to include
            requirements: Writing requirements
            include_executive_summary: Whether to include executive summary
            include_recommendations: Whether to include recommendations

        Returns:
            Generated report content
        """
        logger.info(f"Generating report: {title}")

        try:
            # Build report structure
            sections = await self._build_report_structure(
                title,
                data,
                requirements,
                include_executive_summary,
                include_recommendations,
            )

            # Generate content for each section
            report_content = await self._generate_report_content(sections, requirements)

            # Create metadata
            from .models import ContentMetadata, TechnicalLevel

            metadata = ContentMetadata(
                title=title,
                author=None,
                created_at=datetime.now(),
                word_count=len(report_content.split()),
                character_count=len(report_content),
                reading_time_minutes=max(1, len(report_content.split()) // 200),
                readability_score=0.0,  # Will be calculated separately
                technical_level=requirements.technical_level,
                keywords=requirements.keywords,
                outline=self._extract_outline(report_content),
            )

            return GeneratedContent(
                content=report_content,
                metadata=metadata,
                citations=[],  # Will be populated by citation manager
                quality_score=0.85,  # Placeholder
                readability_score=0.0,  # Will be calculated separately
                structure_analysis=None,  # Will be calculated separately
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise

    async def generate_technical_documentation(
        self,
        title: str,
        code_examples: List[Dict[str, str]],
        api_specs: Optional[Dict] = None,
        requirements: WritingRequirements = None,
    ) -> GeneratedContent:
        """
        Generate technical documentation with code examples and explanations.

        Args:
            title: Documentation title
            code_examples: List of code examples with explanations
            api_specs: Optional API specifications
            requirements: Writing requirements

        Returns:
            Generated technical documentation
        """
        logger.info(f"Generating technical documentation: {title}")

        try:
            # Build documentation structure
            doc_prompt = await self._build_technical_doc_prompt(
                title, code_examples, api_specs, requirements
            )

            # Generate documentation content
            doc_content = await self.llm_client.generate_response(
                messages=[{"role": "user", "content": doc_prompt}],
                max_tokens=4000,
                temperature=0.3,
            )

            # Post-process to ensure proper code formatting
            doc_content = self._format_code_blocks(doc_content)

            # Create metadata
            from .models import ContentMetadata, TechnicalLevel

            metadata = ContentMetadata(
                title=title,
                author=None,
                created_at=datetime.now(),
                word_count=len(doc_content.split()),
                character_count=len(doc_content),
                reading_time_minutes=max(1, len(doc_content.split()) // 200),
                readability_score=0.0,
                technical_level=TechnicalLevel.ADVANCED,
                keywords=requirements.keywords if requirements else [],
                outline=self._extract_outline(doc_content),
            )

            return GeneratedContent(
                content=doc_content,
                metadata=metadata,
                citations=[],
                quality_score=0.88,
                readability_score=0.0,
                structure_analysis=None,
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Technical documentation generation failed: {str(e)}")
            raise

    async def generate_presentation(
        self,
        title: str,
        content_outline: List[str],
        slide_count: int = 10,
        style: WritingStyle = None,
    ) -> GeneratedContent:
        """
        Generate presentation content with slides.

        Args:
            title: Presentation title
            content_outline: Outline of content to cover
            slide_count: Target number of slides
            style: Presentation style

        Returns:
            Generated presentation content
        """
        logger.info(f"Generating presentation: {title} ({slide_count} slides)")

        try:
            presentation_prompt = f"""
            Create a {slide_count}-slide presentation on "{title}".

            CONTENT OUTLINE:
            {chr(10).join(f"- {item}" for item in content_outline)}

            PRESENTATION STRUCTURE:
            1. Title slide with engaging subtitle
            2. Agenda/Overview slide
            3. {slide_count - 3} content slides covering the outline
            4. Conclusion/Summary slide

            FORMAT REQUIREMENTS:
            - Use markdown format with clear slide separators
            - Each slide should have a clear title (## Slide Title)
            - Include bullet points for key information
            - Add speaker notes where helpful
            - Keep text concise and presentation-friendly
            - Include suggested visuals or diagrams where appropriate

            STYLE:
            - Professional and engaging tone
            - Clear, concise language
            - Logical flow between slides
            - Strong opening and closing

            Generate the complete presentation content:
            """

            presentation_content = await self.llm_client.generate_response(
                messages=[{"role": "user", "content": presentation_prompt}],
                max_tokens=3000,
                temperature=0.5,
            )

            # Create metadata
            from .models import ContentMetadata, TechnicalLevel

            metadata = ContentMetadata(
                title=title,
                author=None,
                created_at=datetime.now(),
                word_count=len(presentation_content.split()),
                character_count=len(presentation_content),
                reading_time_minutes=slide_count * 2,  # Estimate 2 minutes per slide
                readability_score=0.0,
                technical_level=TechnicalLevel.INTERMEDIATE,
                keywords=[],
                outline=self._extract_slide_titles(presentation_content),
            )

            return GeneratedContent(
                content=presentation_content,
                metadata=metadata,
                citations=[],
                quality_score=0.82,
                readability_score=0.0,
                structure_analysis=None,
                suggestions=[],
            )

        except Exception as e:
            logger.error(f"Presentation generation failed: {str(e)}")
            raise

    async def export_to_format(
        self,
        content: GeneratedContent,
        format: str,
        output_path: str,
        options: Optional[Dict] = None,
    ) -> str:
        """
        Export content to various formats (PDF, Word, HTML, Markdown).

        Args:
            content: Content to export
            format: Target format ('pdf', 'docx', 'html', 'markdown')
            output_path: Output file path
            options: Export options

        Returns:
            Path to exported file
        """
        logger.info(f"Exporting content to {format} format")

        try:
            handler = self.export_handlers.get(format.lower())
            if not handler:
                raise ValueError(f"Unsupported export format: {format}")

            return await handler(content, output_path, options or {})

        except Exception as e:
            logger.error(f"Export to {format} failed: {str(e)}")
            raise

    def _load_templates(self) -> Dict[str, str]:
        """Load document templates."""
        return {
            "report": """
# {title}

## Executive Summary
{executive_summary}

## Introduction
{introduction}

## Key Findings
{key_findings}

## Analysis
{analysis}

## Recommendations
{recommendations}

## Conclusion
{conclusion}

## Appendices
{appendices}
            """,
            "technical_doc": """
# {title}

## Overview
{overview}

## Getting Started
{getting_started}

## API Reference
{api_reference}

## Code Examples
{code_examples}

## Best Practices
{best_practices}

## Troubleshooting
{troubleshooting}
            """,
            "presentation": """
# {title}

## Slide 1: Title
{title_slide}

## Slide 2: Agenda
{agenda_slide}

{content_slides}

## Final Slide: Conclusion
{conclusion_slide}
            """,
        }

    def _setup_export_handlers(self) -> Dict[str, callable]:
        """Setup export format handlers."""
        return {
            "markdown": self._export_markdown,
            "html": self._export_html,
            "pdf": self._export_pdf,
            "docx": self._export_docx,
            "json": self._export_json,
            "csv": self._export_csv,
        }

    async def _build_report_structure(
        self,
        title: str,
        data: Dict[str, Any],
        requirements: WritingRequirements,
        include_executive_summary: bool,
        include_recommendations: bool,
    ) -> Dict[str, str]:
        """Build report structure and sections."""
        sections = {
            "title": title,
            "introduction": "Introduction and background information",
            "key_findings": "Key findings and results",
            "analysis": "Detailed analysis of the data",
            "conclusion": "Summary and conclusions",
        }

        if include_executive_summary:
            sections["executive_summary"] = "Executive summary of key points"

        if include_recommendations:
            sections["recommendations"] = "Recommendations and next steps"

        # Add data-specific sections
        if "methodology" in data:
            sections["methodology"] = "Methodology and approach"

        if "results" in data:
            sections["results"] = "Detailed results and findings"

        return sections

    async def _generate_report_content(
        self, sections: Dict[str, str], requirements: WritingRequirements
    ) -> str:
        """Generate content for each report section."""
        report_parts = []

        for section_name, section_description in sections.items():
            if section_name == "title":
                report_parts.append(f"# {section_description}\n")
                continue

            section_prompt = f"""
            Write a {section_name.replace('_', ' ')} section for a report.

            SECTION PURPOSE: {section_description}

            REQUIREMENTS:
            - Target audience: {requirements.target_audience}
            - Technical level: {requirements.technical_level.value}
            - Tone: {requirements.tone.value}
            - Length: 2-3 paragraphs

            Write the section content:
            """

            try:
                section_content = await self.llm_client.generate_response(
                    messages=[{"role": "user", "content": section_prompt}],
                    max_tokens=800,
                    temperature=0.4,
                )

                report_parts.append(f"## {section_name.replace('_', ' ').title()}\n")
                report_parts.append(f"{section_content}\n\n")

            except Exception as e:
                logger.error(f"Failed to generate {section_name} section: {str(e)}")
                report_parts.append(f"## {section_name.replace('_', ' ').title()}\n")
                report_parts.append(f"[Content for {section_name} section]\n\n")

        return "".join(report_parts)

    async def _build_technical_doc_prompt(
        self,
        title: str,
        code_examples: List[Dict[str, str]],
        api_specs: Optional[Dict],
        requirements: Optional[WritingRequirements],
    ) -> str:
        """Build prompt for technical documentation generation."""
        prompt = f"""
        Create comprehensive technical documentation for "{title}".

        CODE EXAMPLES:
        """

        for i, example in enumerate(code_examples, 1):
            prompt += f"""
        Example {i}: {example.get('title', f'Example {i}')}
        Language: {example.get('language', 'python')}
        Code:
        ```{example.get('language', 'python')}
        {example.get('code', '')}
        ```
        Description: {example.get('description', 'Code example')}
        """

        if api_specs:
            prompt += f"""

        API SPECIFICATIONS:
        {json.dumps(api_specs, indent=2)}
        """

        prompt += """

        DOCUMENTATION REQUIREMENTS:
        - Include clear overview and getting started section
        - Provide detailed explanations for each code example
        - Include API reference if applicable
        - Add best practices and common pitfalls
        - Include troubleshooting section
        - Use clear headings and structure
        - Format code blocks properly with syntax highlighting

        Generate the complete technical documentation:
        """

        return prompt

    def _format_code_blocks(self, content: str) -> str:
        """Ensure proper code block formatting."""
        # Add language specification to code blocks if missing
        import re

        # Find code blocks without language specification
        pattern = r"```\n((?:(?!```).*\n)*?)```"

        def add_language(match):
            code = match.group(1)
            # Simple heuristic to detect language
            if "def " in code or "import " in code or "print(" in code:
                return f"```python\n{code}```"
            elif "function " in code or "const " in code or "let " in code:
                return f"```javascript\n{code}```"
            elif "public class" in code or "System.out" in code:
                return f"```java\n{code}```"
            else:
                return f"```\n{code}```"

        return re.sub(pattern, add_language, content)

    def _extract_outline(self, content: str) -> List[str]:
        """Extract outline from content headings."""
        outline = []
        for line in content.split("\n"):
            if line.strip().startswith("#"):
                outline.append(line.strip())
        return outline

    def _extract_slide_titles(self, content: str) -> List[str]:
        """Extract slide titles from presentation content."""
        titles = []
        for line in content.split("\n"):
            if line.strip().startswith("## "):
                titles.append(line.strip())
        return titles

    async def _export_markdown(
        self, content: GeneratedContent, output_path: str, options: Dict
    ) -> str:
        """Export content to Markdown format."""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Add metadata header if requested
            markdown_content = content.content
            if options.get("include_metadata", False):
                metadata_header = f"""---
title: {content.metadata.title}
author: {content.metadata.author or 'Generated'}
date: {content.metadata.created_at.strftime('%Y-%m-%d')}
word_count: {content.metadata.word_count}
reading_time: {content.metadata.reading_time_minutes} minutes
---

"""
                markdown_content = metadata_header + markdown_content

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            logger.info(f"Exported to Markdown: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Markdown export failed: {str(e)}")
            raise

    async def _export_html(
        self, content: GeneratedContent, output_path: str, options: Dict
    ) -> str:
        """Export content to HTML format."""
        try:
            # Simple Markdown to HTML conversion
            html_content = self._markdown_to_html(content.content)

            # Wrap in HTML document
            full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{content.metadata.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_html)

            logger.info(f"Exported to HTML: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"HTML export failed: {str(e)}")
            raise

    async def _export_pdf(
        self, content: GeneratedContent, output_path: str, options: Dict
    ) -> str:
        """Export content to PDF format."""
        try:
            # This would require a PDF library like reportlab or weasyprint
            # For now, create a simple text-based PDF placeholder

            # Convert to HTML first
            html_path = output_path.replace(".pdf", ".html")
            await self._export_html(content, html_path, options)

            # In production, use a library like weasyprint:
            # from weasyprint import HTML
            # HTML(html_path).write_pdf(output_path)

            # For now, just copy the HTML file
            import shutil

            shutil.copy(html_path, output_path.replace(".pdf", "_pdf.html"))

            logger.info(f"PDF export placeholder created: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"PDF export failed: {str(e)}")
            raise

    async def _export_docx(
        self, content: GeneratedContent, output_path: str, options: Dict
    ) -> str:
        """Export content to Word document format."""
        try:
            # This would require python-docx library
            # For now, create a simple text file

            with open(output_path.replace(".docx", ".txt"), "w", encoding="utf-8") as f:
                f.write(f"Title: {content.metadata.title}\n")
                f.write(f"Created: {content.metadata.created_at}\n")
                f.write(f"Word Count: {content.metadata.word_count}\n\n")
                f.write(content.content)

            logger.info(f"DOCX export placeholder created: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"DOCX export failed: {str(e)}")
            raise

    async def _export_json(
        self, content: GeneratedContent, output_path: str, options: Dict
    ) -> str:
        """Export content to JSON format."""
        try:
            # Convert content to JSON structure
            json_data = {
                "metadata": {
                    "title": content.metadata.title,
                    "author": content.metadata.author,
                    "created_at": content.metadata.created_at.isoformat(),
                    "word_count": content.metadata.word_count,
                    "character_count": content.metadata.character_count,
                    "reading_time_minutes": content.metadata.reading_time_minutes,
                    "readability_score": content.metadata.readability_score,
                    "technical_level": content.metadata.technical_level.value,
                    "keywords": content.metadata.keywords,
                    "outline": content.metadata.outline,
                },
                "content": content.content,
                "quality_score": content.quality_score,
                "readability_score": content.readability_score,
                "citations": [
                    {
                        "id": citation.id,
                        "title": citation.title,
                        "authors": citation.authors,
                        "url": citation.url,
                        "publication_date": (
                            citation.publication_date.isoformat()
                            if citation.publication_date
                            else None
                        ),
                    }
                    for citation in content.citations
                ],
                "suggestions": content.suggestions,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported to JSON: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"JSON export failed: {str(e)}")
            raise

    async def _export_csv(
        self, content: GeneratedContent, output_path: str, options: Dict
    ) -> str:
        """Export content metadata to CSV format."""
        try:
            import csv

            # Extract data for CSV export
            rows = [
                ["Field", "Value"],
                ["Title", content.metadata.title],
                ["Author", content.metadata.author or ""],
                ["Created", content.metadata.created_at.isoformat()],
                ["Word Count", content.metadata.word_count],
                ["Character Count", content.metadata.character_count],
                ["Reading Time (minutes)", content.metadata.reading_time_minutes],
                ["Readability Score", content.metadata.readability_score],
                ["Technical Level", content.metadata.technical_level.value],
                ["Quality Score", content.quality_score],
                ["Keywords", "; ".join(content.metadata.keywords)],
            ]

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

            logger.info(f"Exported to CSV: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"CSV export failed: {str(e)}")
            raise

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Simple Markdown to HTML conversion."""
        html = markdown_text

        # Headers
        html = re.sub(r"^### (.*$)", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.*$)", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.*$)", r"<h1>\1</h1>", html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)

        # Code blocks
        html = re.sub(
            r"```(.*?)```", r"<pre><code>\1</code></pre>", html, flags=re.DOTALL
        )
        html = re.sub(r"`(.*?)`", r"<code>\1</code>", html)

        # Paragraphs
        paragraphs = html.split("\n\n")
        html_paragraphs = []
        for p in paragraphs:
            p = p.strip()
            if p and not p.startswith("<"):
                html_paragraphs.append(f"<p>{p}</p>")
            else:
                html_paragraphs.append(p)

        return "\n".join(html_paragraphs)
