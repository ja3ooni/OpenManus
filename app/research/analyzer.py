"""Cross-referencing and conflict detection for research findings."""

import asyncio
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from ..exceptions import ResearchError
from .models import (
    ConflictType,
    CrossReference,
    InformationConflict,
    ResearchFinding,
    SourceType,
)

logger = logging.getLogger(__name__)


class ResearchAnalyzer:
    """Analyzes research findings for cross-references and conflicts."""

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        conflict_threshold: float = 0.7,
        enable_semantic_analysis: bool = True,
    ):
        """Initialize the research analyzer."""
        self.similarity_threshold = similarity_threshold
        self.conflict_threshold = conflict_threshold
        self.enable_semantic_analysis = enable_semantic_analysis

        # Contradiction patterns for conflict detection
        self.contradiction_patterns = [
            # Direct opposites
            (
                r"\b(increase|rise|grow|up|higher|more)\b",
                r"\b(decrease|fall|decline|down|lower|less)\b",
            ),
            (
                r"\b(positive|good|beneficial|improve)\b",
                r"\b(negative|bad|harmful|worsen)\b",
            ),
            (
                r"\b(true|correct|accurate|right)\b",
                r"\b(false|incorrect|inaccurate|wrong)\b",
            ),
            (r"\b(yes|confirm|support|agree)\b", r"\b(no|deny|oppose|disagree)\b"),
            (
                r"\b(safe|secure|protected)\b",
                r"\b(unsafe|insecure|vulnerable|dangerous)\b",
            ),
            (
                r"\b(effective|successful|works)\b",
                r"\b(ineffective|unsuccessful|fails)\b",
            ),
            # Temporal contradictions
            (
                r"\b(before|prior|earlier|previous)\b",
                r"\b(after|later|subsequent|following)\b",
            ),
            (r"\b(past|historical|former)\b", r"\b(future|upcoming|planned)\b"),
            # Quantitative contradictions
            (r"\b(all|every|always|never)\b", r"\b(some|few|sometimes|often)\b"),
            (r"\b(maximum|highest|peak)\b", r"\b(minimum|lowest|bottom)\b"),
        ]

        # Supporting relationship patterns
        self.support_patterns = [
            r"\b(confirm|support|validate|corroborate|verify)\b",
            r"\b(similar|same|identical|equivalent|comparable)\b",
            r"\b(also|additionally|furthermore|moreover|likewise)\b",
            r"\b(consistent|aligned|coherent|compatible)\b",
        ]

        # Evidence strength indicators
        self.evidence_indicators = {
            "strong": [
                "study shows",
                "research proves",
                "data confirms",
                "evidence demonstrates",
                "analysis reveals",
                "findings indicate",
                "results show",
                "statistics prove",
            ],
            "moderate": [
                "suggests",
                "indicates",
                "implies",
                "appears",
                "seems",
                "likely",
                "probably",
                "potentially",
                "may",
                "might",
                "could",
            ],
            "weak": [
                "claims",
                "alleges",
                "believes",
                "thinks",
                "feels",
                "opinions",
                "rumors",
                "speculation",
                "unconfirmed",
                "anecdotal",
            ],
        }

    async def cross_reference_findings(
        self, findings: List[ResearchFinding]
    ) -> List[CrossReference]:
        """Identify cross-references between research findings."""
        try:
            logger.info(f"Cross-referencing {len(findings)} findings")

            cross_references = []

            # Compare each finding with every other finding
            for i, finding1 in enumerate(findings):
                for j, finding2 in enumerate(findings[i + 1 :], i + 1):
                    cross_ref = await self._analyze_relationship(finding1, finding2)
                    if cross_ref:
                        cross_references.append(cross_ref)

            # Group related findings
            grouped_refs = await self._group_related_findings(
                cross_references, findings
            )
            cross_references.extend(grouped_refs)

            logger.info(f"Found {len(cross_references)} cross-references")
            return cross_references

        except Exception as e:
            logger.error(f"Failed to cross-reference findings: {str(e)}")
            raise ResearchError(f"Cross-referencing failed: {str(e)}") from e

    async def detect_conflicts(
        self, findings: List[ResearchFinding]
    ) -> List[InformationConflict]:
        """Detect conflicts between research findings."""
        try:
            logger.info(f"Detecting conflicts in {len(findings)} findings")

            conflicts = []

            # Compare findings for contradictions
            for i, finding1 in enumerate(findings):
                for j, finding2 in enumerate(findings[i + 1 :], i + 1):
                    conflict = await self._detect_contradiction(finding1, finding2)
                    if conflict:
                        conflicts.append(conflict)

            # Detect temporal conflicts
            temporal_conflicts = await self._detect_temporal_conflicts(findings)
            conflicts.extend(temporal_conflicts)

            # Detect methodological conflicts
            method_conflicts = await self._detect_methodological_conflicts(findings)
            conflicts.extend(method_conflicts)

            # Remove duplicate conflicts
            conflicts = self._deduplicate_conflicts(conflicts)

            logger.info(f"Detected {len(conflicts)} conflicts")
            return conflicts

        except Exception as e:
            logger.error(f"Failed to detect conflicts: {str(e)}")
            raise ResearchError(f"Conflict detection failed: {str(e)}") from e

    async def analyze_perspectives(
        self, findings: List[ResearchFinding]
    ) -> Dict[str, List[ResearchFinding]]:
        """Analyze and group findings by different perspectives."""
        try:
            logger.info(f"Analyzing perspectives in {len(findings)} findings")

            perspectives = defaultdict(list)

            # Group by source type
            for finding in findings:
                source_type = finding.source.source_type.value
                perspectives[f"source_{source_type}"].append(finding)

            # Group by stance/sentiment
            for finding in findings:
                stance = await self._determine_stance(finding)
                if stance:
                    perspectives[f"stance_{stance}"].append(finding)

            # Group by temporal context
            for finding in findings:
                temporal_context = await self._determine_temporal_context(finding)
                if temporal_context:
                    perspectives[f"time_{temporal_context}"].append(finding)

            # Filter out perspectives with only one finding
            filtered_perspectives = {
                k: v for k, v in perspectives.items() if len(v) > 1
            }

            logger.info(
                f"Identified {len(filtered_perspectives)} different perspectives"
            )
            return dict(filtered_perspectives)

        except Exception as e:
            logger.error(f"Failed to analyze perspectives: {str(e)}")
            return {}

    async def score_evidence_strength(
        self, findings: List[ResearchFinding]
    ) -> List[ResearchFinding]:
        """Score the evidence strength of research findings."""
        try:
            logger.info(f"Scoring evidence strength for {len(findings)} findings")

            for finding in findings:
                strength_score = await self._calculate_evidence_strength(finding)

                # Update finding metadata with evidence strength
                if "evidence_strength" not in finding.metadata:
                    finding.metadata["evidence_strength"] = strength_score

                # Adjust confidence score based on evidence strength
                finding.confidence_score = min(
                    finding.confidence_score * (0.5 + strength_score * 0.5), 1.0
                )

            logger.info("Evidence strength scoring completed")
            return findings

        except Exception as e:
            logger.error(f"Failed to score evidence strength: {str(e)}")
            return findings

    async def _analyze_relationship(
        self, finding1: ResearchFinding, finding2: ResearchFinding
    ) -> Optional[CrossReference]:
        """Analyze the relationship between two findings."""
        try:
            # Calculate content similarity
            similarity = self._calculate_content_similarity(
                finding1.content, finding2.content
            )

            if similarity < self.similarity_threshold:
                return None

            # Determine relationship type
            relationship_type = await self._determine_relationship_type(
                finding1, finding2, similarity
            )

            # Calculate relationship strength
            strength = await self._calculate_relationship_strength(
                finding1, finding2, similarity
            )

            # Calculate confidence
            confidence = min(finding1.confidence_score, finding2.confidence_score)

            # Generate description
            description = await self._generate_relationship_description(
                finding1, finding2, relationship_type, similarity
            )

            return CrossReference(
                finding_ids=[finding1.id, finding2.id],
                relationship_type=relationship_type,
                strength=strength,
                confidence=confidence,
                description=description,
            )

        except Exception as e:
            logger.debug(f"Failed to analyze relationship: {str(e)}")
            return None

    async def _detect_contradiction(
        self, finding1: ResearchFinding, finding2: ResearchFinding
    ) -> Optional[InformationConflict]:
        """Detect contradiction between two findings."""
        try:
            content1 = finding1.content.lower()
            content2 = finding2.content.lower()

            # Check for direct contradictions using patterns
            for pattern1, pattern2 in self.contradiction_patterns:
                if (
                    re.search(pattern1, content1) and re.search(pattern2, content2)
                ) or (re.search(pattern2, content1) and re.search(pattern1, content2)):

                    severity = self._calculate_conflict_severity(finding1, finding2)

                    if severity >= self.conflict_threshold:
                        return InformationConflict(
                            conflicting_findings=[finding1.id, finding2.id],
                            conflict_type=ConflictType.DIRECT_CONTRADICTION,
                            severity=severity,
                            description=f"Direct contradiction detected between findings",
                            resolution_suggestions=await self._generate_resolution_suggestions(
                                finding1, finding2, ConflictType.DIRECT_CONTRADICTION
                            ),
                        )

            # Check for partial disagreements
            disagreement_score = self._calculate_disagreement_score(content1, content2)
            if disagreement_score >= 0.6:
                return InformationConflict(
                    conflicting_findings=[finding1.id, finding2.id],
                    conflict_type=ConflictType.PARTIAL_DISAGREEMENT,
                    severity=disagreement_score,
                    description=f"Partial disagreement detected (score: {disagreement_score:.2f})",
                    resolution_suggestions=await self._generate_resolution_suggestions(
                        finding1, finding2, ConflictType.PARTIAL_DISAGREEMENT
                    ),
                )

            return None

        except Exception as e:
            logger.debug(f"Failed to detect contradiction: {str(e)}")
            return None

    async def _detect_temporal_conflicts(
        self, findings: List[ResearchFinding]
    ) -> List[InformationConflict]:
        """Detect temporal conflicts between findings."""
        temporal_conflicts = []

        try:
            # Group findings by topic/subject
            topic_groups = self._group_findings_by_topic(findings)

            for topic, topic_findings in topic_groups.items():
                if len(topic_findings) < 2:
                    continue

                # Sort by timestamp or publication date
                sorted_findings = sorted(
                    topic_findings, key=lambda f: f.timestamp or datetime.min
                )

                # Check for temporal inconsistencies
                for i in range(len(sorted_findings) - 1):
                    older_finding = sorted_findings[i]
                    newer_finding = sorted_findings[i + 1]

                    # Check if newer finding contradicts older one
                    if self._has_temporal_contradiction(older_finding, newer_finding):
                        conflict = InformationConflict(
                            conflicting_findings=[older_finding.id, newer_finding.id],
                            conflict_type=ConflictType.TEMPORAL_DIFFERENCE,
                            severity=0.6,
                            description=f"Temporal conflict: newer information contradicts older",
                            resolution_suggestions=[
                                "Prioritize more recent information",
                                "Check for context changes over time",
                                "Verify source update dates",
                            ],
                        )
                        temporal_conflicts.append(conflict)

        except Exception as e:
            logger.debug(f"Failed to detect temporal conflicts: {str(e)}")

        return temporal_conflicts

    async def _detect_methodological_conflicts(
        self, findings: List[ResearchFinding]
    ) -> List[InformationConflict]:
        """Detect methodological conflicts between findings."""
        method_conflicts = []

        try:
            # Look for findings that discuss methodology
            method_findings = [
                f
                for f in findings
                if any(
                    term in f.content.lower()
                    for term in [
                        "study",
                        "research",
                        "analysis",
                        "method",
                        "approach",
                        "survey",
                        "experiment",
                        "data",
                        "sample",
                    ]
                )
            ]

            for i, finding1 in enumerate(method_findings):
                for finding2 in method_findings[i + 1 :]:
                    if self._has_methodological_difference(finding1, finding2):
                        conflict = InformationConflict(
                            conflicting_findings=[finding1.id, finding2.id],
                            conflict_type=ConflictType.METHODOLOGICAL_DIFFERENCE,
                            severity=0.5,
                            description="Different methodological approaches may lead to different conclusions",
                            resolution_suggestions=[
                                "Compare methodology quality and rigor",
                                "Consider sample sizes and selection criteria",
                                "Evaluate statistical significance",
                                "Check for peer review status",
                            ],
                        )
                        method_conflicts.append(conflict)

        except Exception as e:
            logger.debug(f"Failed to detect methodological conflicts: {str(e)}")

        return method_conflicts

    async def _group_related_findings(
        self, cross_references: List[CrossReference], findings: List[ResearchFinding]
    ) -> List[CrossReference]:
        """Group related findings into clusters."""
        grouped_refs = []

        try:
            # Create finding clusters based on cross-references
            clusters = self._create_finding_clusters(cross_references, findings)

            # Create group cross-references for each cluster
            for cluster_id, cluster_findings in clusters.items():
                if len(cluster_findings) > 2:
                    # Create a group cross-reference
                    group_ref = CrossReference(
                        finding_ids=[f.id for f in cluster_findings],
                        relationship_type="cluster",
                        strength=0.7,
                        confidence=sum(f.confidence_score for f in cluster_findings)
                        / len(cluster_findings),
                        description=f"Cluster of {len(cluster_findings)} related findings",
                    )
                    grouped_refs.append(group_ref)

        except Exception as e:
            logger.debug(f"Failed to group related findings: {str(e)}")

        return grouped_refs

    def _create_finding_clusters(
        self, cross_references: List[CrossReference], findings: List[ResearchFinding]
    ) -> Dict[str, List[ResearchFinding]]:
        """Create clusters of related findings."""
        clusters = {}
        finding_to_cluster = {}

        # Create initial clusters from cross-references
        for ref in cross_references:
            if ref.strength >= 0.7:  # High similarity threshold for clustering
                cluster_id = None

                # Check if any finding is already in a cluster
                for finding_id in ref.finding_ids:
                    if finding_id in finding_to_cluster:
                        cluster_id = finding_to_cluster[finding_id]
                        break

                # Create new cluster if needed
                if cluster_id is None:
                    cluster_id = str(uuid4())
                    clusters[cluster_id] = []

                # Add findings to cluster
                for finding_id in ref.finding_ids:
                    if finding_id not in finding_to_cluster:
                        finding = next(
                            (f for f in findings if f.id == finding_id), None
                        )
                        if finding:
                            clusters[cluster_id].append(finding)
                            finding_to_cluster[finding_id] = cluster_id

        return clusters

    async def _determine_relationship_type(
        self, finding1: ResearchFinding, finding2: ResearchFinding, similarity: float
    ) -> str:
        """Determine the type of relationship between two findings."""
        content1 = finding1.content.lower()
        content2 = finding2.content.lower()

        # Check for supporting relationship
        for pattern in self.support_patterns:
            if re.search(pattern, content1) or re.search(pattern, content2):
                return "supports"

        # Check for contradiction
        for pattern1, pattern2 in self.contradiction_patterns:
            if (re.search(pattern1, content1) and re.search(pattern2, content2)) or (
                re.search(pattern2, content1) and re.search(pattern1, content2)
            ):
                return "contradicts"

        # Default relationship based on similarity
        if similarity > 0.8:
            return "supports"
        elif similarity > 0.6:
            return "relates_to"
        else:
            return "mentions"

    async def _calculate_relationship_strength(
        self, finding1: ResearchFinding, finding2: ResearchFinding, similarity: float
    ) -> float:
        """Calculate the strength of relationship between findings."""
        # Base strength from content similarity
        strength = similarity

        # Adjust based on source credibility
        avg_credibility = (
            finding1.source.credibility_score + finding2.source.credibility_score
        ) / 2
        strength *= 0.5 + avg_credibility * 0.5

        # Adjust based on source diversity (different sources = stronger)
        if finding1.source.id != finding2.source.id:
            strength *= 1.1

        return min(strength, 1.0)

    async def _generate_relationship_description(
        self,
        finding1: ResearchFinding,
        finding2: ResearchFinding,
        relationship_type: str,
        similarity: float,
    ) -> str:
        """Generate description for the relationship."""
        source1_name = finding1.source.name
        source2_name = finding2.source.name

        descriptions = {
            "supports": f"Findings from {source1_name} and {source2_name} support each other (similarity: {similarity:.2f})",
            "contradicts": f"Findings from {source1_name} and {source2_name} contradict each other",
            "relates_to": f"Findings from {source1_name} and {source2_name} are related (similarity: {similarity:.2f})",
            "mentions": f"Findings from {source1_name} and {source2_name} mention similar topics",
        }

        return descriptions.get(
            relationship_type, f"Relationship between {source1_name} and {source2_name}"
        )

    async def _generate_resolution_suggestions(
        self,
        finding1: ResearchFinding,
        finding2: ResearchFinding,
        conflict_type: ConflictType,
    ) -> List[str]:
        """Generate suggestions for resolving conflicts."""
        base_suggestions = [
            "Compare source credibility and authority",
            "Check publication dates for temporal relevance",
            "Look for additional sources to break the tie",
            "Consider different contexts or scopes",
        ]

        type_specific = {
            ConflictType.DIRECT_CONTRADICTION: [
                "Verify facts with primary sources",
                "Check for data errors or misinterpretation",
            ],
            ConflictType.TEMPORAL_DIFFERENCE: [
                "Prioritize more recent information",
                "Check for policy or situation changes",
            ],
            ConflictType.METHODOLOGICAL_DIFFERENCE: [
                "Compare methodology rigor and sample sizes",
                "Look for meta-analyses or systematic reviews",
            ],
            ConflictType.DIFFERENT_PERSPECTIVE: [
                "Present multiple viewpoints",
                "Look for common ground or synthesis",
            ],
        }

        suggestions = base_suggestions.copy()
        suggestions.extend(type_specific.get(conflict_type, []))

        return suggestions

    async def _determine_stance(self, finding: ResearchFinding) -> Optional[str]:
        """Determine the stance/sentiment of a finding."""
        content = finding.content.lower()

        positive_indicators = [
            "positive",
            "good",
            "beneficial",
            "effective",
            "successful",
            "improve",
            "increase",
            "better",
            "advantage",
            "support",
        ]

        negative_indicators = [
            "negative",
            "bad",
            "harmful",
            "ineffective",
            "unsuccessful",
            "worsen",
            "decrease",
            "worse",
            "disadvantage",
            "oppose",
        ]

        positive_count = sum(
            1 for indicator in positive_indicators if indicator in content
        )
        negative_count = sum(
            1 for indicator in negative_indicators if indicator in content
        )

        if positive_count > negative_count and positive_count > 0:
            return "positive"
        elif negative_count > positive_count and negative_count > 0:
            return "negative"
        else:
            return "neutral"

    async def _determine_temporal_context(
        self, finding: ResearchFinding
    ) -> Optional[str]:
        """Determine temporal context of a finding."""
        content = finding.content.lower()

        past_indicators = [
            "was",
            "were",
            "had",
            "did",
            "ago",
            "previously",
            "former",
            "historical",
        ]
        present_indicators = [
            "is",
            "are",
            "has",
            "do",
            "currently",
            "now",
            "today",
            "present",
        ]
        future_indicators = [
            "will",
            "shall",
            "going to",
            "planned",
            "expected",
            "future",
            "upcoming",
        ]

        past_count = sum(1 for indicator in past_indicators if indicator in content)
        present_count = sum(
            1 for indicator in present_indicators if indicator in content
        )
        future_count = sum(1 for indicator in future_indicators if indicator in content)

        max_count = max(past_count, present_count, future_count)

        if max_count == 0:
            return None
        elif past_count == max_count:
            return "past"
        elif present_count == max_count:
            return "present"
        else:
            return "future"

    async def _calculate_evidence_strength(self, finding: ResearchFinding) -> float:
        """Calculate evidence strength score for a finding."""
        content = finding.content.lower()

        strong_count = sum(
            1
            for indicator in self.evidence_indicators["strong"]
            if indicator in content
        )
        moderate_count = sum(
            1
            for indicator in self.evidence_indicators["moderate"]
            if indicator in content
        )
        weak_count = sum(
            1 for indicator in self.evidence_indicators["weak"] if indicator in content
        )

        # Calculate weighted score
        score = strong_count * 1.0 + moderate_count * 0.6 + weak_count * 0.2
        total_indicators = strong_count + moderate_count + weak_count

        if total_indicators > 0:
            score = score / total_indicators
        else:
            score = 0.5  # Default neutral score

        # Adjust based on source type
        source_multiplier = {
            SourceType.ACADEMIC: 1.2,
            SourceType.GOVERNMENT: 1.1,
            SourceType.TECHNICAL: 1.0,
            SourceType.NEWS: 0.9,
            SourceType.COMMERCIAL: 0.7,
            SourceType.WEB: 0.6,
            SourceType.SOCIAL: 0.4,
        }.get(finding.source.source_type, 1.0)

        return min(score * source_multiplier, 1.0)

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _calculate_conflict_severity(
        self, finding1: ResearchFinding, finding2: ResearchFinding
    ) -> float:
        """Calculate severity of conflict between findings."""
        # Base severity from source credibility difference
        credibility_diff = abs(
            finding1.source.credibility_score - finding2.source.credibility_score
        )
        base_severity = 0.5 + credibility_diff * 0.3

        # Adjust based on confidence scores
        avg_confidence = (finding1.confidence_score + finding2.confidence_score) / 2
        severity = base_severity * avg_confidence

        return min(severity, 1.0)

    def _calculate_disagreement_score(self, content1: str, content2: str) -> float:
        """Calculate disagreement score between two pieces of content."""
        # Simple disagreement detection based on opposing terms
        disagreement_pairs = [
            ("agree", "disagree"),
            ("support", "oppose"),
            ("accept", "reject"),
            ("confirm", "deny"),
            ("approve", "disapprove"),
            ("like", "dislike"),
        ]

        disagreement_count = 0
        for term1, term2 in disagreement_pairs:
            if (term1 in content1 and term2 in content2) or (
                term2 in content1 and term1 in content2
            ):
                disagreement_count += 1

        return min(disagreement_count * 0.3, 1.0)

    def _group_findings_by_topic(
        self, findings: List[ResearchFinding]
    ) -> Dict[str, List[ResearchFinding]]:
        """Group findings by topic/subject."""
        topic_groups = defaultdict(list)

        for finding in findings:
            # Use key points as topic indicators
            if finding.key_points:
                topic = finding.key_points[0][:50]  # First key point as topic
            else:
                # Use first few words of content as topic
                topic = " ".join(finding.content.split()[:5])

            topic_groups[topic].append(finding)

        return dict(topic_groups)

    def _has_temporal_contradiction(
        self, older_finding: ResearchFinding, newer_finding: ResearchFinding
    ) -> bool:
        """Check if newer finding contradicts older finding."""
        older_content = older_finding.content.lower()
        newer_content = newer_finding.content.lower()

        # Check for direct contradictions
        for pattern1, pattern2 in self.contradiction_patterns:
            if (
                re.search(pattern1, older_content)
                and re.search(pattern2, newer_content)
            ) or (
                re.search(pattern2, older_content)
                and re.search(pattern1, newer_content)
            ):
                return True

        return False

    def _has_methodological_difference(
        self, finding1: ResearchFinding, finding2: ResearchFinding
    ) -> bool:
        """Check if findings have methodological differences."""
        content1 = finding1.content.lower()
        content2 = finding2.content.lower()

        method_terms = [
            "survey",
            "experiment",
            "study",
            "analysis",
            "research",
            "sample",
            "data",
            "methodology",
            "approach",
            "technique",
        ]

        # Check if both findings mention methodology
        method1 = any(term in content1 for term in method_terms)
        method2 = any(term in content2 for term in method_terms)

        if method1 and method2:
            # Simple check for different methodological approaches
            quantitative_terms = [
                "statistical",
                "numerical",
                "quantitative",
                "data",
                "survey",
            ]
            qualitative_terms = [
                "qualitative",
                "interview",
                "observation",
                "case study",
            ]

            quant1 = any(term in content1 for term in quantitative_terms)
            qual1 = any(term in content1 for term in qualitative_terms)
            quant2 = any(term in content2 for term in quantitative_terms)
            qual2 = any(term in content2 for term in qualitative_terms)

            # Different methodological approaches
            return (quant1 and qual2) or (qual1 and quant2)

        return False

    def _deduplicate_conflicts(
        self, conflicts: List[InformationConflict]
    ) -> List[InformationConflict]:
        """Remove duplicate conflicts."""
        seen_pairs = set()
        deduplicated = []

        for conflict in conflicts:
            # Create a sorted tuple of finding IDs to identify duplicates
            finding_pair = tuple(sorted(conflict.conflicting_findings))

            if finding_pair not in seen_pairs:
                seen_pairs.add(finding_pair)
                deduplicated.append(conflict)

        return deduplicated
