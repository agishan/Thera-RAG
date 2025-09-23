"""
Simple quality verification without Unicode characters
"""

import json
import os
from pathlib import Path
from collections import Counter
import re


def verify_chunk_quality():
    """Verify chunk quality"""

    print("Chunk Quality Verification")
    print("=" * 40)

    output_dir = Path("enhanced_output")
    if not output_dir.exists():
        print("ERROR: enhanced_output directory not found")
        return

    enhanced_files = list(output_dir.glob("enhanced_*.json"))
    if not enhanced_files:
        print("ERROR: No enhanced files found")
        return

    print(f"Analyzing {len(enhanced_files)} files...")

    total_chunks = 0
    size_stats = []
    word_stats = []
    issues = []

    for file_path in enhanced_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            file_chunks = len(chunks)
            total_chunks += file_chunks
            print(f"  {file_path.name}: {file_chunks} chunks")

            for i, chunk in enumerate(chunks):
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})

                chunk_size = len(content)
                word_count = len(content.split())

                size_stats.append(chunk_size)
                word_stats.append(word_count)

                # Quality checks
                if chunk_size < 500:
                    issues.append(f"{file_path.name} chunk {i}: Too small ({chunk_size} chars)")

                if chunk_size > 8000:
                    issues.append(f"{file_path.name} chunk {i}: Too large ({chunk_size} chars)")

                if not content.strip():
                    issues.append(f"{file_path.name} chunk {i}: Empty content")

        except Exception as e:
            print(f"  ERROR: {file_path.name}: {e}")

    # Statistics
    if size_stats:
        avg_size = sum(size_stats) / len(size_stats)
        min_size = min(size_stats)
        max_size = max(size_stats)
        avg_words = sum(word_stats) / len(word_stats)

        print(f"\nChunk Statistics:")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Average size: {avg_size:.0f} characters")
        print(f"  Size range: {min_size} - {max_size} characters")
        print(f"  Average words: {avg_words:.0f} words")

        # Quality assessment
        good_sizes = [s for s in size_stats if 1500 <= s <= 5000]
        size_quality = (len(good_sizes) / len(size_stats)) * 100

        print(f"  Optimal size chunks: {size_quality:.1f}%")

        if size_quality > 80:
            print("  ASSESSMENT: Excellent chunk sizes")
        elif size_quality > 60:
            print("  ASSESSMENT: Good chunk sizes")
        else:
            print("  ASSESSMENT: Poor chunk sizes")

        # Report issues
        if issues:
            print(f"\nIssues found ({len(issues)}):")
            for issue in issues[:5]:  # Show first 5
                print(f"  - {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more")
        else:
            print("\nNo critical issues found!")


def verify_metadata_quality():
    """Verify metadata completeness"""

    print("\nMetadata Quality Verification")
    print("=" * 40)

    output_dir = Path("enhanced_output")
    enhanced_files = list(output_dir.glob("enhanced_*.json"))

    metadata_stats = {
        "total_chunks": 0,
        "has_full_text": 0,
        "has_source_info": 0,
        "has_chunk_info": 0,
        "has_processing_info": 0
    }

    for file_path in enhanced_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            for chunk in chunks:
                metadata = chunk.get('metadata', {})
                metadata_stats["total_chunks"] += 1

                # Check metadata completeness
                if metadata.get('text') and len(metadata.get('text', '')) > 500:
                    metadata_stats["has_full_text"] += 1

                if metadata.get('source_file') or metadata.get('source_title'):
                    metadata_stats["has_source_info"] += 1

                if metadata.get('chunk_size') and metadata.get('word_count'):
                    metadata_stats["has_chunk_info"] += 1

                if metadata.get('processed_at') and metadata.get('processing_version'):
                    metadata_stats["has_processing_info"] += 1

        except Exception as e:
            print(f"  ERROR: {file_path.name}: {e}")

    # Report metadata quality
    total = metadata_stats["total_chunks"]
    if total > 0:
        print(f"Metadata Analysis ({total} chunks):")

        for field, count in metadata_stats.items():
            if field == "total_chunks":
                continue

            percentage = (count / total) * 100
            status = "GOOD" if percentage > 80 else "OK" if percentage > 50 else "POOR"
            field_name = field.replace('_', ' ').title()
            print(f"  {field_name}: {percentage:.1f}% ({count}/{total}) - {status}")


def verify_content_quality():
    """Verify content quality"""

    print("\nContent Quality Verification")
    print("=" * 40)

    output_dir = Path("enhanced_output")
    enhanced_files = list(output_dir.glob("enhanced_*.json"))

    content_patterns = Counter()
    readability_scores = []
    content_issues = []

    for file_path in enhanced_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            for i, chunk in enumerate(chunks):
                content = chunk.get('content', '')

                if not content.strip():
                    continue

                # Content pattern analysis
                if re.search(r'(table|figure|fig\.)\s*\d+', content.lower()):
                    content_patterns["has_tables_figures"] += 1

                if re.search(r'(abstract|introduction|methods|results|conclusion)', content.lower()):
                    content_patterns["academic_structure"] += 1

                if re.search(r'\([^)]*\d{4}[^)]*\)', content):
                    content_patterns["has_citations"] += 1

                if re.search(r'(p\.|page|pp\.)\s*\d+', content.lower()):
                    content_patterns["has_page_refs"] += 1

                # Simple readability check
                sentences = re.split(r'[.!?]+', content)
                if len(sentences) > 1:
                    avg_sentence_length = len(content.split()) / len(sentences)
                    readability_scores.append(avg_sentence_length)

                # Check for issues
                if len(set(content.lower())) < len(content) * 0.1:
                    content_issues.append(f"{file_path.name} chunk {i}: Highly repetitive")

        except Exception as e:
            print(f"  ERROR: {file_path.name}: {e}")

    # Report content analysis
    total_chunks = sum(content_patterns.values())
    if content_patterns:
        print("Content Pattern Analysis:")
        for pattern, count in content_patterns.most_common():
            pattern_name = pattern.replace('_', ' ').title()
            print(f"  {pattern_name}: {count} chunks")

    if readability_scores:
        avg_sentence_length = sum(readability_scores) / len(readability_scores)
        print(f"\nReadability Analysis:")
        print(f"  Average sentence length: {avg_sentence_length:.1f} words")

        if 15 <= avg_sentence_length <= 25:
            print("  ASSESSMENT: Good readability")
        elif avg_sentence_length > 30:
            print("  ASSESSMENT: Sentences may be too long")
        else:
            print("  ASSESSMENT: Sentences may be too short")

    if content_issues:
        print(f"\nContent Issues ({len(content_issues)}):")
        for issue in content_issues:
            print(f"  - {issue}")


def verify_citation_quality():
    """Verify citation extraction quality"""

    print("\nCitation Quality Verification")
    print("=" * 40)

    output_dir = Path("enhanced_output")
    enhanced_files = list(output_dir.glob("enhanced_*.json"))

    citation_stats = {
        "files_with_citations": 0,
        "total_citations": 0,
        "chunks_with_citations": 0
    }

    for file_path in enhanced_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            file_has_citations = False
            for chunk in chunks:
                metadata = chunk.get('metadata', {})
                citations = metadata.get('citations', [])

                if citations:
                    file_has_citations = True
                    citation_stats["chunks_with_citations"] += 1
                    citation_stats["total_citations"] += len(citations)

            if file_has_citations:
                citation_stats["files_with_citations"] += 1

        except Exception as e:
            print(f"  ERROR: {file_path.name}: {e}")

    # Report citation analysis
    print(f"Citation Analysis:")
    print(f"  Files with citations: {citation_stats['files_with_citations']}")
    print(f"  Total citations found: {citation_stats['total_citations']}")
    print(f"  Chunks with citations: {citation_stats['chunks_with_citations']}")

    if citation_stats["total_citations"] == 0:
        print("  NOTE: No citations found (API key needed for extraction)")
    else:
        print("  ASSESSMENT: Citation extraction working")


def generate_overall_report():
    """Generate overall quality report"""

    print("\nOverall Quality Report")
    print("=" * 40)

    output_dir = Path("enhanced_output")
    enhanced_files = list(output_dir.glob("enhanced_*.json"))

    if not enhanced_files:
        print("ERROR: No files to analyze")
        return

    # Calculate overall metrics
    total_chunks = 0
    total_size = 0
    full_text_preserved = 0
    good_size_chunks = 0

    for file_path in enhanced_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            for chunk in chunks:
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})

                total_chunks += 1
                chunk_size = len(content)
                total_size += chunk_size

                # Check full text preservation
                if len(metadata.get('text', '')) > 500:
                    full_text_preserved += 1

                # Check chunk size quality
                if 1500 <= chunk_size <= 5000:
                    good_size_chunks += 1

        except Exception as e:
            print(f"  ERROR: {file_path.name}: {e}")

    if total_chunks > 0:
        avg_size = total_size / total_chunks
        full_text_rate = (full_text_preserved / total_chunks) * 100
        size_quality_rate = (good_size_chunks / total_chunks) * 100

        print(f"Overall Quality Metrics:")
        print(f"  Total chunks analyzed: {total_chunks}")
        print(f"  Average chunk size: {avg_size:.0f} characters")
        print(f"  Full text preserved: {full_text_rate:.1f}%")
        print(f"  Optimal size chunks: {size_quality_rate:.1f}%")

        # Calculate overall score
        quality_score = 0

        # Size quality (30 points)
        if 2000 <= avg_size <= 4500:
            quality_score += 30
        elif 1500 <= avg_size <= 6000:
            quality_score += 20
        else:
            quality_score += 10

        # Full text preservation (40 points)
        if full_text_rate > 95:
            quality_score += 40
        elif full_text_rate > 80:
            quality_score += 30
        elif full_text_rate > 60:
            quality_score += 20
        else:
            quality_score += 10

        # Size distribution (30 points)
        if size_quality_rate > 80:
            quality_score += 30
        elif size_quality_rate > 60:
            quality_score += 20
        elif size_quality_rate > 40:
            quality_score += 15
        else:
            quality_score += 10

        print(f"\nOverall Quality Score: {quality_score}/100")

        if quality_score >= 85:
            print("RESULT: EXCELLENT - High quality ingestion")
        elif quality_score >= 70:
            print("RESULT: GOOD - Acceptable quality with minor issues")
        elif quality_score >= 50:
            print("RESULT: FAIR - Needs improvement")
        else:
            print("RESULT: POOR - Significant issues found")

        # Recommendations
        print(f"\nRecommendations:")
        if full_text_rate < 90:
            print("  - Check text preservation - some content may be truncated")
        if size_quality_rate < 70:
            print("  - Adjust chunking configuration for better size distribution")
        if avg_size < 2000:
            print("  - Consider increasing target chunk size")
        if avg_size > 5000:
            print("  - Consider decreasing target chunk size")


def main():
    """Run all quality verifications"""

    print("Enhanced Output Quality Verification")
    print("=" * 50)

    verify_chunk_quality()
    verify_metadata_quality()
    verify_content_quality()
    verify_citation_quality()
    generate_overall_report()

    print("\nQuality verification completed!")


if __name__ == "__main__":
    main()