"""
Standalone quality verification script for ingestion results

This script provides detailed quality analysis of processed chunks and citations
without running the full ingestion pipeline.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import re


class QualityVerifier:
    """Comprehensive quality verification for ingestion results"""

    def __init__(self, output_dir="enhanced_output"):
        """
        Initialize quality verifier

        Args:
            output_dir: Directory containing enhanced chunk files
        """
        self.output_dir = Path(output_dir)
        self.quality_report = {}

    def verify_all(self):
        """Run all quality verifications"""

        print("🔍 Comprehensive Quality Verification")
        print("=" * 50)

        if not self.output_dir.exists():
            print(f"❌ Output directory not found: {self.output_dir}")
            return

        # Find enhanced files
        enhanced_files = list(self.output_dir.glob("enhanced_*.json"))
        if not enhanced_files:
            print(f"❌ No enhanced files found in {self.output_dir}")
            return

        print(f"📁 Found {len(enhanced_files)} enhanced files to analyze")

        # Run verifications
        chunk_report = self.verify_chunk_quality(enhanced_files)
        citation_report = self.verify_citation_quality(enhanced_files)
        metadata_report = self.verify_metadata_quality(enhanced_files)
        content_report = self.verify_content_quality(enhanced_files)

        # Generate comprehensive report
        self.generate_quality_report({
            "chunk_quality": chunk_report,
            "citation_quality": citation_report,
            "metadata_quality": metadata_report,
            "content_quality": content_report
        })

    def verify_chunk_quality(self, enhanced_files):
        """Verify chunk quality metrics"""

        print("\n✂️  Chunk Quality Analysis")
        print("-" * 30)

        chunk_stats = {
            "total_chunks": 0,
            "size_distribution": [],
            "word_count_distribution": [],
            "files_analyzed": 0,
            "size_issues": [],
            "content_issues": []
        }

        for file_path in enhanced_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)

                chunk_stats["files_analyzed"] += 1
                file_chunks = len(chunks)
                chunk_stats["total_chunks"] += file_chunks

                print(f"📄 {file_path.name}: {file_chunks} chunks")

                # Analyze each chunk
                for i, chunk in enumerate(chunks):
                    content = chunk.get('content', '')
                    metadata = chunk.get('metadata', {})

                    chunk_size = len(content)
                    word_count = len(content.split())

                    chunk_stats["size_distribution"].append(chunk_size)
                    chunk_stats["word_count_distribution"].append(word_count)

                    # Check for size issues
                    if chunk_size < 500:
                        chunk_stats["size_issues"].append({
                            "file": file_path.name,
                            "chunk": i,
                            "issue": f"Too small ({chunk_size} chars)",
                            "content_preview": content[:100]
                        })

                    if chunk_size > 8000:
                        chunk_stats["size_issues"].append({
                            "file": file_path.name,
                            "chunk": i,
                            "issue": f"Too large ({chunk_size} chars)"
                        })

                    # Check for content issues
                    if not content.strip():
                        chunk_stats["content_issues"].append({
                            "file": file_path.name,
                            "chunk": i,
                            "issue": "Empty content"
                        })

                    # Check for repetitive content (more than 80% same character)
                    if len(set(content.lower())) < len(content) * 0.2:
                        chunk_stats["content_issues"].append({
                            "file": file_path.name,
                            "chunk": i,
                            "issue": "Highly repetitive content"
                        })

            except Exception as e:
                print(f"   ❌ Error analyzing {file_path.name}: {e}")

        # Calculate statistics
        if chunk_stats["size_distribution"]:
            sizes = chunk_stats["size_distribution"]
            words = chunk_stats["word_count_distribution"]

            avg_size = sum(sizes) / len(sizes)
            avg_words = sum(words) / len(words)

            print(f"\n📊 Size Statistics:")
            print(f"   Total chunks: {chunk_stats['total_chunks']}")
            print(f"   Average size: {avg_size:.0f} characters")
            print(f"   Average words: {avg_words:.0f} words")
            print(f"   Size range: {min(sizes)} - {max(sizes)} characters")

            # Quality assessment
            size_quality = "Good"
            if avg_size < 1000 or avg_size > 6000:
                size_quality = "Needs attention"

            print(f"   Size quality: {size_quality}")

            # Report issues
            if chunk_stats["size_issues"]:
                print(f"\n⚠️  Size Issues Found ({len(chunk_stats['size_issues'])}):")
                for issue in chunk_stats["size_issues"][:5]:  # Show first 5
                    print(f"   - {issue['file']} chunk {issue['chunk']}: {issue['issue']}")
                if len(chunk_stats["size_issues"]) > 5:
                    print(f"   ... and {len(chunk_stats['size_issues']) - 5} more")

            if chunk_stats["content_issues"]:
                print(f"\n⚠️  Content Issues Found ({len(chunk_stats['content_issues'])}):")
                for issue in chunk_stats["content_issues"]:
                    print(f"   - {issue['file']} chunk {issue['chunk']}: {issue['issue']}")

        return chunk_stats

    def verify_citation_quality(self, enhanced_files):
        """Verify citation extraction and linking quality"""

        print("\n📚 Citation Quality Analysis")
        print("-" * 30)

        citation_stats = {
            "files_with_citations": 0,
            "total_citations": 0,
            "total_inline_citations": 0,
            "citation_types": Counter(),
            "linking_quality": [],
            "citation_examples": []
        }

        for file_path in enhanced_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)

                file_citations = 0
                file_inline_citations = 0

                for chunk in chunks:
                    metadata = chunk.get('metadata', {})
                    citations = metadata.get('citations', [])

                    if citations:
                        file_citations += len(citations)
                        citation_stats["total_citations"] += len(citations)

                        # Analyze citation quality
                        for citation in citations:
                            if isinstance(citation, dict):
                                # Count citation types
                                citation_type = citation.get('citation_type', 'unknown')
                                citation_stats["citation_types"][citation_type] += 1

                                # Check linking quality
                                inline_text = citation.get('inline_text', '')
                                full_ref = citation.get('full_reference', '')

                                if inline_text and full_ref:
                                    file_inline_citations += 1
                                    citation_stats["total_inline_citations"] += 1

                                    # Store example
                                    if len(citation_stats["citation_examples"]) < 10:
                                        citation_stats["citation_examples"].append({
                                            "file": file_path.name,
                                            "inline": inline_text,
                                            "full": full_ref[:100] + "..." if len(full_ref) > 100 else full_ref
                                        })

                if file_citations > 0:
                    citation_stats["files_with_citations"] += 1
                    linking_rate = (file_inline_citations / file_citations) * 100 if file_citations > 0 else 0
                    citation_stats["linking_quality"].append(linking_rate)

                    print(f"📄 {file_path.name}: {file_citations} citations, {linking_rate:.1f}% linked")

            except Exception as e:
                print(f"   ❌ Error analyzing {file_path.name}: {e}")

        # Citation summary
        print(f"\n📊 Citation Statistics:")
        print(f"   Files with citations: {citation_stats['files_with_citations']}")
        print(f"   Total citations: {citation_stats['total_citations']}")
        print(f"   Linked citations: {citation_stats['total_inline_citations']}")

        if citation_stats["total_citations"] > 0:
            overall_linking = (citation_stats["total_inline_citations"] / citation_stats["total_citations"]) * 100
            print(f"   Overall linking rate: {overall_linking:.1f}%")

            # Quality assessment
            if overall_linking > 80:
                print("   ✅ Excellent citation linking")
            elif overall_linking > 60:
                print("   ⚠️  Good citation linking")
            else:
                print("   ❌ Poor citation linking")

        # Citation types
        if citation_stats["citation_types"]:
            print(f"\n📋 Citation Types:")
            for ctype, count in citation_stats["citation_types"].most_common():
                print(f"   {ctype}: {count}")

        # Show examples
        if citation_stats["citation_examples"]:
            print(f"\n💡 Citation Examples:")
            for example in citation_stats["citation_examples"][:3]:
                print(f"   📄 {example['file']}:")
                print(f"      Inline: {example['inline']}")
                print(f"      Full: {example['full']}")

        return citation_stats

    def verify_metadata_quality(self, enhanced_files):
        """Verify metadata completeness and quality"""

        print("\n📋 Metadata Quality Analysis")
        print("-" * 30)

        metadata_stats = {
            "total_chunks": 0,
            "metadata_completeness": {
                "source_title": 0,
                "source_authors": 0,
                "source_year": 0,
                "source_journal": 0,
                "page_info": 0,
                "section_info": 0,
                "full_text": 0,
                "citation_info": 0
            },
            "metadata_examples": {},
            "missing_metadata": []
        }

        for file_path in enhanced_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)

                for chunk in chunks:
                    metadata = chunk.get('metadata', {})
                    metadata_stats["total_chunks"] += 1

                    # Check metadata completeness
                    if metadata.get('source_title'):
                        metadata_stats["metadata_completeness"]["source_title"] += 1
                        # Store example
                        if "source_title" not in metadata_stats["metadata_examples"]:
                            metadata_stats["metadata_examples"]["source_title"] = metadata['source_title']

                    if metadata.get('source_authors'):
                        metadata_stats["metadata_completeness"]["source_authors"] += 1
                        if "source_authors" not in metadata_stats["metadata_examples"]:
                            metadata_stats["metadata_examples"]["source_authors"] = metadata['source_authors']

                    if metadata.get('source_year'):
                        metadata_stats["metadata_completeness"]["source_year"] += 1

                    if metadata.get('source_journal'):
                        metadata_stats["metadata_completeness"]["source_journal"] += 1

                    if metadata.get('page_start') or metadata.get('page_end'):
                        metadata_stats["metadata_completeness"]["page_info"] += 1

                    if metadata.get('hierarchy_breadcrumb') or metadata.get('section_hierarchy'):
                        metadata_stats["metadata_completeness"]["section_info"] += 1

                    if metadata.get('text') and len(metadata.get('text', '')) > 500:
                        metadata_stats["metadata_completeness"]["full_text"] += 1

                    if metadata.get('citations') or metadata.get('citation_count'):
                        metadata_stats["metadata_completeness"]["citation_info"] += 1

            except Exception as e:
                print(f"   ❌ Error analyzing {file_path.name}: {e}")

        # Calculate percentages
        total = metadata_stats["total_chunks"]
        if total > 0:
            print(f"📊 Metadata Completeness ({total} chunks analyzed):")

            for field, count in metadata_stats["metadata_completeness"].items():
                percentage = (count / total) * 100
                status = "✅" if percentage > 80 else "⚠️" if percentage > 50 else "❌"
                field_name = field.replace('_', ' ').title()
                print(f"   {status} {field_name}: {percentage:.1f}% ({count}/{total})")

            # Show examples
            print(f"\n💡 Metadata Examples:")
            for field, example in metadata_stats["metadata_examples"].items():
                example_text = str(example)[:80] + "..." if len(str(example)) > 80 else str(example)
                print(f"   {field}: {example_text}")

        return metadata_stats

    def verify_content_quality(self, enhanced_files):
        """Verify content quality and structure"""

        print("\n📝 Content Quality Analysis")
        print("-" * 30)

        content_stats = {
            "total_chunks": 0,
            "language_issues": [],
            "structure_issues": [],
            "content_patterns": Counter(),
            "readability_scores": []
        }

        for file_path in enhanced_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)

                for i, chunk in enumerate(chunks):
                    content = chunk.get('content', '')
                    content_stats["total_chunks"] += 1

                    if not content.strip():
                        continue

                    # Check for language/encoding issues
                    if re.search(r'[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF\u2000-\u206F]', content):
                        content_stats["language_issues"].append({
                            "file": file_path.name,
                            "chunk": i,
                            "issue": "Non-standard characters detected"
                        })

                    # Check structure
                    sentences = re.split(r'[.!?]+', content)
                    paragraphs = content.split('\n\n')

                    if len(sentences) < 3:
                        content_stats["structure_issues"].append({
                            "file": file_path.name,
                            "chunk": i,
                            "issue": f"Very few sentences ({len(sentences)})"
                        })

                    # Identify content patterns
                    if re.search(r'(table|figure|fig\.)\s*\d+', content.lower()):
                        content_stats["content_patterns"]["contains_tables_figures"] += 1

                    if re.search(r'(abstract|introduction|conclusion|methods|results)', content.lower()):
                        content_stats["content_patterns"]["academic_structure"] += 1

                    if re.search(r'\([^)]*\d{4}[^)]*\)', content):
                        content_stats["content_patterns"]["contains_citations"] += 1

                    # Simple readability check (average sentence length)
                    if len(sentences) > 1:
                        avg_sentence_length = len(content.split()) / len(sentences)
                        content_stats["readability_scores"].append(avg_sentence_length)

            except Exception as e:
                print(f"   ❌ Error analyzing {file_path.name}: {e}")

        # Content quality summary
        print(f"📊 Content Analysis ({content_stats['total_chunks']} chunks):")

        if content_stats["content_patterns"]:
            print(f"\n📋 Content Patterns:")
            for pattern, count in content_stats["content_patterns"].most_common():
                percentage = (count / content_stats["total_chunks"]) * 100
                print(f"   {pattern.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

        if content_stats["readability_scores"]:
            avg_sentence_length = sum(content_stats["readability_scores"]) / len(content_stats["readability_scores"])
            print(f"\n📖 Readability:")
            print(f"   Average sentence length: {avg_sentence_length:.1f} words")

            if 15 <= avg_sentence_length <= 25:
                print("   ✅ Good readability")
            elif avg_sentence_length > 30:
                print("   ⚠️  Sentences may be too long")
            else:
                print("   ⚠️  Sentences may be too short")

        # Report issues
        if content_stats["language_issues"]:
            print(f"\n⚠️  Language Issues ({len(content_stats['language_issues'])}):")
            for issue in content_stats["language_issues"][:3]:
                print(f"   - {issue['file']} chunk {issue['chunk']}: {issue['issue']}")

        if content_stats["structure_issues"]:
            print(f"\n⚠️  Structure Issues ({len(content_stats['structure_issues'])}):")
            for issue in content_stats["structure_issues"][:3]:
                print(f"   - {issue['file']} chunk {issue['chunk']}: {issue['issue']}")

        return content_stats

    def generate_quality_report(self, all_reports):
        """Generate comprehensive quality report"""

        print("\n📊 Comprehensive Quality Report")
        print("=" * 50)

        # Overall quality score
        quality_scores = []

        # Chunk quality score (based on size distribution)
        chunk_report = all_reports["chunk_quality"]
        if chunk_report["size_distribution"]:
            sizes = chunk_report["size_distribution"]
            good_sizes = [s for s in sizes if 1500 <= s <= 5000]
            chunk_score = (len(good_sizes) / len(sizes)) * 100
            quality_scores.append(chunk_score)
            print(f"📏 Chunk Quality Score: {chunk_score:.1f}%")

        # Citation quality score
        citation_report = all_reports["citation_quality"]
        if citation_report["total_citations"] > 0:
            citation_score = (citation_report["total_inline_citations"] / citation_report["total_citations"]) * 100
            quality_scores.append(citation_score)
            print(f"📚 Citation Quality Score: {citation_score:.1f}%")

        # Metadata quality score
        metadata_report = all_reports["metadata_quality"]
        if metadata_report["total_chunks"] > 0:
            completeness = metadata_report["metadata_completeness"]
            total_chunks = metadata_report["total_chunks"]
            metadata_score = sum(count / total_chunks for count in completeness.values()) / len(completeness) * 100
            quality_scores.append(metadata_score)
            print(f"📋 Metadata Quality Score: {metadata_score:.1f}%")

        # Overall quality
        if quality_scores:
            overall_score = sum(quality_scores) / len(quality_scores)
            print(f"\n🎯 Overall Quality Score: {overall_score:.1f}%")

            if overall_score > 85:
                print("   ✅ Excellent ingestion quality")
            elif overall_score > 70:
                print("   ⚠️  Good ingestion quality")
            elif overall_score > 50:
                print("   ⚠️  Acceptable quality, room for improvement")
            else:
                print("   ❌ Poor quality, needs significant improvement")

        # Recommendations
        print(f"\n💡 Recommendations:")

        if chunk_report.get("size_issues"):
            print("   - Review chunking configuration for better size distribution")

        if citation_report["total_citations"] == 0:
            print("   - Enable citation extraction or check document format")
        elif citation_report.get("total_inline_citations", 0) < citation_report.get("total_citations", 1) * 0.7:
            print("   - Improve citation linking algorithms")

        if metadata_report["metadata_completeness"].get("source_title", 0) < metadata_report["total_chunks"] * 0.8:
            print("   - Improve document metadata extraction")

        print(f"\n✅ Quality verification completed!")


def main():
    """Run quality verification"""

    verifier = QualityVerifier()
    verifier.verify_all()


if __name__ == "__main__":
    main()