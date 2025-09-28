# RAG Testing Guide

## 🎛️ UI Testing with Sliders

### Streamlit App Controls
Run the app and use the sidebar controls:

```bash
cd apps/inference
streamlit run src/main.py
```

**Available Controls:**
- **Retrieval K (3-30)**: Number of chunks retrieved from vector database
- **Citation Display K (3-20)**: Maximum citations shown in UI
- **Performance Metrics**: Real-time response time and retrieval stats

## 🧪 Automated Testing Scripts

### 1. Question List Testing (`test_questions.py`)

Test a list of questions with different K values:

```bash
# Basic testing with default K=15
python test_questions.py sample_questions.txt --output results.json

# Test multiple K values
python test_questions.py sample_questions.txt --k-values 5,10,15,20 --output k_comparison.json

# With custom config
python test_questions.py sample_questions.txt --config custom_config.json --output custom_results.json
```

**Output Format:**
```json
{
  "metadata": {
    "total_questions": 15,
    "k_values_tested": [5, 10, 15],
    "config": {...}
  },
  "results": [
    {
      "question_index": 1,
      "question": "How should I interpret R-time prolongation on TEG?",
      "k_value_results": [
        {
          "retrieval_k": 5,
          "answer": "...",
          "elapsed_time": 2.34,
          "num_sources": 5,
          "success": true
        }
      ]
    }
  ]
}
```

### 2. Vanilla vs RAG Comparison (`compare_prompts.py`)

Compare responses between vanilla LLM and RAG-enhanced LLM:

```bash
# Basic comparison
python compare_prompts.py sample_questions.txt --output comparison.json

# Multiple K values with analysis
python compare_prompts.py sample_questions.txt --k-values 5,15,25 --analyze --output detailed_comparison.json
```

**Output Format:**
```json
{
  "metadata": {...},
  "comparisons": [
    {
      "question": "How should I interpret R-time prolongation on TEG?",
      "responses": [
        {
          "method": "vanilla",
          "answer": "...",
          "elapsed_time": 1.23,
          "num_sources": 0
        },
        {
          "method": "rag",
          "answer": "...",
          "elapsed_time": 2.45,
          "num_sources": 15,
          "retrieval_k": 15
        }
      ]
    }
  ],
  "analysis": {
    "success_rates": {
      "vanilla": 0.95,
      "rag": 0.98
    },
    "performance_comparison": {
      "vanilla_avg_time": 1.2,
      "rag_avg_time": 2.4,
      "avg_sources_retrieved": 14.2
    }
  }
}
```

## 📋 Question File Format

Create a text file with one question per line:

```
# Comments start with #
How should I interpret R-time prolongation on TEG?
What are the normal values for thromboelastography parameters?
When should viscoelastic testing be used in cardiac surgery?
```

## ⚙️ Configuration Overrides

Create a JSON file to override default settings:

```json
{
  "retrieval_k": 20,
  "llm_temperature": 0.2,
  "llm_max_tokens": 4096
}
```

## 📊 Analysis Examples

### Performance Analysis
```bash
# Test different K values to find optimal performance
python test_questions.py sample_questions.txt --k-values 3,5,10,15,20,25,30 --output k_analysis.json

# Compare vanilla vs RAG effectiveness
python compare_prompts.py sample_questions.txt --k-values 10,20 --analyze --output effectiveness.json
```

### Quality Analysis
1. **Response Time vs K Value**: Plot elapsed_time against retrieval_k
2. **Answer Quality**: Manual review of answers for accuracy
3. **Citation Relevance**: Check if retrieved sources match the question topic
4. **Vanilla vs RAG**: Compare answer comprehensiveness and accuracy

## 🎯 Testing Recommendations

### For Development:
- Use **K=5-10** for fast iteration
- Test with **sample_questions.txt** (15 questions)
- Focus on **response time** optimization

### For Evaluation:
- Use **K=10,15,20** for comprehensive testing
- Create domain-specific question sets
- Run **vanilla vs RAG comparisons**
- Include **citation analysis**

### For Production:
- Test with **real user questions**
- Monitor **success rates** and **response times**
- Use **K=15** as default (good balance)
- Set **citation_display_k=10** to avoid UI overload

## 🔧 Troubleshooting

**Rate Limiting Issues:**
- Increase delays in scripts (currently 1-2 seconds)
- Use smaller question batches
- Monitor Gemini API quotas

**Memory Issues:**
- Reduce max K values
- Process questions in smaller batches
- Clear session state between runs

**Import Errors:**
- Ensure you're in the `apps/inference` directory
- Check Python path includes `src/`
- Verify all dependencies are installed