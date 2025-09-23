# Prompt Architecture Summary

## 📋 **Current Prompt Structure**

### **1. RAG Prompt** (Context-Aware)
- **Template:** `src/app/prompts/templates/medical_rag.txt`
- **Purpose:** Medical Q&A with retrieved document context
- **Variables:** `{context}`, `{question}`
- **Usage:** Main RAG application

### **2. Vanilla Prompt** (No Context)
- **Template:** `src/app/prompts/templates/vanilla_medical.txt`
- **Purpose:** Medical Q&A using LLM knowledge only
- **Variables:** `{question}`
- **Usage:** Comparison baseline

### **3. Cleansing Prompt** (Bias Removal)
- **Template:** `src/app/prompts/templates/cleansing_medical.txt`
- **Purpose:** Remove bias indicators for fair comparison
- **Variables:** `{context_type}`, `{text}`
- **Usage:** Post-processing for comparisons

## 🗂️ **File Organization**

```
src/app/prompts/
├── __init__.py
├── base_prompts.py          # Base PromptManager class
├── medical_prompts.py       # MedicalPromptManager class
└── templates/
    ├── medical_rag.txt      # RAG prompt
    ├── vanilla_medical.txt  # Vanilla prompt
    └── cleansing_medical.txt # Cleansing prompt
```

## 🔄 **Prompt Flow**

```
Question Input
     ↓
┌─────────────────┬─────────────────┐
│   RAG System    │  Vanilla LLM    │
│                 │                 │
│ RAG Prompt +    │ Vanilla Prompt  │
│ {context} +     │ (no context) +  │
│ {question}      │ {question}      │
│       ↓         │       ↓         │
│ RAG Response    │ Vanilla Response│
└─────────────────┴─────────────────┘
     ↓                     ↓
┌─────────────────────────────────────┐
│        Cleansing Prompt             │
│    (removes bias indicators)        │
│              ↓                      │
│   Clean RAG vs Clean Vanilla        │
│        (fair comparison)            │
└─────────────────────────────────────┘
```

## 🎯 **Usage**

### **Main RAG App:**
- Uses `medical_rag` template automatically
- Conversation mode available (hidden from UI)

### **Comparison Pipeline:**
```bash
python rag_comparison.py "Your question here"
```

### **Template Management:**
```python
from prompts import MedicalPromptManager

manager = MedicalPromptManager()
rag_prompt = manager.get_rag_prompt()
vanilla_prompt = manager.get_vanilla_prompt()
cleansing_prompt = manager.get_cleansing_prompt()
```

## ✅ **Key Improvements**

1. **Unified Template System** - All prompts managed consistently
2. **Single RAG Prompt** - Simplified from 4 prompts to 1
3. **Template-Based Cleansing** - No more hardcoded bias removal
4. **Simplified CLI** - Removed complex prompt selection options
5. **Backward Compatibility** - Existing code still works

## 🔧 **Integration Points**

- **RAG Service**: Uses `medical_rag` template
- **Comparison Script**: Uses all 3 template types
- **Main App**: Session state defaults to `medical_rag`
- **Template Loading**: Automatic from files and code registration