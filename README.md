# LLM Insight Evaluation Agent

A system that automatically evaluates the quality and accuracy of data-driven insights using AI. Built for clients who need privacy-focused solutions with open-source language models.

## What This Does

When someone makes a claim about data (like "The average item weight is 12.5 kg"), this system:

1. **Analyzes the actual data** to check if the claim is true
2. **Evaluates the claim quality** across 5 key dimensions
3. **Provides a detailed assessment** with scores and explanations

## Key Features

- **Privacy-Focused**: Uses local AI models instead of cloud services
- **Data-Driven Validation**: Actually analyzes datasets to verify claims
- **Comprehensive Evaluation**: Scores insights across multiple quality metrics
- **Extensible**: Can work with different datasets and evaluation criteria

## How It Works

The system evaluates claims across 5 dimensions:

1. **Correctness** - Is the claim factually accurate?
2. **Helpfulness** - Is it useful for decision-making?
3. **Complexity** - Does it show deep analysis?
4. **Coherence** - Is it well-structured and clear?
5. **Verbosity** - Is the level of detail appropriate?

## Example Output

**Input Claim**: "The average item weight in the Big Mart Sales dataset is around 12.5 kg"

**System Analysis**:
- ✅ **Data Analysis**: Found actual average = 12.86 kg (very close to claim)
- ✅ **Evaluation Results**:
  - Correctness: 3/5 (mostly accurate)
  - Helpfulness: 5/5 (very useful for business decisions)
  - Complexity: 1/5 (simple statistical fact)
  - Coherence: 5/5 (clear and well-structured)
  - Verbosity: 2/5 (appropriately concise)
- ✅ **Overall Score**: 3.2/5

## Technical Architecture

Built using:
- **LangGraph**: For workflow orchestration
- **Ollama**: For local AI model inference
- **E2B**: For secure code execution in sandboxed environments
- **Python**: Core implementation language

## Business Value

This system addresses a critical need in data-driven organizations:

- **Quality Control**: Ensures insights are accurate and valuable
- **Privacy Compliance**: Keeps sensitive data local
- **Scalability**: Can evaluate insights automatically
- **Transparency**: Provides detailed reasoning for each evaluation

## Use Cases

- **Business Intelligence**: Validating data insights before presentation
- **Research**: Quality control for data analysis reports
- **Consulting**: Ensuring client deliverables meet quality standards
- **Education**: Teaching data literacy and critical thinking

## Project Status

✅ **Core functionality complete**
- Data analysis integration working
- Multi-metric evaluation system operational
- Privacy-focused architecture implemented
- End-to-end workflow functional

🔄 **Next steps**
- Create comprehensive test dataset
- Optimize performance for production use
- Extend to additional datasets and domains

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ollama (for local AI models)
# Download from https://ollama.ai/

# Pull a model
ollama pull llama3.2:3b

# Run evaluation
python test_e2b_integration.py
```

## Files Overview

- `src/graph.py` - Main evaluation workflow
- `src/metric_evaluator.py` - Quality assessment logic
- `src/llm_client.py` - AI model integration
- `test_e2b_integration.py` - Main test script
- `data/train.csv` - Sample dataset (Big Mart Sales)

## Results

The system successfully:
- ✅ Analyzed real-world retail data
- ✅ Generated accurate statistical insights
- ✅ Evaluated claim quality across multiple dimensions
- ✅ Provided detailed, actionable feedback
- ✅ Maintained data privacy throughout the process

This demonstrates a working solution for automated insight quality assessment using modern AI techniques while respecting privacy and security requirements.
