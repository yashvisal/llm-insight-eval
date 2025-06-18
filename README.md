# LLM Insight Evaluation Agent

A LangGraph-based agentic system that evaluates the quality of natural language insights or claims generated from structured datasets. Built with privacy-focused open source LLMs via Ollama.

## Features

- **Multi-metric Evaluation**: Evaluates claims across 5 key metrics (Correctness, Helpfulness, Complexity, Coherence, Verbosity)
- **Data Analysis Integration**: Performs relevant statistical analysis to support claim evaluation
- **Parallel Processing**: Runs metric evaluations in parallel for improved performance
- **Privacy-Focused**: Uses local/open source LLMs via Ollama
- **Extensible Architecture**: Modular design for easy extension to new datasets and metrics
- **Validation System**: Built-in test data generation and validation framework

## Architecture

The system uses LangGraph to model the evaluation process as a state machine:

```
Start → Data Analysis → Evaluate Metrics (Parallel) → Output
```

### Components

- **Start Node**: Validates inputs and prepares for analysis
- **Data Analysis Node**: Performs relevant statistical analysis on the dataset
- **Evaluate Metrics Node**: Runs all 5 evaluation metrics in parallel
- **Output Node**: Aggregates results and formats final output

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd llm-insight-eval
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install and setup Ollama**:
```bash
# Install Ollama (https://ollama.ai/)
# Then pull a model:
ollama pull llama3.2
```

4. **Verify setup**:
```bash
python main.py --claim "Items with higher MRP tend to have higher sales."
```

## Usage

### Command Line Interface

**Evaluate a single claim**:
```bash
python main.py --claim "Items with higher MRP tend to have higher sales."
```

**With custom dataset summary**:
```bash
python main.py --claim "Your claim here" --dataset-summary "Your dataset description"
```

**Run validation tests**:
```bash
python main.py --validate --num-claims 20
```

**Generate test dataset**:
```bash
python main.py --generate-test-data
```

### Interactive Mode

Run without arguments for interactive mode:
```bash
python main.py
```

### Programmatic Usage

```python
from src.graph import EvaluationGraph
import asyncio

async def evaluate_claim():
    graph = EvaluationGraph()
    result = await graph.evaluate_claim(
        claim="Items with higher MRP tend to have higher sales.",
        dataset_summary="Big Mart Sales dataset with 8,523 records",
        task_description="Retail analytics for pricing optimization"
    )
    print(result)

asyncio.run(evaluate_claim())
```

## Configuration

Edit `src/config.py` to customize:

- **LLM Settings**: Model name, temperature, timeout
- **Dataset Settings**: Data path, description
- **Evaluation Settings**: Metrics, parallel processing, score ranges

### Environment Variables

Create a `.env` file for sensitive configuration:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

## Evaluation Metrics

1. **Correctness** (1-5): Factual alignment with data
2. **Helpfulness** (1-5): Usefulness and actionability
3. **Complexity** (1-5): Depth and novelty of reasoning
4. **Coherence** (1-5): Clarity and logical flow
5. **Verbosity** (1-5): Appropriate level of detail

## Output Format

```json
{
  "claim": "Items with higher MRP tend to have higher sales.",
  "scores": {
    "correctness": 5,
    "helpfulness": 4,
    "complexity": 3,
    "coherence": 5,
    "verbosity": 4
  },
  "explanations": {
    "correctness": "Strong positive correlation found in data analysis.",
    "helpfulness": "Useful for pricing optimization strategies.",
    "complexity": "Basic but correct statistical relationship.",
    "coherence": "Clear and well-structured claim.",
    "verbosity": "Appropriate level of detail."
  },
  "average_score": 4.2,
  "data_analysis_summary": "✓ statistical_analysis: Correlation analysis performed...",
  "timestamp": "2024-01-15T10:30:00",
  "execution_time": 12.5
}
```

## Project Structure

```
llm-insight-eval/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── models.py          # Data models and state
│   ├── llm_client.py      # LLM integration
│   ├── graph.py           # LangGraph workflow
│   └── test_data_generator.py  # Test data generation
├── prompts/
│   └── metrics.py         # Evaluation prompt templates
├── data/
│   └── train.csv          # Big Mart Sales dataset
├── main.py                # Main entry point
├── requirements.txt       # Dependencies
└── README.md
```

## Development

### Adding New Metrics

1. Add metric prompt to `prompts/metrics.py`
2. Update `src/models.py` EvaluationScores class
3. Add metric to configuration in `src/config.py`

### Extending to New Datasets

1. Update dataset configuration in `src/config.py`
2. Modify dataset summary in `src/graph.py`
3. Update test data generator patterns

### Custom LLM Integration

Replace `src/llm_client.py` with your preferred LLM client while maintaining the same interface.

## Testing

### Validation Framework

The system includes a comprehensive validation framework:

1. **Generate test data**:
```bash
python main.py --generate-test-data
```

2. **Run validation**:
```bash
python main.py --validate --num-claims 50
```

3. **Review results**:
- Check `validation_results.json` for detailed results
- Review accuracy metrics and individual test cases

### Test Data Categories

- **Correct Claims**: Factually accurate with expected scores 4-5
- **Partial Claims**: Partially correct with expected scores 2-3
- **Incorrect Claims**: Factually wrong with expected scores 1-2

## Performance

- **Typical evaluation time**: 10-30 seconds per claim
- **Parallel metric evaluation**: Reduces total time by ~60%
- **Data analysis overhead**: 5-15 seconds depending on claim complexity

## Troubleshooting

### Common Issues

1. **Ollama not running**:
   ```bash
   ollama serve
   ```

2. **Model not found**:
   ```bash
   ollama pull llama3.2
   ```

3. **Import errors**:
   ```bash
   pip install -r requirements.txt
   ```

### Logs

Check `logs/evaluation.log` for detailed execution logs.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here]

## Roadmap

- [ ] Web interface for easy interaction
- [ ] Support for more LLM providers
- [ ] Advanced statistical analysis capabilities
- [ ] Custom metric definitions
- [ ] Batch processing for multiple claims
- [ ] Integration with data visualization tools
