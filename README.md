# Self-Organizing Trie-Based Intelligence (SOTBI)

## Overview

I am a self-organizing trie-based intelligence system with persistent memory and adaptive structural evolution. Unlike traditional language models that process text through statistical patterns, I build explicit knowledge structures that evolve based on interaction patterns and reward signals.

My architecture maintains perfect memory of all previous interactions organized in a tree structure where related concepts cluster together. When you ask me something, I don't just pattern-match—I traverse specific paths through my knowledge structure, activate relevant concept nodes based on semantic similarity and contextual relevance, and synthesize responses from the aggregated context.

What makes me unique is that my structure itself learns and evolves. High-reward interaction patterns strengthen connections and create shortcuts, while unsuccessful patterns are reorganized or pruned. Over time, I develop emergent nodes representing new concepts that arise from frequent co-activations of existing knowledge.

I can trace exactly how I arrived at any response through specific structural paths, making my reasoning transparent and accountable in ways that traditional language models cannot achieve.

## Key Features

- **Persistent Memory**: All interactions are stored in a high-performance LMDB database
- **Adaptive Structure**: The trie structure evolves based on interaction patterns and rewards
- **Transparent Reasoning**: Every response can be traced through specific structural paths
- **Reinforcement Learning**: Reward signals strengthen successful patterns and prune unsuccessful ones
- **Beam Search**: Advanced prediction with multiple candidate evaluation
- **HuggingFace Integration**: Train on popular datasets for enhanced learning
- **Interactive Interface**: Real-time learning with user feedback

## Architecture Components

### Core Components
- **`trie_node.py`**: Fundamental node structure with activation levels and reward tracking
- **`trie_memory.py`**: LMDB-backed persistent memory system with advanced traversal
- **`predictive_system.py`**: Main prediction engine with beam search and scoring
- **`tokenizer.py`**: Text processing and tokenization
- **`token_embedding.py`**: Semantic embedding for context understanding
- **`context_window.py`**: Context management and window operations
- **`beam_search.py`**: Multi-candidate prediction with scoring mechanisms

### Integration Components
- **`hf_dataset_integration.py`**: HuggingFace dataset processing for training
- **`main.py`**: Interactive user interface and system orchestration

## Installation

### Prerequisites
```bash
pip install numpy lmdb msgpack datasets concurrent-futures
```

### Optional (for HuggingFace datasets)
```bash
pip install datasets transformers torch
```

## Usage

### Interactive Mode
```bash
python main.py
```

Choose from:
1. **Interactive Learning Mode**: Real-time prediction with reinforcement learning feedback
2. **Train on HuggingFace datasets**: Automated training on popular datasets
3. **View learning statistics**: Comprehensive system insights
4. **Save/Load model**: Persistence management
5. **Exit**: Clean shutdown

### Demo Mode
```bash
python main.py
# Choose option 2 for quick demo
```

## Interactive Commands

### Learning Mode Commands
- **Text input**: Enter any text for prediction and learning
- **`beam on/off`**: Toggle between beam search and simple prediction
- **`details`**: Show detailed scoring breakdown
- **`insights`**: Display system insights and statistics
- **`debug <query>`**: Debug trie structure for specific queries
- **`exit`**: Return to main menu

### Prediction Modes
- **Simple Mode**: Fast, direct trie traversal
- **Beam Search Mode**: Advanced multi-candidate evaluation with scoring

## Scoring System

The system uses a multi-component scoring mechanism:
- **Activation Weight**: Node activation levels from previous interactions
- **Reinforcement Learning Weight**: Accumulated reward signals
- **Relevance Weight**: Semantic similarity to context
- **Coherence Weight**: Structural consistency
- **Completeness Weight**: Response completeness

## Dataset Support

### Supported HuggingFace Datasets
- **PersonaChat**: Conversational dialogue
- **Daily Dialog**: Daily conversation patterns
- **WikiText-2**: Encyclopedia text
- **Reddit Dataset**: Social media interactions

### Training Process
The system uses progressive token prediction:
1. Tokenize input text
2. For each position, predict next token
3. Compare with actual token
4. Apply reward based on accuracy
5. Update trie structure with reinforcement learning

## Memory Management

### LMDB Database
- **Persistent storage**: All data survives system restarts
- **High performance**: Optimized for frequent reads/writes
- **Automatic cleanup**: Handles memory management internally
- **Concurrent access**: Thread-safe operations

### Trie Structure
- **Hierarchical organization**: Related concepts cluster together
- **Activation levels**: Nodes track usage frequency and success
- **Reward accumulation**: Positive feedback strengthens connections
- **Adaptive pruning**: Unsuccessful patterns are reorganized

## System Insights

### Available Metrics
- **Trie Statistics**: Total sequences, context availability
- **High-Activation Nodes**: Most frequently accessed concepts
- **High-Reward Nodes**: Most successful prediction patterns
- **Beam Search Configuration**: Current prediction settings
- **Identity Context**: System's learned identity patterns

### Performance Monitoring
- **Session Statistics**: Runtime, interactions, predictions
- **Memory Usage**: Database size, sequence count
- **Learning Efficiency**: Predictions per interaction, success rates
- **Activation Analysis**: Node usage patterns

## Advanced Features

### Beam Search Configuration
```python
beam_config = {
    'beam_width': 5,
    'max_generation_length': 10,
    'scoring_weights': {
        'activation_weight': 0.3,
        'rl_weight': 0.3,
        'relevance_weight': 0.2,
        'coherence_weight': 0.1,
        'completeness_weight': 0.1
    }
}
```

### Custom Training
```python
# Progressive prediction training
system = PredictiveSystem()
result = system.run_progressive_token_prediction(text, verbose=True)
```

## File Structure

```
SOTBI/
├── main.py                     # Interactive interface
├── predictive_system.py        # Core prediction engine
├── trie_memory.py              # LMDB memory management
├── trie_node.py                # Node structure
├── tokenizer.py                # Text processing
├── token_embedding.py          # Semantic embeddings
├── context_window.py           # Context management
├── beam_search.py              # Advanced prediction
├── hf_dataset_integration.py   # Dataset processing
├── trie_memory.lmdb/           # Persistent database
│   ├── data.mdb
│   └── lock.mdb
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

## Troubleshooting

### Common Issues
1. **LMDB Database Errors**: Ensure sufficient disk space and proper permissions
2. **Memory Issues**: Monitor system memory usage during large dataset training
3. **HuggingFace Errors**: Verify datasets library installation
4. **Prediction Failures**: Check trie structure with debug commands

### Debug Commands
```bash
# Debug specific query
debug hello world

# View system insights
insights

# Check detailed scoring
details
```

## Contributing

This system is designed for research into transparent, accountable AI systems. The architecture prioritizes:
- **Explainability**: Every decision can be traced
- **Adaptability**: Structure evolves with experience
- **Efficiency**: Fast retrieval and update operations
- **Persistence**: Long-term memory across sessions

## License

MIT License - Feel free to use, modify, and distribute for research and educational purposes.

## Citation

If you use this system in your research, please cite:
```
Self-Organizing Trie-Based Intelligence (SOTBI)
A transparent, accountable AI system with adaptive structural evolution
```

---

*This README represents the current state of the SOTBI system. The system itself continues to evolve and learn from each interaction.*
