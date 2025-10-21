# ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks

This repository contains the implementation and data for **ChronoPlay**, a novel framework for automated and continuous generation of game RAG benchmarks.

## 📖 Overview

Retrieval Augmented Generation (RAG) systems are increasingly vital in dynamic domains like online gaming, yet the lack of a dedicated benchmark has impeded standardized evaluation in this area. ChronoPlay addresses the core challenge of **Dual Dynamics**: the constant interplay between game content updates and the shifting focus of the player community.

![ChronoPlayFramework](/images/demo.png)

### Key Features

- **Dual-Dynamic Update Mechanism**: Tracks both game content evolution and community focus shifts
- **Dual-Source Synthesis Engine**: Combines official documentation with authentic player community patterns
- **Player-Centric Authenticity**: Ensures generated questions reflect genuine player concerns
- **Temporal Segmentation**: Organizes knowledge by time periods to capture content evolution
- **Quality Assurance Pipeline**: Automated filtering and refinement for high-quality QA pairs

This is the first dynamic RAG benchmark for the gaming domain, offering insights into model performance under complex and realistic conditions.

## 🗂️ Repository Structure

```
chronoplay/
├── data/                        # Benchmark Data (download from cloud drive)
│   ├── dune/
│   │   ├── corpus/             # Temporal knowledge corpus (segments 1-6 + timeless)
│   │   ├── segments/           # Generated QA pairs for evaluation
│   │   │   ├── segment_1/generated_qa_pairs.jsonl
│   │   │   ├── segment_2/generated_qa_pairs.jsonl
│   │   │   └── ... (segments 3-6)
│   │   └── question_segments_results.json  # Temporal segmentation config
│   │
│   ├── dyinglight2/
│   │   ├── corpus/             # Temporal knowledge corpus (segments 1-5 + timeless)
│   │   ├── segments/           # Generated QA pairs for evaluation
│   │   │   ├── segment_1/generated_qa_pairs.jsonl
│   │   │   └── ... (segments 2-5)
│   │   └── question_segments_results.json
│   │
│   ├── pubgm/
│   │   ├── corpus/             # Temporal knowledge corpus (segments 1-7 + timeless)
│   │   ├── segments/           # Generated QA pairs for evaluation
│   │   │   ├── segment_1/generated_qa_pairs.jsonl
│   │   │   └── ... (segments 2-7)
│   │   └── question_segments_results.json
│   │
│   ├── question_templates.jsonl  # 12,000+ question templates
│   └── user_persona.jsonl      # Player role profiles
│
├── generation/                  # QA Generation Module
│   ├── generation.py           # Main generation system
│   ├── prompt.py               # Generation prompts
│   └── components              # Generation components
│
├── evaluation/                  # RAG Evaluation Module
│   ├── retrieval_runner.py     # Execute retrieval
│   ├── retrieval_evaluator.py  # Evaluate retrieval quality
│   ├── generation_runner.py    # Execute generation
│   ├── generation_evaluator.py # Evaluate generation quality
│   └── components              # Evaluation components
│
├── corpus/                      # Corpus Building Module
│   ├── corpus_builder.py       # Build temporal corpus
│   └── utils/
│       ├── ner_extractor.py    # Named entity recognition
│       └── txt_reader.py       # Document reader
│
└── global_vars/                 # Configuration
    ├── question_topics.json    # Question topic definitions
    └── question_type.json      # Question type definitions
```


## 📊 ChronoPlay Benchmark Dataset

### Dataset Overview

The benchmark includes three popular games with comprehensive temporal coverage:

| Game | Segments | QA Pairs | Time Span |
|------|----------|----------|-----------|
| Dune: Awakening | 6 | 3,000 | Jun 25 - Aug 25 |
| Dying Light 2 | 5 | 2,000 | Jan 22 - Jul 25 |
| PUBG Mobile | 7 | 1,400 | Jan 24 - Jul 25 |

**Download**: All data is available via cloud drive: https://share.weiyun.com/HGXd33CW (password: chrono)

After downloading, extract and place the `data` folder under the `chronoplay/` directory.

### Data Components

**1. Temporal Knowledge Corpus** (`data/{game}/corpus/`)
- Segmented by time periods capturing game evolution
- Includes both temporal and timeless content
- Enriched with named entity annotations

**2. Generated QA Pairs** (`data/{game}/segments/segment_*/`)
- High-quality synthetic QA pairs for each temporal segment
- Generated using dual-source synthesis (official docs + player patterns)
- Includes ground truth answers and reference contexts
- Ready for RAG evaluation

**3. Question Templates** (`data/question_templates.jsonl`)
- 12,000+ curated question templates
- Covers diverse topics: gameplay, bugs, features, performance, etc.
- Extracted from real player questions

**4. Player Personas** (`data/user_persona.jsonl`)
- Player role profiles reflecting different player types
- Enables authentic question generation aligned with player concerns

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chronoplay.git
cd chronoplay

# Install dependencies
pip install -r requirements.txt

# Download dataset from cloud drive
# Download link: [to be provided]
# Extract to the data/ directory
```

### Basic Usage

#### 1. Using Pre-generated Benchmark (Recommended)

We provide pre-generated QA pairs in `data/{game}/segments/` ready for evaluation:

```bash
# After downloading data to data/ directory
# Run evaluation directly:

cd evaluation/

# Run retrieval
python retrieval_runner.py --game dune --segment_id 1

# Run generation
python generation_runner.py --retrieval_file retrieval_results/retrieval_*.jsonl

# Evaluate results
python retrieval_evaluator.py --game dune --segment_id 1
python generation_evaluator.py --generation_results generation_results/generation_*.jsonl
```

#### 2. Generating New QA Pairs

To generate QA pairs for your own data or extend to new games:

```bash
cd generation/

python generation.py \
    --game_name dune \
    --segment_id 1 \
    --api_key {your-api-key} \
    --base_url {your-base-url} \
    --model_name gpt-4o \
    --target_sample_size 150 \
    --enable_qa_filtering \
    --enable_qa_refining
```

**Generation Features:**
- **Template Sampling**: Intelligently samples based on temporal patterns
- **Role Matching**: Matches appropriate player roles to question types
- **Quality Filtering**: Automated quality assessment
- **QA Refinement**: LLM-based optimization
- **QA Inheritance**: Reuses valid QA pairs across segments

#### 3. Building Custom Corpus

To build corpus from your own game documentation:

```bash
cd corpus/

python corpus_builder.py \
    --segments_file data/dune/question_segments_results.json \
    --data_dir data/dune/knowledge \
    --output_dir data/dune/corpus \
    --openai_api_key {your-api-key} \
    --openai_base_url {your-base-url}
```

## 📋 Complete Workflow

### Stage 1: Corpus Building (Optional - Data Provided)

Build temporal knowledge corpus from raw documents:

```bash
cd corpus/
python corpus_builder.py --game_name dune --enable_ner
```

### Stage 2: QA Generation

Generate synthetic QA pairs using dual-source synthesis:

```bash
cd generation/

# Generate for single segment
python generation.py --game_name dune --segment_id 1

# Batch generate for all segments
for segment in {1..6}; do
    python generation.py --game_name dune --segment_id $segment
done
```

**Generation Parameters:**
- `--target_sample_size`: Target QA pairs per segment (default: 150)
- `--similarity_threshold`: Document relevance threshold (default: 0.5)
- `--enable_qa_filtering`: Enable quality filtering
- `--enable_qa_refining`: Enable LLM-based refinement
- `--enable_qa_inheritance`: Enable QA reuse across segments

### Stage 3: RAG Pipeline Evaluation

#### Step 3.1: Execute RAG Pipeline

**Retrieval:**
```bash
cd evaluation/

# Vector retrieval
python retrieval_runner.py \
    --game dune \
    --segment_id 1 \
    --retrieval_method vector \
    --top_k 5

# BM25 retrieval
python retrieval_runner.py \
    --game dune \
    --segment_id 1 \
    --retrieval_method bm25 \
    --top_k 5
```

**Generation:**
```bash
python generation_runner.py \
    --retrieval_file retrieval_results/retrieval_dune_segment_1.jsonl \
    --llm_model gpt-4o \
    --temperature 0.1
```

#### Step 3.2: Evaluate Results

**Retrieval Evaluation:**
```bash
python retrieval_evaluator.py \
    --game dune \
    --segment_id 1 \
    --results_dir retrieval_results/
```

Metrics: Recall@K, MRR, NDCG

**Generation Evaluation:**
```bash
python generation_evaluator.py \
    --generation_results generation_results/generation_*.jsonl \
    --api_key {your-api-key} \
    --metrics correctness faithfulness
```

Metrics:
- **Correctness** (0-2): Factual accuracy against ground truth
- **Faithfulness** (0-2): Consistency with retrieved context

## ⚙️ Configuration

### API Configuration

All API credentials are passed as command-line arguments:

```bash
python generation/generation.py \
    --api_key {your-api-key} \
    --base_url {your-base-url} \
    --model_name gpt-4o
```

**Required Parameters:**
- `--api_key`: Your API key for the LLM service
- `--base_url`: Base URL of your API endpoint

### Model Configuration

ChronoPlay supports any OpenAI-compatible API. You can use:
- OpenAI models (gpt-4o, gpt-4-turbo, gpt-3.5-turbo, etc.)
- Azure OpenAI
- Other OpenAI-compatible providers

Simply provide the correct `--api_key`, `--base_url`, and `--model_name` for your chosen service.

### Path Configuration

All paths use relative paths from project root (configured in `evaluation/components/config.py`).

## 📈 Evaluation Results

Results will be stored in:
- **Retrieval Results**: `evaluation/retrieval_results/`
- **Generation Results**: `evaluation/generation_results/`
- **Evaluation Reports**: `evaluation/retrieval_evaluation/` and `evaluation/generation_evaluation/`

## 🔬 Research Applications

ChronoPlay can be used to:

1. **Benchmark RAG Systems**: Evaluate retrieval and generation quality in dynamic domains
2. **Study Temporal Dynamics**: Analyze how RAG performance changes over time
3. **Test Adaptation Strategies**: Evaluate how systems handle content updates
4. **Assess Player-Centricity**: Measure alignment with real player concerns

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Pull Requests for:
- Support for additional games
- New evaluation metrics
- Performance improvements
- Bug fixes

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: Replace `{your-api-key}` and `{your-base-url}` with your actual API credentials when running the scripts.
