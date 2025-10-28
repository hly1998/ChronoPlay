# ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks

This repository contains the implementation and data for **ChronoPlay**, a novel framework for automated and continuous generation of game RAG benchmarks.

## 🏆 Leaderboard

**View the leaderboard**: [https://hly1998.github.io/ChronoPlay/leaderboard/](https://hly1998.github.io/ChronoPlay/leaderboard/)

Submit your RAG system evaluation results and compare with other approaches! See [leaderboard/README.md](leaderboard/README.md) for submission instructions.

## 📖 Overview

Retrieval Augmented Generation (RAG) systems are increasingly vital in dynamic domains like online gaming, yet the lack of a dedicated benchmark has impeded standardized evaluation in this area. ChronoPlay addresses the core challenge of **Dual Dynamics**: the constant interplay between game content updates and the shifting focus of the player community.

📄 **Paper**: [ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks](https://arxiv.org/pdf/2510.18455)

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
├── leaderboard/                 # Interactive Leaderboard
│   ├── index.html              # Leaderboard webpage
│   ├── aggregate.py            # Data aggregation script
│   ├── leaderboard.json        # Aggregated results
│   ├── submissions/            # User submissions
│   └── README.md               # Submission guide
│
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
├── evaluation/                  # Leaderboard Evaluation Module
│   ├── retrieval_runner.py     # Execute retrieval for leaderboard
│   ├── retrieval_evaluator.py  # Evaluate retrieval on leaderboard data
│   ├── generation_runner.py    # Execute generation for leaderboard
│   ├── generation_evaluator.py # Evaluate generation on leaderboard data
│   ├── summarize_results.py    # Aggregate leaderboard metrics
│   └── components              # Evaluation components
│
├── experiments/                 # Paper Experiments (Dual-Dynamic Analysis)
│   ├── retrieval_runner.py     # Execute retrieval experiments
│   ├── retrieval_evaluator.py  # Evaluate retrieval experiments
│   ├── generation_runner.py    # Execute generation experiments
│   ├── generation_evaluator.py # Evaluate generation experiments
│   └── components              # Experiment components
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

**Download Options:**
- **Full Dataset** (with corpus): [Weiyun Cloud Drive](https://share.weiyun.com/HGXd33CW) (password: chrono)
  - For paper experiments and reproducing dual-dynamic analysis
  - Includes complete temporal corpus and all segments
- **Leaderboard QA Dataset**: [🤗 HuggingFace](https://huggingface.co/datasets/leoner24/ChronoPlay-QA)
  - For standard leaderboard evaluation and system comparison
  - Lightweight dataset with QA pairs only

After downloading the full dataset, extract and place the `data` folder under the `chronoplay/` directory.

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

#### 1. Leaderboard Evaluation (Recommended)

To evaluate your RAG system on the leaderboard dataset and submit results:

**👉 See [leaderboard/README.md](leaderboard/README.md) for complete instructions**

The leaderboard guide includes:
- Dataset download and setup
- Evaluation pipeline usage
- Metric calculations
- Submission format and process

#### 2. Paper Experiments (Dual-Dynamic Analysis)

Use the `experiments/` module to reproduce the dual-dynamic experiments from the paper:

```bash
cd experiments/

# Run full experimental pipeline
bash run_retrieval.sh
bash run_generation.sh
bash run_retrieval_evaluator.sh
bash run_generation_evaluator.sh
```

#### 3. Generating New QA Pairs

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

#### 4. Building Custom Corpus

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

Choose between two evaluation paths:

#### Option A: Leaderboard Evaluation (`evaluation/`)

For submitting to the official leaderboard and comparing with other systems.

**👉 See [leaderboard/README.md](leaderboard/README.md) for complete evaluation instructions**

#### Option B: Dual-Dynamic Experiments (`experiments/`)

For reproducing paper results and analyzing temporal dynamics:

```bash
cd experiments/

# Run complete experimental workflow
bash run_retrieval.sh
bash run_generation.sh
bash run_retrieval_evaluator.sh
bash run_generation_evaluator.sh
```

**Evaluation Metrics:**
- **Retrieval**: Recall@K, F1@K, MRR, NDCG
- **Generation**: Correctness (0-2), Faithfulness (0-2)

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

Paths are configured separately for each module:
- **Leaderboard Evaluation**: `evaluation/components/config.py`
- **Paper Experiments**: `experiments/components/config.py`

## 📈 Evaluation Results

**Leaderboard Evaluation** (`evaluation/`):
- See [leaderboard/README.md](leaderboard/README.md) for result locations and formats

**Paper Experiments** (`experiments/`):
- **Retrieval Results**: `experiments/retrieval_results/`
- **Generation Results**: `experiments/generation_results/`

## 🔬 Evaluation Modules

ChronoPlay provides two evaluation modules for different purposes:

### 📊 Leaderboard Evaluation (`evaluation/`)

**Purpose**: Standard benchmark evaluation for comparing RAG systems

**Use Case**: 
- Submit results to the official leaderboard
- Compare your system with others
- Evaluate on the standardized leaderboard dataset

**Dataset**: [HuggingFace leaderboard dataset](https://huggingface.co/datasets/leoner24/ChronoPlay-QA)

**📖 Documentation**: See [leaderboard/README.md](leaderboard/README.md) for detailed instructions

### 🔬 Paper Experiments (`experiments/`)

**Purpose**: Reproduce and analyze dual-dynamic phenomena from the paper

**Use Case**:
- Reproduce paper experiments
- Study temporal knowledge evolution
- Analyze user interest drift
- Investigate dual-dynamic challenges

**Dataset**: Full temporal segments from cloud drive

## 🎯 Research Applications

ChronoPlay can be used to:

1. **Benchmark RAG Systems**: Evaluate retrieval and generation quality in dynamic domains
2. **Study Temporal Dynamics**: Analyze how RAG performance changes over time
3. **Test Adaptation Strategies**: Evaluate how systems handle content updates
4. **Assess Player-Centricity**: Measure alignment with real player concerns

## 🏆 Submit to Leaderboard

To submit your RAG system results to the ChronoPlay leaderboard:

**👉 See [leaderboard/README.md](leaderboard/README.md) for complete submission guide**

The submission guide includes:
- How to run the evaluation pipeline
- Required metrics and calculation methods
- Submission file format and template
- Pull request submission process

View the current rankings: [ChronoPlay Leaderboard](https://hly1998.github.io/ChronoPlay/leaderboard/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Pull Requests for:
- RAG system evaluation results (see [leaderboard/README.md](leaderboard/README.md))
- Support for additional games
- New evaluation metrics
- Performance improvements
- Bug fixes

## 📚 Citation

If you use ChronoPlay in your research, please cite our paper:

```bibtex
@article{he2025chronoplay,
  title={ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks},
  author={He, Liyang and Zhang, Yuren and Zhu, Ziwei and Li, Zhenghui and Tong, Shiwei},
  journal={arXiv preprint arXiv:2510.18455},
  year={2025}
}
```


