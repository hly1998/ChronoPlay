# ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2510.18455-b31b1b.svg)](https://arxiv.org/pdf/2510.18455)
[![Dataset](https://img.shields.io/badge/🤗-LeaderbordDataset-FFD21E)](https://huggingface.co/datasets/leoner24/ChronoPlay-QA)
[![Leaderboard](https://img.shields.io/badge/🏆-Leaderboard-006EFF)](https://hly1998.github.io/ChronoPlay/)
[![GitHub](https://img.shields.io/github/stars/hly1998/ChronoPlay?style=social)](https://github.com/hly1998/ChronoPlay)

</div>

This repository contains the implementation and data for **ChronoPlay**, a novel framework for automated and continuous generation of game RAG benchmarks.

**Resources:**

- **Paper**: [ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks](https://arxiv.org/pdf/2510.18455)
- **Dataset**: [HuggingFace - ChronoPlay-QA (Leaderboard)](https://huggingface.co/datasets/leoner24/ChronoPlay-QA)
- **Leaderboard**: [Submit your results and compare with others](https://hly1998.github.io/ChronoPlay/)
- **Full Dataset**: [Weiyun Cloud Drive (with corpus)](https://share.weiyun.com/HGXd33CW) (password: chrono)

> 💡 **Submit to Leaderboard**: See [leaderboard/README.md](leaderboard/README.md) for submission instructions.

## 📖 Overview

Retrieval Augmented Generation (RAG) systems are increasingly vital in dynamic domains like online gaming, yet the lack of a dedicated benchmark has impeded standardized evaluation in this area. ChronoPlay addresses the core challenge of **Dual Dynamics**: the constant interplay between game content updates and the shifting focus of the player community.

![ChronoPlayFramework](/images/demo.png)

### Key Features

- **Dual-Dynamic Update Mechanism**: Tracks both game content evolution and community focus shifts
- **Dual-Source Synthesis Engine**: Combines official documentation with authentic player community patterns
- **Player-Centric Authenticity**: Ensures generated questions reflect genuine player concerns
- **Quality Assurance Pipeline**: Automated filtering and refinement for high-quality QA pairs

This is the first dynamic RAG benchmark for the gaming domain, offering insights into model performance under complex and realistic conditions.

## 🗂️ Repository Structure

```
chronoplay/
├── leaderboard/                 # Interactive Leaderboard
│
├── data/                        # Benchmark Data (download from cloud drive)
│   ├── {game name}/
│   │   ├── corpus/             # Temporal knowledge corpus (segments 1-6 + timeless)
│   │   ├── segments/           # Generated QA pairs for evaluation
│   │   │   ├── segment_1/generated_qa_pairs.jsonl
│   │   │   ├── segment_2/generated_qa_pairs.jsonl
│   │   │   └── ... (segments 3-6)
│   │   └── question_segments_results.json  # Temporal segmentation config
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
- **Full Dataset** [Weiyun Cloud Drive](https://share.weiyun.com/HGXd33CW) (password: chrono): For paper experiments and leaderboard evaluation datasets
- **Leaderboard QA Dataset** [HuggingFace](https://huggingface.co/datasets/leoner24/ChronoPlay-QA): For leaderboard evaluation datasets only

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


## 📋 Complete Workflow

### Stage 1: Corpus Building (Optional - Data Provided)

Build temporal knowledge corpus from raw documents:

```bash
cd corpus/
python corpus_builder.py --game_name dune
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

### Stage 3: RAG Pipeline Evaluation

Choose between two evaluation paths:

#### Option A: Leaderboard Evaluation (`evaluation/`)

For submitting to the official leaderboard and comparing with other systems.

**👉 See [leaderboard/README.md](leaderboard/README.md) for complete evaluation instructions**

#### Option B: Dual-Dynamic Experiments (`experiments/`)

For reproducing paper results and analyzing temporal dynamics.

**Evaluation Metrics:**

- **Retrieval**: Recall@K, F1@K, MRR, NDCG
- **Generation**: Correctness (0-2), Faithfulness (0-2)



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



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

