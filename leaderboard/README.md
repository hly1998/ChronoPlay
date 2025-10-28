# 🎮 ChronoPlay RAG Leaderboard

An interactive leaderboard for evaluating RAG systems on the ChronoPlay benchmark across three popular games.

## 📊 View the Leaderboard

Visit: [https://hly1998.github.io/ChronoPlay/leaderboard/](https://hly1998.github.io/ChronoPlay/leaderboard/)

## 🚀 Quick Submit

### Submit Your Results via Pull Request

1. **Fork this repository**

2. **Create submission file in `leaderboard/submissions/`**
   ```json
   {
     "system_name": "My RAG System v1.0",
     "description": "Dense retrieval with BM25 reranking and GPT-4 for generation",
     "games": {
       "dune": {
         "topk": 3,
         "recall": 0.85,
         "f1": 0.78,
         "ndcg": 0.82,
         "correctness": 0.88,
         "faithfulness": 0.91
       },
       "dying_light_2": {
         "topk": 3,
         "recall": 0.83,
         "f1": 0.76,
         "ndcg": 0.80,
         "correctness": 0.86,
         "faithfulness": 0.89
       },
       "pubg_mobile": {
         "topk": 3,
         "recall": 0.81,
         "f1": 0.74,
         "ndcg": 0.78,
         "correctness": 0.84,
         "faithfulness": 0.87
       }
     }
   }
   ```

3. **Submit Pull Request to the main repository**

4. **After merge, leaderboard auto-updates via GitHub Actions**

## 📏 Evaluation Metrics

### Base Metrics
- **Top-K**: Number of documents retrieved
- **Recall**: Proportion of relevant documents in top-k results (0-1)
- **F1**: Harmonic mean of precision and recall (0-1)
- **NDCG**: Normalized Discounted Cumulative Gain (0-1)
- **Correctness**: Accuracy of generated answers (0-1)
- **Faithfulness**: Consistency with retrieved documents (0-1)

### Computed Scores
- **R (Retrieval Score)** = (Recall + F1 + NDCG) / 3
- **G (Generation Score)** = Correctness × 0.75 + Faithfulness × 0.25
- **Total Score** = R × 0.25 + G × 0.75

All scores are displayed as percentages (0-100).

## 🔧 Local Testing

```bash
cd leaderboard

# Test aggregation
python aggregate.py

# View with local server
python3 -m http.server 8000
# Visit http://localhost:8000
```

## 📚 Citation

If you use this leaderboard, please cite our paper:

```bibtex
@article{he2025chronoplay,
  title={ChronoPlay: A Framework for Modeling Dual Dynamics and Authenticity in Game RAG Benchmarks},
  author={He, Liyang and Zhang, Yuren and Zhu, Ziwei and Li, Zhenghui and Tong, Shiwei},
  journal={arXiv preprint arXiv:2510.18455},
  year={2025}
}
```

## 📄 License

MIT License

