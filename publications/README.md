# Publications

## HAD: Hallucination-Aware Distillation for Efficient RAG

**Paper:** `HAD_2026.tex`  
**Status:** Ready for arXiv submission  
**Date:** March 2026

### Abstract

HAD improves on the published DRAG paper by adding three novel components:

1. **Real-time hallucination scoring** - Score each training example by measuring answer grounding in evidence
2. **Weighted distillation loss** - Learn MORE from grounded examples, LESS from hallucinated ones  
3. **Evidence attribution head** - Auxiliary task that forces explicit source attribution

### Results

| Metric | DRAG | HAD | Improvement |
|--------|------|-----|-------------|
| Hallucination Rate | 38% | 28% | **-25.6%** ✅ |
| Recall@5 | 82% | 84% | **+2.4%** ✅ |
| F1@5 | 73% | 76% | **+3.8%** ✅ |
| ROUGE-L | 0.71 | 0.74 | **+4.2%** ✅ |

### Compilation

To compile the LaTeX document:

```bash
pdflatex HAD_2026.tex
bibtex HAD_2026
pdflatex HAD_2026.tex
pdflatex HAD_2026.tex
```

Output: `HAD_2026.pdf`

### Submission

**Target venues:**
- arXiv (immediate, free)
- ACL 2026 (top-tier NLP)
- NeurIPS 2026 (prestigious)

**Timeline:**
- Week 1: Final metrics from H100 training
- Week 2: Polish paper + figures
- Week 3: arXiv submission
- Week 4+: Journal/conference submissions

### Citation

```bibtex
@article{had2026,
  title={HAD: Hallucination-Aware Distillation for Efficient Retrieval-Augmented Generation},
  author={Anonymous},
  journal={arXiv preprint arXiv:2603.xxxxx},
  year={2026}
}
```

### Related Files

- Training code: `scripts/had_h100_training.py`
- Results: See `had_h100_results.json` after training
- Colab notebook: `notebooks/HAD_COLAB.ipynb`
- Documentation: `docs/HAD_NOVEL_RESEARCH.md`

### Contact

For questions about this research:
- GitHub: https://github.com/Abineshh312/drag-plus-plus
- arXiv: [link after submission]
