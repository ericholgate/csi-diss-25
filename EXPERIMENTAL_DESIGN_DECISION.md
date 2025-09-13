# Sequential Cross-Validation Training Decision

**Date**: 2025-01-15  
**Status**: Decided - Implement Sequential CV Training Paradigm

## Background

During implementation of killer prediction evaluation, we identified two fundamental approaches to cross-validation training for character embedding evaluation:

## Two Approaches Compared

### Current Implementation (Parallel CV)
- **Training**: Train character embeddings on ALL episodes using "Did I Say This" task
- **Evaluation**: Apply fixed classifiers (trained once on initial embeddings) to evolved embeddings
- **Interpretation**: Tests correlation between speaker-distinguishing embeddings and killer labels

### Sequential CV Training (New Decision)
- **Training**: For each fold, train embeddings only on training episodes (4/5 of data)
- **Evaluation**: Train killer classifier on learned embeddings, test on held-out fold embeddings
- **Interpretation**: Tests true generalization of character archetypes across episodes

## Key Decision Factors

### Research Question Alignment
**Core Question**: "Can the 'Did I Say This' paradigm learn character embeddings that capture generalizable character archetypes (like killer-ness)?"

- **Sequential approach directly tests this**: Do killer characters in unseen episodes share speaking patterns?
- **Parallel approach tests weaker claim**: Do embeddings correlate with killer labels (potential confounding)

### Theoretical Rigor
- **Sequential**: True generalization test with no data leakage between train/test
- **Sequential**: Tests archetype discovery across completely different characters (episode-isolated)
- **Sequential**: Tests character consistency across contexts (cross-episode)
- **Parallel**: Potential information leakage through shared training data

### Practical Implementation
- **Computational Cost**: 5x longer (5 separate training runs vs 1)
- **Code Complexity**: Manageable - orchestration layer changes, core logic identical
- **Statistical Power**: Acceptable trade-off (31 vs 39 episodes for training)

## Final Decision

**Implement Sequential CV Training as primary method**, with configurable option to use parallel approach.

### Implementation Strategy
```python
# Configuration flag
sequential_cv_training: bool = True  # Default to theoretically superior approach

# Experiment orchestration
if self.config.sequential_cv_training:
    return self._run_sequential_cv_experiment(dataset, model)
else:
    return self._run_standard_experiment(dataset, model)
```

### Sequential CV Process (Per Fold)
1. **Training Phase**: 
   - Create dataset from 4 training folds
   - Train character embeddings using "Did I Say This" task
   - Train killer classifier on learned embeddings + gold labels
2. **Test Phase**:
   - Create dataset from 1 test fold 
   - Train character embeddings using "Did I Say This" task
   - Apply pre-trained killer classifier to test embeddings
   - Record accuracy metrics

### Benefits for Both Character Modes
- **Episode-Isolated**: Tests killer archetype generalization across different individuals
- **Cross-Episode**: Tests character consistency + archetype discovery across contexts

## Implementation Priority
1. ✅ Document decision rationale (this file)
2. 🔄 Update CLAUDE.md with new experimental design
3. 🔲 Implement sequential training in ExperimentManager
4. 🔲 Add configuration support for both approaches
5. 🔲 Test with actual CSI data

## Validation Approach
- Implement both paradigms to enable comparative analysis
- Primary results use sequential approach for stronger scientific claims
- Parallel approach available for computational efficiency during development

---

**Conclusion**: Sequential CV training provides the theoretical rigor needed to make strong claims about character archetype learning while remaining computationally feasible and implementationally clean.