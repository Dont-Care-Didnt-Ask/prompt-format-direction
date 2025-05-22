# Is prompt formatting mediated by a single direction?

Problem: prompt formatting (spacing, capitalization, using ";" instead of ",") might significantly affect LLMs performance.
Importance: in practice one has to spend time and resources selecting the best prompt format, or settle for suboptimal performance.

Hypothesis: 
1) Information about prompt formatting might be represented in a low-dimensional linear subspace of latent space of an LLM.
2) If so, we can project project this information away and reduce the variations in performance in responce to format changes.

### Experiment design

Dataset: GSM8K, with GSM8K-platinum test set (this version has less labeling errors).


1. Generate a set of formats varying three components: capitalization of descriptors ("Question" and "Answer")