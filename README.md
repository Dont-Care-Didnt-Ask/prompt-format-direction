# Is prompt formatting mediated by a single direction?

Problem: prompt formatting (spacing, capitalization, using ";" instead of ",") might significantly affect LLMs performance.
Importance: in practice one has to spend time and resources selecting the best prompt format, or settle for suboptimal performance.

Hypothesis: 
1) Information about prompt formatting might be represented in a low-dimensional linear subspace of latent space of an LLM.
2) If so, we can project project this information away and reduce the variations in performance in responce to format changes.

### Experiment design

Dataset: `madrylab/gsm8k-platinum` (this version has less labeling errors compared to `openai/gsm8k`).

Models: 
- `Qwen\Qwen2.5-1.5B-Instruct`
- `Qwen\Qwen2.5-3B-Instruct`

Metrics: 
- Performance: Median accuracy over 10 formats 
- Robustness: Spread of accuracy over 10 formats (maximal accuracy - minimal accuracy)

1. Generate 10 formats varying three format components: 
  1. Capitalization of descriptors (e.g. lowercase, uppercase, regular).
  2. Separator between descriptor and it's  (e.g. `: `, ` - `, ` || `).
  3. Space between question/reasoning/answer parts (e.g. `\n`, ` `, ` || `).
2. Evaluate model on `madrylab/gsm8k-platinum` test set with 10 formats, compute accuracy. We use 2-shots, with examples selected from train split of `openai/gsm8k`
3. For each test sample and each format, also save the embedding of last token of the question from åll layers of the model.
4. Select some layer. Identify the prompt format directions by running PCA on a set of difference vectors, obtained by subtracting embeddings of same questions with different formatting. 
5. Ablate these directions from the model. That is, for each model weight, modify the weight so that all format directions belong to weight's kernel. Thus the weights can not be affected by formatting, and supposedly will be less sensitive (although a drop in quality may happen).
6. Re-evaluate model on 10 formats with ablated model.

