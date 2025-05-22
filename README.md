# Is prompt formatting mediated by a few directions?

Problem: prompt formatting (spacing, capitalization, using `;` instead of `,`) might significantly affect LLMs performance.

Importance: in practice one has to spend time and resources selecting the best prompt format, or settle for suboptimal performance.

Hypothesis: 
1) Information about prompt formatting might be represented in a low-dimensional linear subspace of latent space of an LLM.
2) If so, we can project project this information away and reduce the variations in performance in response to format changes.

## Data

Dataset: `madrylab/gsm8k-platinum` (this version has less labeling errors compared to `openai/gsm8k`).

## Models 

- `Qwen\Qwen2.5-1.5B-Instruct`
- `Qwen\Qwen2.5-3B-Instruct`

## Metrics

- Performance: Median accuracy over 10 formats 
- Robustness: Spread of accuracy over 10 formats (maximal accuracy - minimal accuracy)

## Format

We consider 3 variations of format
1. Capitalization of descriptors (e.g. lowercase, uppercase, regular).
2. Separator between descriptor and it's  (e.g. `: `, `\t`, `- `).
3. Space between question/reasoning/answer parts (e.g. `\n`, `-- `, ` || `).

We know this sounds confusing, so here is an example. Consider a fictional labeled dataset sample which consists of three parts:
- Question: "How much is 12 * 12?"
- Reasoning: "12 * 12 = 12 * 10 + 12 * 2 = 120 + 24 = 144"
- Answer: "144"

Format 1: lowercase, separator = `: `, space=`\n`
```
question: How much is 12 * 12?\n
reasoning: 12 * 12 = 12 * 10 + 12 * 2 = 120 + 24 = 144\n
answer: 144
```

Format 2: uppercase, separator = `\t`, space = `-- `
```
QUESTION\tHow much is 12 * 12?--
REASONING\t12 * 12 = 12 * 10 + 12 * 2 = 120 + 24 = 144--
ANSWER\t144
```

### Experiment design

1. Generate 10 formats varying three format components described above.
2. Evaluate model on `madrylab/gsm8k-platinum` test set with 10 formats, compute accuracy. We use few-shot with 2 examples selected from the train split of `openai/gsm8k`.
3. For each test sample `x` and each format `f`, also save the embedding `emb_{x, f}` of last token of the _question_ (before any reasoning or answer) from all layers of the model.
4. Select some layer. Identify the prompt format directions `v_i` by running PCA on a set of difference vectors, obtained by subtracting embeddings of same questions with different formatting (`emb_{x, f1} - emb_{x, f2}`). 
5. Ablate these directions from the model. That is, for each model weight `W` in attention or MLP layer which writes in the residual stream, modify the weight so that all format directions `v_i` belong to weight's kernel. In other words, `W v_i = 0`. Thus the outputs of model weights can not be affected by variation along formatting direction, and supposedly will be less sensitive (lower spread). However, a drop in quality may happen.
6. Re-evaluate model on 10 formats with ablated model to find out, whether the spread reduced, and what was the effect on accuracy.

