# Is prompt formatting mediated by a single direction?

Problem: prompt formatting (spacing, capitalization, using ";" instead of ",") might significantly affect LLMs performance.
Importance: in practice one has to spend time and resources selecting the best prompt format, or settle for suboptimal performance.

Hypothesis: 
1) Information about prompt formatting might be represented in a low-dimensional linear subspace of latent space of an LLM.
2) If so, we can project project this information away and reduce the variations in performance in responce to format changes.

### Experiment design

Dataset: GSM8K, with GSM8K-platinum test set (this version has less labeling errors).


1. Generate 10 formats varying three components: 
  1. Capitalization of descriptors (e.g. "Question"/"question", "Answer"/"ANSWER").
  2. Separator between descriptor.
  3. Space between question and answer.
2. Evaluate model on GSM8K-platinum test set with 10 formats, compute accuracy.
3. For each test sample and each format, also save the embedding of last token of the question from middle layer of the model.
4. Identify the prompt format directions with PCA. Abliterate them from the model.
5. Re-evaluate model on 10 formats with abliterated model.