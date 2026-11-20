# Sample Output Evaluation

## Purpose

This report records simple text statistics for one generated sample from each model. It is an illustrative comparison, not a formal benchmark.

## Sample measurements

| Metric | N-gram | DistilGPT-2 + LoRA |
|---|---:|---:|
| Word count | 3 | 35 |
| Unique words | 3 | 34 |
| Unique-word ratio | 100% | 97.1% |

## How the measurements were calculated

- **Word count:** Number of words in the recorded output.
- **Unique words:** Number of distinct words, ignoring capitalization.
- **Unique-word ratio:** Unique words divided by total words, multiplied by 100.

## Observations

- The N-gram model produced a short phrase: "the powerful shot".
- DistilGPT-2 + LoRA produced a longer, sentence-like passage.
- The longer output contains more varied vocabulary in this sample.
- The LoRA output also contains awkward phrasing, showing that fine-tuning does not guarantee grammatical or factually reliable text.

## Limitations

- Only one output per model is measured.
- The prompts differ between the two models.
- Word count and vocabulary diversity do not measure coherence, factual accuracy, or writing quality.
- The training corpus is a small, manually written sports-commentary sample.

These results should be treated as an example of model behavior, not evidence that either model is generally superior.
