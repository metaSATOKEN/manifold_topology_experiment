P3 input excerpts are derived from Salesforce/WikiText (WikiText-2 raw v1), English Wikipedia.
Dataset card: https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md
License metadata: CC-BY-SA-3.0 and GFDL.
https://creativecommons.org/licenses/by-sa/3.0/
https://www.gnu.org/licenses/fdl-1.3.html
Article titles, source row identifiers and full-body hashes are in input_manifest.json. Wikipedia article URLs are https://en.wikipedia.org/wiki/ followed by each title with spaces replaced by underscores. The source revision denotes the dataset snapshot, not current Wikipedia content.
Modifications: removed headings, stripped and joined paragraphs, retained leading text and first 512 GPT-2 tokens.
The text excerpts retain their source licenses; these notices must accompany redistribution.

P3 uses 64 evaluation-only articles from the dataset train split, excluding all 192 P1/P2 articles, P0 exact prefixes and near duplicates as specified in input_manifest.json. No new fitting is performed: the P2/EDA-0001 readout coefficients, PCA/random bases and training frequencies are embedded unchanged. Evaluation articles are unused by this study's model observations and readout adjustment, not necessarily unseen in GPT-2 pretraining.
