"""Build frozen inputs only. No model weights, inference, or outcome selection."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "design_inputs/p1_v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build():
    downloads = json.loads((SOURCE / "downloads.json").read_text())
    for name, entry in downloads.items():
        assert sha((SOURCE/name).read_bytes()) == entry["sha256"]
    tokenizer = Tokenizer.from_file(str(SOURCE/"tokenizer.json"))
    rows = pq.read_table(SOURCE/"train.parquet", columns=["text"])["text"].to_pylist()
    articles = []
    current = None
    # Top-level heading: exactly one equals sign on each side (not section headings).
    for index, row in enumerate(rows):
        heading = re.fullmatch(r"\s*=\s+([^=].*?)\s+=\s*", row)
        if heading and not heading.group(1).strip().endswith("="):
            if current is not None:
                articles.append(current)
            current = dict(title=heading.group(1).strip(), row_start=index, rows=[])
        elif current is not None and row.strip() and not row.strip().startswith("="):
            current["rows"].append(row.strip())
    if current is not None:
        articles.append(current)
    candidates, excluded, seen_titles, seen_prefixes = [], [], set(), set()
    for article in articles:
        full = "\n\n".join(article["rows"])
        text = full[:8192]
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:512]
        title_key = " ".join(article["title"].casefold().split())
        token_sha = sha(np.array(ids, dtype="<i8").tobytes())
        reason = ("shorter_than_512" if len(ids) < 512 else
                  "duplicate_title" if title_key in seen_titles else
                  "duplicate_token_prefix" if token_sha in seen_prefixes else None)
        if reason:
            excluded.append(dict(title=article["title"], row_start=article["row_start"], reason=reason))
            continue
        seen_titles.add(title_key)
        seen_prefixes.add(token_sha)
        full_sha = sha(full.encode())
        candidates.append(dict(
            id=f"wt2_train_{article['row_start']:05d}", title=article["title"],
            row_start=article["row_start"], text=text, text_sha256=sha(text.encode()),
            full_article_body_sha256=full_sha, token_ids=ids, input_ids_sha256=token_sha,
            selection_key=sha(("pcgeom-p1-20261020:"+full_sha).encode())))
    candidates.sort(key=lambda x: (x["selection_key"], x["id"]))
    assert len(candidates) >= 64
    selected = candidates[:64]
    manifest = dict(
        dataset="Salesforce/wikitext", revision="b08601e04326c79dfdd32d625aee71d232d685c3",
        configuration="wikitext-2-raw-v1", split="train",
        source=downloads["train.parquet"], tokenizer=downloads["tokenizer.json"],
        rows=len(rows), articles=len(articles), eligible_articles=len(candidates),
        selected_articles=64, tokens_per_article=512, excluded=excluded,
        ordering="sha256('pcgeom-p1-20261020:' + full_article_body_sha256), then id",
        text_rule="Remove title and all heading rows; strip nonempty body rows; join with two newlines; first 8192 Unicode code points; first 512 GPT-2 tokens",
        token_rule="No BOS/EOS/padding; position 0 is first body token",
        reserved="Validation/test untouched; P1 inputs cannot be P2 held-out evaluation",
        license=["CC-BY-SA-3.0", "GFDL"], model_inference_performed=False,
        selected=[{k:v for k,v in d.items() if k not in ("text","token_ids")} for d in selected])
    for name, data in (("fixtures.json", selected), ("input_manifest.json", manifest)):
        output = HERE/name
        content = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)+"\n"
        if output.exists():
            assert output.read_text() == content, "Do not replace frozen inputs"
        else:
            output.write_text(content)
    print(json.dumps({k:manifest[k] for k in ("rows","articles","eligible_articles","selected_articles")}))


if __name__ == "__main__":
    build()
