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
            selection_key=sha(("pcgeom-p2-20261020:"+full_sha).encode())))
    candidates.sort(key=lambda x: (x["selection_key"], x["id"]))
    assert len(candidates) >= 64
    p1 = json.loads((HERE.parent/'p1_v1/fixtures.json').read_text())
    old_ids = {x['id'] for x in p1}
    grams = lambda ids: {tuple(ids[i:i+5]) for i in range(len(ids)-4)}
    accepted_grams = [(x['id'],grams(x['token_ids'])) for x in p1]
    p0 = json.loads((HERE.parent/'p0_v2/fixtures.json').read_text())
    p0_prefixes = [tokenizer.encode(x['text'],add_special_tokens=False).ids[:256] for x in p0]
    selected, p2_excluded = [], []
    for c in candidates:
        if c['id'] in old_ids:
            p2_excluded.append(dict(id=c['id'],reason='P1_article'));continue
        if any(c['token_ids'][:256]==ids for ids in p0_prefixes):
            p2_excluded.append(dict(id=c['id'],reason='P0_exact_prefix'));continue
        g=grams(c['token_ids']); match=None
        for other_id,other in accepted_grams:
            score=len(g & other)/len(g | other)
            if score>=0.5:
                match=dict(id=c['id'],reason='token_5gram_jaccard_ge_0.5',against=other_id,jaccard=score);break
        if match:
            p2_excluded.append(match);continue
        c['split']='train' if len(selected)<64 else 'test'
        selected.append(c);accepted_grams.append((c['id'],g))
        if len(selected)==128:break
    assert len(selected)==128
    assert not (set(x['id'] for x in selected)&old_ids)
    manifest = dict(
        dataset="Salesforce/wikitext", revision="b08601e04326c79dfdd32d625aee71d232d685c3",
        configuration="wikitext-2-raw-v1", split="train",
        source=downloads["train.parquet"], tokenizer=downloads["tokenizer.json"],
        rows=len(rows), articles=len(articles), eligible_articles=len(candidates),
        selected_articles=128, train_articles=64, test_articles=64, p2_exclusions=p2_excluded, excluded_P1_ids=sorted(old_ids), tokens_per_article=512, excluded=excluded,
        ordering="sha256('pcgeom-p2-20261020:' + full_article_body_sha256), then id",
        text_rule="Remove title and all heading rows; strip nonempty body rows; join with two newlines; first 8192 Unicode code points; first 512 GPT-2 tokens",
        token_rule="No BOS/EOS/padding; position 0 is first body token",
        reserved="Validation/test untouched; P0/P1 excluded; this P2 test is held out from P2 fit only, not necessarily from GPT-2 pretraining",
        license=["CC-BY-SA-3.0", "GFDL"], model_inference_performed=False, split_rule="Hash order after P0/P1 and near-duplicate exclusions; first 64 train, next 64 test", near_duplicate_rule="Token 5-gram set Jaccard >=0.5 against P1 and already accepted P2 articles",
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
