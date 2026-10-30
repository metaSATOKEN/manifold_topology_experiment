"""Append the EDA-0003 outcome; preserve original evidence and old ledger prefix."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE.parents[1]
before = json.loads((OUT / "preservation_before.json").read_text())
ops = before["operations"]
handoff = "2026-10-20_" + "z" * 14 + "_eda3_results.md"
anchor = "eda3_final_checks_2026-10-20.json"
card = "EDA-0003_v1_p4_components_radius_direction.md"
result = "EDA-0003_v1_results_2026-10-20.md"
state = """**最新：EDA-0003の保存P4探索１セットを完了し、探索も終了（台帳§19）。**
PC1/PC2〜8、長さ/向き、記事/token偏りを全６条件で解析。位置・ノルム考慮後の追加関連はPC2〜8に多く残り、
柔軟な長さ対照後も向きからの増分中央値は全６点で正の記述95%区間。ただしOPT中央層では向きPC対ランダムの差は不明瞭。
Qwen中央層はraw8増分中央値+0.07077に対し記事平均−0.31019、１記事で−24.22476の失敗を保持。
探索でありP4の４支持/２不確定は不変。因果機序は未同定。条件追加・第２探索・新規取得・介入へ自動延長しない。
旧原本と旧結果を保持、新LM取得/forward/GPU0、原稿未変更。"""
state = state.replace("\n", "")
links = (
    "[日本語レポート・４図](exploratory/p4_components_v1/ANALYSIS_REPORT_JA.md)、"
    "[全条件表](exploratory/p4_components_v1/DETAIL_TABLES.md)、"
    f"[固定条件](prereg/{card})、[結果追補](prereg/{result})、"
    f"[最新引き継ぎ](HANDOFF/{handoff})、[現行照合](evidence/{anchor})。"
)
for name in ["README.md", "PLAN.md"]:
    path = BASE / name
    original = ops[str(path.relative_to(ROOT))]
    assert path.read_text() == original
    header, rest = original.split("\n", 1)
    if name == "PLAN.md":
        assert "v0.16" in header
        header = header.replace("v0.16", "v0.17")
    path.write_text(header + "\n\n" + state + "\n\n" + links + "\n\n以下は過去時点の記録。\n" + rest)

path = BASE / "REQUIREMENTS.md"
original = ops[str(path.relative_to(ROOT))]
assert path.read_text() == original
updated = original.replace("## 1. 成果物と一次証拠\n", "## 1. 成果物と一次証拠\n\n" + state +
                          f"\n現行アンカー `evidence/{anchor}`、結果 `exploratory/p4_components_v1/ANALYSIS_REPORT_JA.md`。\n", 1)
row = "| EDA-0003保存P4最終探索 | PC分割・長さ/向き・記事/token偏り・原本照合 | 条件先行固定、全６点・384記事点、16,355,394予測値、13,824指標、39,936区分寄与、888集計照合PASS。４図・全条件表、自己監査。大失敗記事も保持、P4確認判定不変、探索終了 |\n"
updated = updated.replace("| 面 | 検査・担当 | 現在の状態 |\n| --- | --- | --- |\n",
                          "| 面 | 検査・担当 | 現在の状態 |\n| --- | --- | --- |\n" + row, 1)
path.write_text(updated)

path = BASE / "prereg/README.md"
original = ops[str(path.relative_to(ROOT))]
assert path.read_text() == original
header, rest = original.split("\n", 1)
path.write_text(header + "\n\n" +
    f"最新：[EDA-0003固定条件]({card}) / [結果追補]({result}) "
    "COMPLETED / EXPLORATORY / FINAL_SAVED_DATA_SET_CLOSED（台帳§19）。"
    "既知P4の保存状態だけでPC1/PC2〜8と長さ/向きを分解。全６点、記事/token偏りとQwen中央層の大失敗も保存。"
    "P4確認的判定は４支持/２不確定のまま。新規実験も第２探索も自動延長しない。"
    "[レポート](../exploratory/p4_components_v1/ANALYSIS_REPORT_JA.md)。以下は過去時点の記録。\n" + rest)

path = ROOT / "AGENTS.md"
text = path.read_text()
start = text.index("**今回進行中：EDA-0003")
end = text.index("\n\n", start)
replacement = state + (
    "\n入口 `research/pc_geometry_rerun/exploratory/p4_components_v1/ANALYSIS_REPORT_JA.md`。"
    f"最新引き継ぎ `research/pc_geometry_rerun/HANDOFF/{handoff}`、"
    f"現行アンカー `research/pc_geometry_rerun/evidence/{anchor}`。"
    "preserved_sha256/artifact_sha256を実物照合して再開する。以下は以前の作業時点の記録。"
)
path.write_text(text[:start] + replacement + text[end:])

addendum = """# EDA-0003 v1 結果追補

2026-10-20。COMPLETED / EXPLORATORY / FINAL_SAVED_DATA_SET_CLOSED。
原カードは変更せず結果を別追補とする。P4結果既知、ローカル条件先行固定であり、確認的事前登録ではない。

## 実施と主要所見

保存P4の３モデル×中央/最終正規化後、全６点、訓練64・既知評価64記事。C1・記事×ID群内中心化を維持。
PC1/PC2〜8/全8、同次元既存20ランダム、位置・線形ノルム・５ヒンジ、単位長PCA8を固定して比較した。
数値94.41秒、集計検証44.13秒、表示用の寄与総和集計約１秒。新LM取得・forward・GPU0。

- PC2〜8の追加増分と７次元ランダムとの差は全６点で正の記述95%区間。PC1単独は条件依存で、OPT中央層は増分/ランダム差とも正。PC1が無情報という結果ではない。
- 柔軟な長さ対照後の向きNUの増分中央値は全６点で正。PC対ランダムの差はOPT中央層で0.000161 [−0.006501, 0.003788]と不明瞭。他５点は正の記述区間。
- 長さ対照の強化でraw8増分はGPT-2最終・OPT中央で縮むが、一律でない。rawとunitの基底・相互作用の違いがあり、純粋な長さの因果寄与と解釈しない。
- Qwen中央層のraw8増分中央値+0.070771、記事平均−0.310188 [−1.074623, 0.075817]。`wt2_train_27981`の増分−24.224762を除外せず保持。NUでもこの記事は−0.265742、ID等重みでもraw8の大失敗は残る。

全６点、全18主題量・等ID重み・全20ランダム・全区分・384記事点を保存した。上記は抜粋であり、全文表が報告の分母。
95%区間は多重比較未調整の探索的記述。P4の99.583333%区間と最小効果裁定４支持/２不確定は変更しない。

## 検証と終了

元P4 W全予測の最大差5.48e−11。新系列を含む16,355,394予測値、13,824指標、
39,936区分寄与、840集計＋寄与総和48集計を別計算経路で照合。
基底・群内中心化・拡大行列解・勾配・CV選択・保存係数再読込・区分加法性を検査した。
同じagentの自己監査であり独立レビューではない。

固定条件・終了点を変更せず完了。因果機序は未解決として残し、この１セットで探索も終了。
新規モデル・層・記事・乱数・介入・第２探索を自動追加しない。原稿変更・主張採用・公開は未実施。
原本保持と容量上限は最終アンカーで検算。台帳§19、FP1/FN0/対照1/未裁定0、TL正式評価0は不変。

[日本語レポート](../exploratory/p4_components_v1/ANALYSIS_REPORT_JA.md)、
[全条件表](../exploratory/p4_components_v1/DETAIL_TABLES.md)、
[固定記録](../exploratory/p4_components_v1/plan_freeze.json)、
[最終アンカー](../evidence/eda3_final_checks_2026-10-20.json)。
"""
# Use source-derived rounding for the explicitly reported mean interval.
d = json.loads((OUT / "report_data.json").read_text())["results"]["qwen/middle"]["B_to_A8"]
addendum = addendum.replace("−0.310188 [−1.074623, 0.075817]",
                            f'{d["estimate"]:.6f} [{d["lo"]:.6f}, {d["hi"]:.6f}]')
(BASE / "prereg" / result).write_text(addendum)

ledger = BASE / "audits/ledger_2026-10.md"
assert ledger.read_text().startswith(ops[str(ledger.relative_to(ROOT))])
with ledger.open("a") as f:
    f.write("""

### 実施結果・検証・終了

状態：COMPLETED / EXPLORATORY / FINAL_SAVED_DATA_SET_CLOSED。
固定12:55:19 UTC、解析開始12:59:15 UTC。原カード・spec・元データ・元配布ソースは不変。
全６点・各64記事有効、欠測0。元W全予測最大差5.48e−11、全16,355,394予測値・13,824指標・
39,936区分寄与・840記事bootstrap＋寄与総和48平均bootstrapを別経路照合。
合成４検査、全選択CV・拡大行列解・勾配・単位長PCA・群内中心化・保存係数再構成もPASS。
数値約139.3秒、固定CPU60分枠内。４図PNG/SVGを目視確認、全系列/全20対照/全記事/全区分を保存。
生成と検証は自己監査。新LM取得/forward・GPU0、原稿未変更。

PC2〜8の追加関連は全６点に残るが、PC1はOPT中央で単独の追加価値もある。
柔軟な長さ対照N後の向き増分は全６点で正の記述区間、向きPC対ランダム差はOPT中央で0をまたぐ。
長さとの重複は条件依存。Qwen中央raw8では中央値+0.070771、記事平均−0.310188、
１記事wt2_train_27981が−24.224762。元W予測でも再現する失敗として除外せず保持。
ID等重みでもその失敗は消えない。因果機序・専用意味軸は未同定。

新しい結果は多重比較未調整の探索。元P4の４支持/２不確定、旧P3 O逆転、全旧結果は不変。
台帳§18の「PC1〜8をまとめた比較」は当時の範囲であり、今回の別探索で分解した。
有利な中央値だけで平均性能や全記事の頑健性を主張しない。
条件追加・再取得・第２探索・介入を行わず、この１セットで探索を終了する。
旧条件付き介入枠を復活させない。FP1/FN0/対照1/未裁定0、TL正式評価0は不変。
""")
    f.write(f"\nレポート `exploratory/p4_components_v1/ANALYSIS_REPORT_JA.md`、結果追補 `prereg/{result}`、"
            f"現行アンカー `evidence/{anchor}`、最新引き継ぎ `HANDOFF/{handoff}`。"
            "最終アンカーで旧ハッシュ・規律契約・台帳旧prefix・凍結条件・容量を実物照合する。\n")

hand = f"""# 引き継ぎ EDA-0003：保存P4の１セット完了、探索終了

2026-10-20。ユーザーのPC1/PC2〜8、長さ/向き、記事/token偏りの依頼を完了。台帳§19。
**新規実験も今回の保存データ探索も終了。自動継続しない。**

## 再開入口

[現行アンカー](../evidence/{anchor})のpreserved_sha256/artifact_sha256を実物照合する。
開始時旧アンカー5,110ファイル一致。原本保持に旧アンカー自身も追加し、６運用文書だけ現状を追記した。
[日本語レポート](../exploratory/p4_components_v1/ANALYSIS_REPORT_JA.md)、
[全条件表](../exploratory/p4_components_v1/DETAIL_TABLES.md)、
[固定条件](../prereg/{card})、[結果追補](../prereg/{result})。

## 実施したこと

run4b013b35152b/snapshot76c5c40ed614の保存状態のみ。３モデル×２点、既存64訓練/既知64評価、
先頭除外・同一記事×token ID内の比較。条件と終了点を12:55:19 UTCに固定してから12:59:15 UTCに数値処理開始。
PC1/PC2〜8/全8と同次元既存20対照、位置・線形logノルム・５ヒンジ、
単位長状態PCA8、ID等重み再採点、記事分布/種類/頻度/位置の寄与。
既知評価データなので全て探索、95%区間は多重比較未調整。条件/グリッド変更・除外なし。

６点・384記事点、欠測0。16,355,394予測値・13,824指標・39,936区分寄与・888集計を別計算経路で照合。
元Wの最大差5.48e−11。係数・CV・FIT_SEAL・予測・全対照・４図PNG/SVGを保存。
合成４検査、正規方程式/拡大行列解・基底・中心化・区分加法性も合格。
CPU数値処理約139.3秒、出力512MiB以内を最終アンカーで検証。自己監査、独立レビューではない。
LM取得/forward/GPU0、原稿・公開は未変更。

## 科学的結果と限界

PC2〜8は位置/ノルム考慮後にも追加関連を持つ。PC1単独は弱い条件が多いがOPT中央では明確な追加価値もある。
PC1無情報とは言わない。PC2〜8内の個別意味機能も未同定。
柔軟な長さ対照後の向き増分NUは全６点で正の記述区間、向きPC対ランダム差はOPT中央で不明瞭。
raw対unitの差は基底・相互作用・モデル形式も違い、純粋な長さや正規化の因果効果にできない。

Qwen中央raw8の中央値+0.070771、記事平均−0.310188。
wt2_train_27981（2nd Battalion 9th Marines）が増分−24.224762の大失敗。
元P4 Wと同じ予測で再現した。除外せず保持。単位長NUで失敗幅は縮むが−0.265742、ID等重みでもraw8失敗は残る。
原因未同定、典型記事と損失の尾を区別する。最小記事の詳細はreport_data.json、全記事はarticle_metrics.json。

**元P4確認的判定４支持/２不確定は不変。** 旧P3 O逆転も保持。
探索95%で元の99.583333%・最小効果基準を上書きしない。全モデル共通の機序や意味専用軸を主張しない。

## 終了境界

この固定１セットで探索を完了。新規モデル・層・記事・乱数・介入・第２探索を自動追加しない。
因果機序は未解決の問いとして残す。旧介入枠を復活させない。
原稿整理や投稿は別指示に従う。規律§2/§3不変、台帳追記、旧原稿・全ZIP・旧結果・固定カード・throughline保持。
FP1/FN0/対照1/未裁定0、TL正式評価0は不変。
"""
(BASE / "HANDOFF" / handoff).write_text(hand)
figs = [p for p in (OUT / "figures").glob("*") if p.suffix in [".png", ".svg"]]
assert len(figs) == 8
(OUT / "figure_qa.json").write_text(json.dumps(dict(
    status="PASS", at=datetime.now(timezone.utc).isoformat(),
    method="All four PNG figures visually inspected with view_image; SVG uses the same Matplotlib figures.",
    checks=["legible labels", "unadjusted intervals identified", "all six conditions shown",
            "outlier full range and explicitly marked zoom", "mean versus median distinguished"],
    sha256={str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in figs},
    independent_reviewer=False), ensure_ascii=False, indent=2) + "\n")
print("Operations, append-only ledger, separate results addendum and handoff written.")
