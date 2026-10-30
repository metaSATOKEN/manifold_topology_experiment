"""Render the fixed EDA results; no fits, exclusions, or new analysis conditions."""
from pathlib import Path
import json
import time
import numpy as np
import numeric as n

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
read = lambda path: json.loads(path.read_text())
S = read(OUT / "summary.json")["results"]
ROWS = read(OUT / "article_metrics.json")
POINTS = [(m, p) for m in ["gpt2", "opt", "qwen"] for p in ["middle", "final_norm"]]
LABELS = ["GPT-2 中央", "GPT-2 最終LN", "OPT 中央", "OPT 最終LN", "Qwen 中央", "Qwen 最終RMSNorm"]
SHORT = ["GPT-2 middle", "GPT-2 final LN", "OPT middle", "OPT final LN", "Qwen middle", "Qwen final RMSNorm"]
MEANS = {}
start = time.monotonic()
mapping = dict(B_to_A1="A1_gain", B_to_A7="A7_gain", B_to_A8="A8_gain",
               B_to_N="N_minus_B", N_to_NR="NR_gain", N_to_NU="NU_gain",
               A8_adv="A8_adv", NU_adv="NU_adv")
for m, p in POINTS:
    rows = [r for r in ROWS if r["model"] == m and r["point"] == p]
    result = {}
    for name, metric in mapping.items():
        # The total of the prespecified additive token contributions, not a new endpoint.
        values = np.array([sum(t["contributions"][name] for t in r["bias"]["type"]) for r in rows])
        np.testing.assert_allclose(values, [r["metrics"][metric] for r in rows], atol=1e-10)
        stat = n.describe(values, mean=True)
        rng = np.random.default_rng(20261029)
        check = values[rng.integers(0, 64, (20000, 64))].sum(1) / 64
        np.testing.assert_allclose([stat["lo"], stat["hi"]], np.quantile(check, [.025, .975]), atol=1e-12)
        result[name] = stat
    result["worst_A8_article"] = min(rows, key=lambda r: r["metrics"]["A8_gain"])
    MEANS[m + "/" + p] = result
report_data = dict(
    role="Rendering aggregation only: totals of prespecified token contributions; no new fits or exclusions",
    results=MEANS, additional_mean_bootstrap_checks=48,
    seconds=time.monotonic() - start,
)
(OUT / "report_data.json").write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n")


def fmt(d, digits=5):
    return f'{d["estimate"]:.{digits}f} [{d["lo"]:.{digits}f}, {d["hi"]:.{digits}f}]'


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main_table(metrics):
    return table(["条件", *metrics], [
        [label, *[fmt(S[m][p]["metrics"][metric]) for metric in metrics]]
        for label, (m, p) in zip(LABELS, POINTS)])


def stats_table(stats):
    return table(["量", "推定値 [記述95%区間]", "IQR", "正/負/0", "最小〜最大", "１記事除外範囲", "上位５の正寄与割合"],
                 [[k, fmt(v), f'{v["q25"]:.5f}〜{v["q75"]:.5f}',
                   f'{v["positive"]}/{v["negative"]}/{v["zero"]}', f'{v["min"]:.5f}〜{v["max"]:.5f}',
                   f'{v["leave_one_out_min"]:.5f}〜{v["leave_one_out_max"]:.5f}',
                   "NA" if v["top5_positive_share"] is None else f'{v["top5_positive_share"]:.3%}']
                  for k, v in stats.items()])


details = [
    "# EDA-0003 全条件表",
    "全６点・各64記事、欠測0。多重比較未調整の探索的な95%記事bootstrap区間。"
    "主題・等ID重み・個別ランダムは記事中央値、token寄与とその総和だけ記事平均。"
    "IQR・最小/最大は記事値。上位５割合は正の値だけの合計に対する集中度で、負の失敗を含む正味改善率ではない。"
    "主題表の１記事除外は中央値、寄与表の１記事除外は平均。除外して学習し直していない。",
    "略号：P=位置４項、B=P+線形logノルム、N=B+５ヒンジ。"
    "A1/A7/A8=B+raw PC1/PC2–8/PC1–8。NR=N+raw PC8、PU=P+unit-state PCA8、NU=N+unit-state PCA8。"
    "gainは各基礎からのΔR²、advは同次元ランダム20組の増分平均との差。",
]
for label, (m, p) in zip(LABELS, POINTS):
    d = S[m][p]
    details += [f"## {label}", f"有効範囲：{d['coverage']}", "### 元のtoken等重み", stats_table(d["metrics"]),
                "### 同じ予測をID等重みで再採点", stats_table(d["equal_ID_metrics"]),
                "### PCと基礎のR²水準", stats_table(d["r2_levels"]),
                "### token寄与の総和（記事平均）",
                stats_table({k: v for k, v in MEANS[m + "/" + p].items() if k != "worst_A8_article"})]
    for part, strata in d["bias"].items():
        details += [f"### token区分：{part}", table(["区分", "tokens", "token割合", "該当記事"],
                    [[t["label"], t["total_tokens"], f'{t["token_fraction"]:.3%}', t["articles_present"]] for t in strata])]
        for t in strata:
            details += [f"#### {t['label']}：寄与は記事平均", stats_table(t["contributions"])]
    details += ["### ランダム20組の全結果（各組の増分の記事中央値）"]
    for family, reps in d["random_replicates"].items():
        details += [f"#### {family}", stats_table({f"random {i + 1:02}": v for i, v in enumerate(reps)})]
    details += ["### 訓練CVの選択（順序はPC、random 01〜20）",
                "基礎alphaは0.0001。NのヒンジgammaはNR/NUでも固定。端の選択は記録し、グリッドを拡張していない。"]
    cvrows = []
    for family, cv in d["cv"].items():
        if "selected_gammas" not in cv:
            continue
        gs = cv["selected_gammas"]
        cvrows.append([family, ", ".join(f"{g:g}" for g in gs),
                       sum(g == .0001 for g in gs), sum(g == 1 for g in gs)])
    details += [table(["系列", "選択gamma", "下端件数", "上端件数"], cvrows),
                f"全CV候補のMSEと検算：`results/{m}/{p}/cv.json`。"]
details += ["## 全記事の主要値",
            "全18指標、等ID重み指標、全ランダムR²、全token区分の未丸め値は `article_metrics.json`。以下も除外なし。",
            table(["model/point", "article", "tokens/ID群", "A1 gain", "A7 gain", "A8 gain", "NR gain", "NU gain", "NU adv"],
                  [[r["model"] + "/" + r["point"], r["document_id"], f'{r["tokens"]}/{r["groups"]}',
                    *[f'{r["metrics"][k]:.6f}' for k in ["A1_gain", "A7_gain", "A8_gain", "NR_gain", "NU_gain", "NU_adv"]]]
                   for r in ROWS])]
(OUT / "DETAIL_TABLES.md").write_text("\n\n".join(details) + "\n")

bad = MEANS["qwen/middle"]["worst_A8_article"]
assert bad["document_id"] == "wt2_train_27981"
means_table = table(["条件", "元raw8：中央値", "元raw8：記事平均 [95%]", "N+向き：中央値", "N+向き：記事平均 [95%]"],
                   [[label, f'{S[m][p]["metrics"]["A8_gain"]["estimate"]:.5f}',
                     fmt(MEANS[m + "/" + p]["B_to_A8"]), f'{S[m][p]["metrics"]["NU_gain"]["estimate"]:.5f}',
                     fmt(MEANS[m + "/" + p]["N_to_NU"])] for label, (m, p) in zip(LABELS, POINTS)])
equal_table = table(["条件", "raw8：token等重み → ID等重み", "向き：token等重み → ID等重み", "向きのID等重みランダム差 [95%]"],
                    [[label,
                      f'{S[m][p]["metrics"]["A8_gain"]["estimate"]:.5f} → {S[m][p]["equal_ID_metrics"]["A8_gain"]["estimate"]:.5f}',
                      f'{S[m][p]["metrics"]["NU_gain"]["estimate"]:.5f} → {S[m][p]["equal_ID_metrics"]["NU_gain"]["estimate"]:.5f}',
                      fmt(S[m][p]["equal_ID_metrics"]["NU_adv"])] for label, (m, p) in zip(LABELS, POINTS)])
coverage_table = table(["条件", "評価記事", "反復tokens", "記事×ID群", "欠測"],
                      [[label, 64, S[m][p]["coverage"]["tokens"], S[m][p]["coverage"]["groups"], 0]
                       for label, (m, p) in zip(LABELS, POINTS)])
worst_table = table(["条件", "最小A8増分の記事", "最小", "正/64", "１記事除外中央値範囲"],
                    [[label, MEANS[m + "/" + p]["worst_A8_article"]["document_id"],
                      f'{S[m][p]["metrics"]["A8_gain"]["min"]:.5f}', S[m][p]["metrics"]["A8_gain"]["positive"],
                      f'{S[m][p]["metrics"]["A8_gain"]["leave_one_out_min"]:.5f}〜{S[m][p]["metrics"]["A8_gain"]["leave_one_out_max"]:.5f}']
                     for label, (m, p) in zip(LABELS, POINTS)])
verification = read(OUT / "verification.json")
max_old = max(x["P4_W_prediction_max_difference"] for x in verification["point_audits"].values())
report = f"""# EDA-0003：PC1と残りのPC、長さと向きの最終探索

2026-10-20。**完了／この１セットで探索終了。保存P4だけを使用、新しいLM forward・取得・GPU実行は0。旧結果・旧原稿は保持。**

## 結果を先に

今回の読み出しでは、**位置とノルムを考慮した後の追加の関連は、PC1単独よりPC2〜8の組に多く残った**。また、状態の長さを柔軟に考慮しても、向きの８方向から読める関連は６条件とも残った。ただしOPT中央層では、向きのPCを選ぶ利点をランダム方向との差としては明確に示せなかった。

同時に、**Qwen中央層には元のraw８PC読み出しが極端に外れる１記事があり、中央値は正でも記事平均は負になる**ことが分かった。「多くの記事で有利」と「大きな失敗も含め平均的に有利」は別の主張になる。この失敗を残したまま報告する。

到達点は、どの座標群・どの長さ対照・どの記事やtokenで関連が読めるかを限定したこと。モデルがその方向を因果的に利用している仕組みは未同定であり、ここで未解決の問いとして区切る。

## 1. 何を固定して調べたか

[解析条件と終了点](../../prereg/EDA-0003_v1_p4_components_radius_direction.md)を12:55:19 UTC（21:55:19 JST）にローカル固定し、12:59:15 UTCに数値解析開始。[固定記録](plan_freeze.json)と[実行記録](execution_manifest.json)を保存した。**P4の結果を知った上の探索であり、確認的な事前登録ではない**。公開レジストリの第三者時刻証明もない。

GPT-2、OPT-1.3B、Qwen2.5-1.5B baseの中央層と最終正規化後、全６条件。既存64訓練記事で新しい読み出しを当てはめ、既知の64評価記事で採点した。後者を再び未見の評価とは呼ばない。先頭tokenを除き、記事×token ID内で中心化し、反復２回以上のIDを評価する。32tokens・８ID群等の元規則を変更していない。

{coverage_table}

同じ記事がモデル間・層間で共有されるため、384点は384個の独立記事ではない。目的は同じtoken IDの出現ごとのsurprisal差。目的y[t]は前時刻logits[t−1]に由来し、状態h[t]はすでにtoken tを見ている。**事後的関連の読み出しで、未知の次tokenの予測性能ではない。**

記号は以下の通り。

| 記号 | 意味 |
| --- | --- |
| P | 位置４項 |
| B | P＋線形log1p(元ノルム) |
| N | B＋長さの５ヒンジ。節点は訓練10/25/50/75/90百分位 |
| A1 / A7 / A8 | B＋元PC1 / 元PC2〜8 / 元PC1〜8 |
| NR | N＋元のraw PC1〜8 |
| PU / NU | P / N＋単位長状態u=h/||h||を訓練PCAした８方向 |

元PC基底と既存20ランダム基底を使用し、１・７・８次元の比較では対照の次元も揃えた。向きのPCAは訓練記事だけから作成し、同じ20ランダム基底を単位長状態に射影した。単位長PCAと元のPCAは別の軸なので、番号を同じ意味と解釈しない。

基礎alphaは元P4 Wの0.0001に固定。追加係数のgammaは既存５候補、訓練４foldで選択した。基礎と追加の係数は共同推定し、評価記事では調整していない。平均・尺度・単位長PCAは訓練全体で共有する条件付きCVで、完全nested CVではない。元の基底も全訓練に依存する。

表のgainは基礎からのΔR²、advはPCの増分と同次元ランダム20組の増分平均との差。主題の集計は**記事中央値と20,000回記事bootstrapの記述95%区間**（seed20261029）。多重比較未調整で、確認的p値や新たな「支持」判定を作らない。

## 2. PC1とPC2〜8を切り分けた結果

{main_table(["A1_gain", "A7_gain", "A8_gain"])}

{main_table(["A1_adv", "A7_adv", "A8_adv"])}

PC2〜8の７方向は全６条件で正の増分を持ち、同次元ランダムとの差も今回の95%記述区間では正だった。PC1単独はGPT-2の両点とOPT最終層で明瞭な増分を示さず、Qwenでは小さい正の増分があるがランダム選択に対する優位性は明確でない。**OPT中央層は例外で、PC1単独にも増分とランダム優位差が残る。**

ただしこれは「位置とlogノルムをすでに使った後」の比較である。PC1が絶対的に無情報、分散集中が無意味、あるいはPC1が長さだけを表すとは言わない。１方向と７方向の容量も違うので、単純に「１方向あたり７倍」等とは比較できない。

{main_table(["A8_minus_A1", "A8_minus_A7"])}

PC1に残り７本を足す追加価値は大きい。一方、PC2〜8にPC1を足す差はGPT-2の両点とOPTの両点で小さく区間が0をまたぐ。Qwenでは両点で小さい正の差が残る。各モデルを罰則ごと当てはめ直しているため、これは加算可能な因果的寄与の分解ではない。PC2〜8内の各PCの役割も今回分けていない。

![PC分割と追加価値](figures/01_components.png)

## 3. 長さだけで説明できる部分、向きから残る部分

{main_table(["B_minus_P", "N_minus_B"])}

線形logノルムを加えた差と、さらに非線形な長さ関係を許す差を分けると、単純な長さ対照だけでは不十分な条件が見える。例えばGPT-2最終層は線形logノルムの上乗せがほぼない一方、ヒンジ追加の増分は0.02238。OPT中央層もヒンジで0.01578増える。これはこの読み出しにおける非線形な長さとの関連であり、長さがモデルの判断を引き起こす証明ではない。

{main_table(["NR_gain", "PU_gain", "NU_gain"])}

{main_table(["NR_adv", "PU_adv", "NU_adv"])}

柔軟な長さ対照Nを超える向きの増分NU_gainは６条件で正の記述区間になった。したがって、**今回許した位置・長さの有限の説明だけで、読み出せる関連が全て消えるわけではない**。一方、向きのPC対ランダムの差はOPT中央層で0.00016 [−0.00650, 0.00379]となり、方向一般から読めることと、上位PCを選ぶ利点があることを分ける必要がある。他の５条件では正の記述区間だった。

{main_table(["NR_gain_minus_A8_gain", "NR_minus_NU"])}

長さ対照を強めると元raw PC8の増分が減る条件は、特にGPT-2最終層とOPT中央層。記事ごとの増分変化の中央値はそれぞれ−0.01703、−0.01577で、一部が長さの非線形な説明と重複する。ただし他条件では一律に減らず、「PC効果の一定割合は長さ」と一般化できない。

N条件のrawと向きの比較もモデル依存で、GPT-2中央層では向きの方が有利、OPT中央層とQwenではrawの方が有利になる。raw射影には長さ×方向の積が含まれ、向き側は別基底かつ加法的なモデルである。**この差は長さの純粋な寄与でも、正規化による情報喪失の因果効果でもない。** 状態の単位長化は保存データの座標変換で、Transformer内部への介入ではない。

![長さと向き](figures/02_radius_direction.png)

## 4. 記事の偏り：中央値だけでは捉えられない失敗

{means_table}

ここだけ右側の区間は記事**平均**のbootstrap。token区分寄与の合計としてあらかじめ定めた量を示している。全記事tokenをプールしたR²ではない。

Qwen中央層の `wt2_train_27981`（2nd Battalion 9th Marines、363反復tokens・72ID群）では、BのR²=0.03278に対しA8のR²=−24.19198、**増分=−24.22476**。A7だけでも−18.24336と大きく外す。柔軟な長さ対照後のraw8増分NRは−30.57883、向き側のNUは−0.26574。NUでは失敗幅が小さくなるが、改善に転じてはいない。ID等重みにしてもraw8の増分は−32.38148であり、繰り返し回数を揃えるだけでは解消しない。

R²は0〜1に制限されず、このように大きな負値を取りうる。今回は元P4 Wの予測を再構成しても同じ失敗を再現した。この記事を除外せず、外れ値処理も罰則の再選択もしていない。原因がノルムの異常・特定語・領域外挿等のどれかは、この固定セットから確定しない。

{worst_table}

１記事ずつ除いた中央値が安定していても、大損失がないとは保証しない。正の寄与上位５記事の集中度も負の失敗を扱わないため、それだけで頑健と結論しない。全記事の最小/最大、正負件数、IQR、集中度は全条件表に併記した。

![Qwen中央層の全記事と平均寄与](figures/04_article_bias.png)

## 5. token種類・頻度・位置・ID重みの偏り

{equal_table}

同じ保存予測を記事内で各反復IDが等重みになるよう再採点しても、raw8と向きの増分中央値は各条件で正に残る。NUのランダム優位差は、この採点でもOPT中央層が明確ではない。多数回現れるIDだけで中央値の傾向全体が生じたとは考えにくいが、前節の大きな失敗は解消しない。再学習やID別の因果効果の推定ではない。

![token等重みとID等重み](figures/03_equal_id.png)

補助区分の値は各tokenの二乗誤差改善を記事全体の目的SSで割った寄与で、区分内R²ではない。全区分の寄与が記事増分に足し合わさる。20ランダムは二乗誤差の平均であり、平均予測の誤差に置き換えていない。

- 種類ではlettersとpunct_symbolの両方が正に寄与する条件が多く、句読点だけの現象ではない。ただしlettersは「Unicode文字を含むtoken」という分類で、意味語・品詞を同定したわけではない。Qwen中央層のraw8ではletters区分が記事平均−0.33491と失敗を担い、向きNUでは同区分+0.03651になる。これは区分全般の因果的性質とは言わない。
- 頻度は訓練ID出現数で区分した。100回以上のtokenはGPT-2/OPTで54.4%、Qwenで60.8%を占めるため、寄与総量が大きいだけで１tokenあたり特別とは解釈しない。GPT-2最終層のraw8では訓練未出現ID区分にも+0.03022、100回以上区分に+0.03232の平均寄与がある。Qwen中央層のraw8では1〜9回区分が−0.36584となり、平均の負値がそこに集中する。
- 位置は先頭そのものを除いた後も前半、とくに最初の四分区間の寄与が大きい条件がある。Qwen中央層のraw8では最初の四分区間が−0.25593。NUでは全四区間の平均寄与が正になる。これは最初の四分を除外すべきという裁定ではなく、固定区分での記述である。

全種類・全頻度・全位置区分の寄与、95%区間、token割合、該当記事数、負の値を[全条件表](DETAIL_TABLES.md)に保存した。本文の例だけを都合よく選んで全体の証拠にしない。

## 6. 旧結果と論文の主張に対する含意

| これまでの見方 | 今回追加できる限定 | 残る限界 |
| --- | --- | --- |
| top8全体から追加の関連が読める | この群内読み出しではPC2〜8に多く残り、PC1の上乗せは条件依存 | PC2〜8が各々独立した意味機能を持つとは未確認 |
| ノルムを考慮しても関連が残る | 長さの非線形対照で一部縮むが、単位長の向きにも関連が残る | 有限の加法モデルであり、全交絡除去でも因果機序でもない |
| ３モデルでPCの有利さを比較できた | 向きのPC対ランダムはOPT中央層で明確でない | 全モデル・全層共通の優位性には広げられない |
| 記事中央値で有利な条件がある | Qwen中央層では大失敗１記事のため記事平均が負になる | 個々の記事への安定性・平均損失改善は保証しない |

**元P4の確認的裁定は４条件支持・２条件不確定のまま**。今回の未調整95%区間を使って、元P4の99.583333%区間・最小効果基準を緩めたり、６条件支持へ変更したりしない。元P3の通常読み出しOでの逆転も保持する。今回のW系統の分解はその結果を取り消すものではない。

原稿で採用するなら、「主成分の分散順位と追加の読み出し価値は同じではない」「長さと向きの関係はモデルと読み出しに依存する」「典型的記事と失敗の尾を区別する」という範囲が妥当。PC1の普遍的ノルム軸、PC2〜8の意味専用軸、正規化層が情報を壊す機序といった主張には進めない。**今回、原稿自体は編集していない。**

## 7. 検証・保存物・終了

合成検査４件に合格後、全６点の元P4 W全予測を再構成。今回の最大差は{max_old:.3g}。新しい系列を含む16,355,394予測値、384記事点の13,824指標、39,936区分寄与を別計算経路で点検した。主要・補助集計840組のbootstrapと、寄与総和48組の平均bootstrapを別実装で照合。基底直交性、固有方程式、単位ノルム、群内中心化、全選択fitの拡大行列解・勾配、選択CV、保存係数再読込からの予測、区分加法性も合格した。

数値解析94.41秒、集計・別経路検算44.13秒、寄与総和の表示用集計{report_data["seconds"]:.2f}秒（図表・文章作成時間を除く）。CPU数値処理60分の固定枠内。技術的欠測・解析不能は0。生成と検証は同じagentの**自己監査**であり、独立レビューではない。

| 保存物 | 内容 |
| --- | --- |
| [全条件表](DETAIL_TABLES.md) | 全系列・全20ランダム・全記事・全token区分・CV端選択 |
| [集計JSON](summary.json) / [記事指標](article_metrics.json) | 未丸め値と全分母 |
| [寄与総和](report_data.json) | 記事平均、全６条件の最小記事、表示用集計の役割 |
| [数値検証](verification.json) / [実行完了](execution_result.json) | 各点の別計算経路・検算数・時間 |
| `results/model/point/` | 係数、基底、CV、予測、評価前FIT_SEAL、COMMIT |
| [開始時保持](preservation_before.json) / [最終照合](../../evidence/eda3_final_checks_2026-10-20.json) | 旧原本・旧結果・凍結条件・今回成果物のハッシュ |
| [結果追補](../../prereg/EDA-0003_v1_results_2026-10-20.md) | 解析カードを変えずに結果を追加 |

条件と終了点は変更せず、今回の１セットを完了した。新しいモデル・層・記事・乱数・介入・第２探索セットへ自動延長しない。追加実験枠も復活させない。因果機序は未解決として残し、次の作業を行う場合は、今回までの証拠を使った原稿整理などを別に指示する。
"""
(OUT / "ANALYSIS_REPORT_JA.md").write_text(report)
print("Report, full tables and 48 contribution-total checks written.")
print("Qwen middle raw8 mean:", fmt(MEANS["qwen/middle"]["B_to_A8"]))
print("Total numerical seconds:", verification["analysis_seconds"] + verification["aggregation_seconds"] + report_data["seconds"])
