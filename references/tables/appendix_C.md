<a id="appendix-c"></a>
## Appendix C　短文入力と形状解析の補足

### C.1 短文入力

以下の文字列を全文入力した。反復文には末尾空白を一つ含む。以下をUTF-8文字列として与え、Appendix A.2の固定tokenizerを用いる。入力の識別情報はAppendix Mにまとめた。

**物語文**

```text
The old lighthouse keeper climbed the spiral staircase each evening, carrying a lantern that cast long shadows across the stone walls. He had performed this ritual for forty years, ever since the automated systems had failed during the great storm. The sea below crashed against the rocks with a rhythm that matched his breathing, and he found comfort in the predictability of waves and wind. Tonight, however, something was different. A strange light flickered on the horizon, pulsing with an irregular beat that made him uneasy.
```

**技術文**

```text
Transformer architectures process sequential input through layers of multi-head self-attention and feed-forward networks. Each attention head computes query, key, and value projections, producing a weighted sum over value vectors. The weights are determined by the softmax of scaled dot-product similarities between queries and keys. Residual connections and layer normalization stabilize training across many layers, enabling models with billions of parameters to converge.
```

**反復文**

```text
tick tock tick tock tick tock the clock goes round and round tick tock tick tock tick tock the pendulum swings back and forth tick tock tick tock tick tock time flows like a river tick tock tick tock tick tock the seconds become minutes tick tock tick tock tick tock and minutes become hours 
```

### C.2 主条件の曲線診断と補足量

**表C1　CPU・第7ブロック・C0の補足。** Qは二次曲線。全点当てはめと区間保留を併記し、同じ対象点のTSSを用いる。PC1–ノルム相関は元ノルムに対するPearson r。

| 文章 | PC2分散（%） | PC1–ノルム r | 全点Q：2D | 5分割Q：2D | 5分割Q：全空間 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 物語文 | 0.254 | 0.999606 | 0.997977 | -1141.91 | -52536.92 |
| 技術文 | 0.236 | 0.999611 | 0.998201 | -106.09 | -84591.77 |
| 反復文 | 0.409 | 0.999850 | 0.995927 | -21056.87 | -266432.95 |

全空間の曲線誤差は、三文章とも99.9999%以上が先頭点に集中した。先頭点を含むfoldでの外挿を、保存した各foldの誤差から点検した。Appendix Mに各抽出点の先頭誤差寄与と、保留区間に対する学習座標範囲を示す。

### C.3 保存記事の全形状条件

**表C2　EDA-0004の全18条件。** Qは各64記事の中央値。分布基準と曲がりの定義は表B1。全空間基準はQを全状態で計算する。

| モデル・点・処理 | 曲線Q：2D | 曲線Q：全空間 | 2D分布基準 | うち曲がり | 全空間分布基準 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 中央 C0 | -796.1 | -2.803e+04 | 0/64 | 0/64 | 0/64 |
| GPT-2 中央 C1 | 0.1404 | -0.04134 | 0/64 | 0/64 | 0/64 |
| GPT-2 中央 C4 | 0.1039 | -0.04338 | 0/64 | 0/64 | 0/64 |
| GPT-2 最終正規化後 C0 | 0.9282 | 0.8605 | 39/64 | 19/64 | 5/64 |
| GPT-2 最終正規化後 C1 | 0.929 | 0.8613 | 39/64 | 21/64 | 5/64 |
| GPT-2 最終正規化後 C4 | 0.9665 | 0.8379 | 2/64 | 2/64 | 1/64 |
| OPT 中央 C0 | 0.48 | -0.3569 | 0/64 | 0/64 | 0/64 |
| OPT 中央 C1 | 0.5893 | 0.03469 | 0/64 | 0/64 | 0/64 |
| OPT 中央 C4 | 0.5305 | 0.0203 | 0/64 | 0/64 | 0/64 |
| OPT 最終正規化後 C0 | 0.5365 | 0.03684 | 0/64 | 0/64 | 0/64 |
| OPT 最終正規化後 C1 | 0.5358 | 0.03684 | 0/64 | 0/64 | 0/64 |
| OPT 最終正規化後 C4 | 0.5355 | 0.03701 | 0/64 | 0/64 | 0/64 |
| Qwen 中央 C0 | -1.105e+07 | -1.49e+08 | 0/64 | 0/64 | 0/64 |
| Qwen 中央 C1 | 0.5453 | 0.03575 | 0/64 | 0/64 | 0/64 |
| Qwen 中央 C4 | 0.5542 | 0.03755 | 0/64 | 0/64 | 0/64 |
| Qwen 最終正規化後 C0 | 0.7039 | 0.2451 | 0/64 | 0/64 | 0/64 |
| Qwen 最終正規化後 C1 | 0.7039 | 0.2452 | 0/64 | 0/64 | 0/64 |
| Qwen 最終正規化後 C4 | 0.6526 | 0.1826 | 0/64 | 0/64 | 0/64 |

### C.4 残余特徴の読み出しと失敗記事

**表C3　EDA-0004のQwen中央層における残余七座標の追加価値。** 位置・元ノルム・文字数・訓練頻度を含む対照へのsurprisal増分。各64記事を全て保持した記述。C0/C1の最小値はいずれもwt2_train_27981。同記事の位置1に誤差の99.99%以上が集中した。

| 条件 | 増分中央値 | 記事平均 | 最小値 |
| --- | ---: | ---: | ---: |
| C0 | 0.12779 | -2.514e+06 | -1.609e+08 |
| C1 | 0.099351 | -6.3291e+05 | -4.0507e+07 |
| C4 | 0.12054 | 0.12429 | -0.18366 |

同記事は元の共有基底による読み出しでも失敗が記録されたが、表13と本表は異なる予測器・分割・対照を用いる。全18条件の基礎・残余追加後のR²、増分、平均・最小最大、全特徴の成績はAppendix Kに示す。

### C.5 短文検証の実行先比較

**表C4　P5のCPU/GPU照合。** 全12組を固定許容差で比較した。全12組の状態差はAppendix L、曲線外挿の診断値はMに示す。

| 項目 | 観測 |
| --- | --- |
| GPU | NVIDIA L4。実行候補A100/RTX PRO 6000からの差異 |
| 許容差内の状態組 | 11/12 |
| 超過した組 | 反復文・最終LayerNorm前、55,296座標中1座標（位置46、座標447） |
| 当該座標の絶対差／許容差 | 0.00024414／0.00022780 |
| 当該状態の相対Frobenius差 | 7.72×10⁻⁷ |
| 形状判定が一致した条件 | 36/36 |
| PC1分散比のCPU/GPU差の最大値 | 1.73×10⁻⁷ |


