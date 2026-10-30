# P5 全条件表

事前指定のCPU36条件とCUDA36条件をすべて掲載。C0全点、C1先頭除外、C4単位長化。
各文章・点・処理で基底を再計算。V2/V3/V8は累積寄与率。PRはparticipation ratio。
Qは固定５分割、Qfitは全点当てはめ。PC1/残差軸の符号は基底の規約による。

## 分散とノルム

| 実行先 | 文章 | 点 | 処理 | V1 | pc2_variance | V2 | V3 | V8 | participation | pc1_norm_r | float32_V1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | narrative | block06 | C0 | 0.9670292 | 0.00231735 | 0.9693466 | 0.9714868 | 0.9769865 | 1.069323 | 0.9996282 | 0.9670295 |
| cpu | technical | block06 | C0 | 0.9709887 | 0.002018633 | 0.9730073 | 0.9749049 | 0.9804015 | 1.060624 | 0.9996293 | 0.9709889 |
| cpu | repetitive | block06 | C0 | 0.9816214 | 0.003223196 | 0.9848446 | 0.9873568 | 0.9931497 | 1.037767 | 0.9998775 | 0.9816214 |
| cpu | narrative | block06 | C1 | 0.07311969 | 0.06375389 | 0.1368736 | 0.182226 | 0.3381096 | 40.77873 | 0.6998396 | 0.07311969 |
| cpu | technical | block06 | C1 | 0.07454157 | 0.06459904 | 0.1391406 | 0.1884094 | 0.3583438 | 37.37056 | -0.5428162 | 0.07454157 |
| cpu | repetitive | block06 | C1 | 0.1730693 | 0.1381331 | 0.3112024 | 0.4144921 | 0.6628793 | 12.75185 | 0.4013988 | 0.1730693 |
| cpu | narrative | block06 | C4 | 0.06787501 | 0.0642387 | 0.1321137 | 0.1791277 | 0.3358757 | 41.87453 | -0.04935525 | 0.06787501 |
| cpu | technical | block06 | C4 | 0.07163286 | 0.06468145 | 0.1363143 | 0.1858568 | 0.3519777 | 38.58197 | 0.005352057 | 0.07163286 |
| cpu | repetitive | block06 | C4 | 0.1610562 | 0.1535713 | 0.3146275 | 0.4139829 | 0.6515551 | 12.96158 | 0.07224074 | 0.1610562 |
| cpu | narrative | block07 | C0 | 0.9606163 | 0.002540854 | 0.9631572 | 0.9655989 | 0.9725149 | 1.083635 | 0.9996062 | 0.9606164 |
| cpu | technical | block07 | C0 | 0.9657098 | 0.002361806 | 0.9680716 | 0.9702547 | 0.9768095 | 1.072241 | 0.9996107 | 0.9657098 |
| cpu | repetitive | block07 | C0 | 0.9764441 | 0.00409082 | 0.9805349 | 0.9839093 | 0.9909488 | 1.048783 | 0.99985 | 0.9764441 |
| cpu | narrative | block07 | C1 | 0.06956966 | 0.0609184 | 0.1304881 | 0.1766437 | 0.338754 | 41.35673 | 0.6358068 | 0.06956965 |
| cpu | technical | block07 | C1 | 0.0737963 | 0.06282612 | 0.1366224 | 0.1873393 | 0.3575507 | 37.39365 | 0.2930843 | 0.0737963 |
| cpu | repetitive | block07 | C1 | 0.1710274 | 0.1430694 | 0.3140969 | 0.4073897 | 0.6538416 | 13.01468 | 0.4198776 | 0.1710274 |
| cpu | narrative | block07 | C4 | 0.06354073 | 0.06028765 | 0.1238284 | 0.1726756 | 0.3350609 | 42.48482 | 0.09085347 | 0.06354073 |
| cpu | technical | block07 | C4 | 0.07009264 | 0.06230202 | 0.1323947 | 0.1823895 | 0.3513649 | 38.71078 | 0.08514446 | 0.07009263 |
| cpu | repetitive | block07 | C4 | 0.1625673 | 0.1549851 | 0.3175523 | 0.41027 | 0.6489086 | 13.07328 | 0.00960444 | 0.1625673 |
| cpu | narrative | block12_preln | C0 | 0.1981946 | 0.1269829 | 0.3251775 | 0.3902538 | 0.575729 | 14.08991 | 0.8184847 | 0.1981946 |
| cpu | technical | block12_preln | C0 | 0.3069921 | 0.08347804 | 0.3904701 | 0.451657 | 0.6306295 | 8.68316 | 0.9174441 | 0.3069921 |
| cpu | repetitive | block12_preln | C0 | 0.6966459 | 0.1478022 | 0.8444481 | 0.8730648 | 0.94519 | 1.963083 | -0.4885814 | 0.6966459 |
| cpu | narrative | block12_preln | C1 | 0.1937341 | 0.1186006 | 0.3123347 | 0.3785626 | 0.5681543 | 14.74475 | 0.7331811 | 0.1937341 |
| cpu | technical | block12_preln | C1 | 0.2873606 | 0.08493287 | 0.3722935 | 0.4395265 | 0.6133686 | 9.584905 | 0.9567479 | 0.2873606 |
| cpu | repetitive | block12_preln | C1 | 0.7173839 | 0.1434776 | 0.8608615 | 0.8877959 | 0.9482302 | 1.861917 | -0.4875519 | 0.7173839 |
| cpu | narrative | block12_preln | C4 | 0.1746286 | 0.09453741 | 0.269166 | 0.3394828 | 0.5235607 | 17.61015 | 0.424523 | 0.1746286 |
| cpu | technical | block12_preln | C4 | 0.1740175 | 0.1137296 | 0.2877471 | 0.3512644 | 0.5537546 | 16.34429 | 0.6328623 | 0.1740175 |
| cpu | repetitive | block12_preln | C4 | 0.705219 | 0.1293668 | 0.8345858 | 0.8665126 | 0.9354105 | 1.936017 | -0.3541467 | 0.7052189 |
| cpu | narrative | final_ln | C0 | 0.7298604 | 0.09437861 | 0.824239 | 0.888408 | 0.9661781 | 1.826616 | 0.9972496 | 0.7298604 |
| cpu | technical | final_ln | C0 | 0.7914776 | 0.09097638 | 0.882454 | 0.9052506 | 0.9671648 | 1.571847 | 0.9967521 | 0.7914776 |
| cpu | repetitive | final_ln | C0 | 0.9275426 | 0.06052876 | 0.9880713 | 0.9947366 | 0.9984756 | 1.157344 | 0.7957181 | 0.9275424 |
| cpu | narrative | final_ln | C1 | 0.7204449 | 0.09656128 | 0.8170062 | 0.883247 | 0.9642173 | 1.870419 | 0.9971854 | 0.7204449 |
| cpu | technical | final_ln | C1 | 0.7750861 | 0.1008725 | 0.8759586 | 0.8994515 | 0.9638559 | 1.632691 | 0.9960141 | 0.7750862 |
| cpu | repetitive | final_ln | C1 | 0.9282393 | 0.06027142 | 0.9885107 | 0.9948533 | 0.9985469 | 1.155662 | 0.7980531 | 0.9282393 |
| cpu | narrative | final_ln | C4 | 0.4516106 | 0.2046369 | 0.6562475 | 0.752073 | 0.903522 | 3.835699 | 0.5907455 | 0.4516105 |
| cpu | technical | final_ln | C4 | 0.4561774 | 0.1785071 | 0.6346846 | 0.7191226 | 0.8857723 | 3.930733 | -0.699027 | 0.4561774 |
| cpu | repetitive | final_ln | C4 | 0.8571361 | 0.1173714 | 0.9745075 | 0.9900365 | 0.9971315 | 1.335623 | 0.4090384 | 0.8571362 |
| cuda | narrative | block06 | C0 | 0.9670292 | 0.002317351 | 0.9693466 | 0.9714868 | 0.9769865 | 1.069323 | 0.9996282 | 0.9670293 |
| cuda | technical | block06 | C0 | 0.9709887 | 0.002018633 | 0.9730073 | 0.9749049 | 0.9804015 | 1.060624 | 0.9996293 | 0.9709888 |
| cuda | repetitive | block06 | C0 | 0.9816214 | 0.003223195 | 0.9848446 | 0.9873568 | 0.9931497 | 1.037767 | 0.9998775 | 0.9816214 |
| cuda | narrative | block06 | C1 | 0.07311969 | 0.06375389 | 0.1368736 | 0.182226 | 0.3381097 | 40.77872 | 0.6998397 | 0.07311969 |
| cuda | technical | block06 | C1 | 0.07454157 | 0.06459905 | 0.1391406 | 0.1884094 | 0.3583438 | 37.37056 | -0.5428158 | 0.07454156 |
| cuda | repetitive | block06 | C1 | 0.1730694 | 0.1381331 | 0.3112024 | 0.4144921 | 0.6628794 | 12.75185 | 0.4013991 | 0.1730694 |
| cuda | narrative | block06 | C4 | 0.06787501 | 0.0642387 | 0.1321137 | 0.1791277 | 0.3358758 | 41.87453 | -0.04935518 | 0.06787501 |
| cuda | technical | block06 | C4 | 0.07163286 | 0.06468144 | 0.1363143 | 0.1858568 | 0.3519777 | 38.58197 | 0.005352083 | 0.07163286 |
| cuda | repetitive | block06 | C4 | 0.1610562 | 0.1535713 | 0.3146275 | 0.4139829 | 0.6515551 | 12.96158 | 0.07224095 | 0.1610562 |
| cuda | narrative | block07 | C0 | 0.9606163 | 0.002540854 | 0.9631572 | 0.9655989 | 0.9725149 | 1.083635 | 0.9996062 | 0.9606164 |
| cuda | technical | block07 | C0 | 0.9657098 | 0.002361806 | 0.9680716 | 0.9702547 | 0.9768095 | 1.072241 | 0.9996107 | 0.9657097 |
| cuda | repetitive | block07 | C0 | 0.9764441 | 0.004090818 | 0.9805349 | 0.9839093 | 0.9909488 | 1.048783 | 0.99985 | 0.9764441 |
| cuda | narrative | block07 | C1 | 0.06956966 | 0.0609184 | 0.1304881 | 0.1766437 | 0.3387541 | 41.35673 | 0.635807 | 0.06956966 |
| cuda | technical | block07 | C1 | 0.07379629 | 0.06282612 | 0.1366224 | 0.1873393 | 0.3575507 | 37.39366 | 0.2930836 | 0.07379629 |
| cuda | repetitive | block07 | C1 | 0.1710275 | 0.1430694 | 0.3140969 | 0.4073897 | 0.6538416 | 13.01468 | 0.4198776 | 0.1710275 |
| cuda | narrative | block07 | C4 | 0.06354074 | 0.06028765 | 0.1238284 | 0.1726756 | 0.3350609 | 42.48482 | 0.09085359 | 0.06354073 |
| cuda | technical | block07 | C4 | 0.07009263 | 0.06230201 | 0.1323946 | 0.1823895 | 0.3513649 | 38.71078 | 0.08514442 | 0.07009263 |
| cuda | repetitive | block07 | C4 | 0.1625673 | 0.1549851 | 0.3175523 | 0.41027 | 0.6489086 | 13.07328 | 0.009604449 | 0.1625672 |
| cuda | narrative | block12_preln | C0 | 0.1981946 | 0.1269829 | 0.3251775 | 0.3902538 | 0.5757289 | 14.08991 | 0.8184852 | 0.1981946 |
| cuda | technical | block12_preln | C0 | 0.3069921 | 0.08347811 | 0.3904702 | 0.4516571 | 0.6306295 | 8.68316 | 0.9174445 | 0.3069921 |
| cuda | repetitive | block12_preln | C0 | 0.6966459 | 0.1478022 | 0.8444481 | 0.8730649 | 0.94519 | 1.963083 | -0.4885814 | 0.6966459 |
| cuda | narrative | block12_preln | C1 | 0.1937341 | 0.1186006 | 0.3123347 | 0.3785625 | 0.5681543 | 14.74476 | 0.7331811 | 0.1937341 |
| cuda | technical | block12_preln | C1 | 0.2873606 | 0.08493293 | 0.3722935 | 0.4395265 | 0.6133686 | 9.584905 | 0.9567478 | 0.2873605 |
| cuda | repetitive | block12_preln | C1 | 0.717384 | 0.1434776 | 0.8608615 | 0.887796 | 0.9482302 | 1.861917 | -0.4875519 | 0.717384 |
| cuda | narrative | block12_preln | C4 | 0.1746286 | 0.09453735 | 0.269166 | 0.3394827 | 0.5235607 | 17.61015 | 0.4245222 | 0.1746286 |
| cuda | technical | block12_preln | C4 | 0.1740175 | 0.1137296 | 0.2877471 | 0.3512644 | 0.5537546 | 16.34429 | 0.632863 | 0.1740175 |
| cuda | repetitive | block12_preln | C4 | 0.705219 | 0.1293668 | 0.8345858 | 0.8665126 | 0.9354105 | 1.936017 | -0.3541467 | 0.705219 |
| cuda | narrative | final_ln | C0 | 0.7298604 | 0.09437863 | 0.824239 | 0.888408 | 0.9661781 | 1.826616 | 0.9972496 | 0.7298604 |
| cuda | technical | final_ln | C0 | 0.7914776 | 0.09097641 | 0.882454 | 0.9052507 | 0.9671648 | 1.571847 | 0.9967521 | 0.7914777 |
| cuda | repetitive | final_ln | C0 | 0.9275426 | 0.06052876 | 0.9880713 | 0.9947366 | 0.9984756 | 1.157344 | 0.7957182 | 0.9275424 |
| cuda | narrative | final_ln | C1 | 0.720445 | 0.09656129 | 0.8170062 | 0.883247 | 0.9642173 | 1.870419 | 0.9971854 | 0.7204449 |
| cuda | technical | final_ln | C1 | 0.7750861 | 0.1008725 | 0.8759586 | 0.8994515 | 0.963856 | 1.632691 | 0.9960141 | 0.775086 |
| cuda | repetitive | final_ln | C1 | 0.9282393 | 0.06027142 | 0.9885107 | 0.9948533 | 0.9985469 | 1.155662 | 0.7980532 | 0.9282393 |
| cuda | narrative | final_ln | C4 | 0.4516107 | 0.2046368 | 0.6562475 | 0.752073 | 0.903522 | 3.835698 | 0.5907458 | 0.4516107 |
| cuda | technical | final_ln | C4 | 0.4561773 | 0.1785074 | 0.6346847 | 0.7191226 | 0.8857723 | 3.930734 | -0.6990266 | 0.4561773 |
| cuda | repetitive | final_ln | C4 | 0.8571362 | 0.1173713 | 0.9745075 | 0.9900365 | 0.9971315 | 1.335622 | 0.409039 | 0.8571362 |

## 点の分布

| 実行先 | 文章 | 点 | 処理 | occupancy | max_gap | occupancy90 | max_gap90 | max_pc1_share | max_pc1_position | first_pc1_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | narrative | block06 | C0 | 0.1 | 0.9838956 | 0.9 | 0.1574691 | 0.989483 | 0 | 0.989483 |
| cpu | technical | block06 | C0 | 0.1 | 0.9852295 | 0.9 | 0.1035946 | 0.9877736 | 0 | 0.9877736 |
| cpu | repetitive | block06 | C0 | 0.1 | 0.9906163 | 0.95 | 0.08963508 | 0.9858213 | 0 | 0.9858213 |
| cpu | narrative | block06 | C1 | 0.95 | 0.06638875 | 0.95 | 0.06421708 | 0.09024982 | 2 | NA |
| cpu | technical | block06 | C1 | 0.95 | 0.09821234 | 1 | 0.04736283 | 0.07178226 | 13 | NA |
| cpu | repetitive | block06 | C1 | 0.7 | 0.3234947 | 0.65 | 0.3662184 | 0.04689795 | 6 | NA |
| cpu | narrative | block06 | C4 | 1 | 0.05029365 | 1 | 0.0587801 | 0.03097272 | 6 | 0.0005655286 |
| cpu | technical | block06 | C4 | 1 | 0.08081283 | 0.9 | 0.09600158 | 0.04593871 | 84 | 0.0007234331 |
| cpu | repetitive | block06 | C4 | 0.6 | 0.4733022 | 0.55 | 0.5052325 | 0.02804529 | 34 | 0.006223802 |
| cpu | narrative | block07 | C0 | 0.1 | 0.9822424 | 0.85 | 0.1339907 | 0.9893788 | 0 | 0.9893788 |
| cpu | technical | block07 | C0 | 0.1 | 0.9852378 | 0.95 | 0.07721762 | 0.9876786 | 0 | 0.9876786 |
| cpu | repetitive | block07 | C0 | 0.1 | 0.9885722 | 1 | 0.05612372 | 0.9856691 | 0 | 0.9856691 |
| cpu | narrative | block07 | C1 | 0.95 | 0.1248973 | 1 | 0.06571762 | 0.09018887 | 2 | NA |
| cpu | technical | block07 | C1 | 0.9 | 0.08228879 | 0.9 | 0.1105821 | 0.1080114 | 13 | NA |
| cpu | repetitive | block07 | C1 | 0.7 | 0.1832876 | 0.75 | 0.1981371 | 0.0466315 | 6 | NA |
| cpu | narrative | block07 | C4 | 0.95 | 0.07254937 | 1 | 0.07667394 | 0.05797035 | 84 | 0.009520296 |
| cpu | technical | block07 | C4 | 1 | 0.06165004 | 0.95 | 0.07231537 | 0.07051426 | 13 | 0.009309586 |
| cpu | repetitive | block07 | C4 | 0.8 | 0.1319274 | 0.8 | 0.1432384 | 0.04378064 | 32 | 0.0004148544 |
| cpu | narrative | block12_preln | C0 | 0.75 | 0.2156388 | 1 | 0.09279379 | 0.1246124 | 0 | 0.1246124 |
| cpu | technical | block12_preln | C0 | 0.7 | 0.336658 | 1 | 0.06186955 | 0.1838718 | 0 | 0.1838718 |
| cpu | repetitive | block12_preln | C0 | 0.75 | 0.2617655 | 0.75 | 0.2873136 | 0.04923784 | 64 | 0.000972272 |
| cpu | narrative | block12_preln | C1 | 0.9 | 0.1084152 | 0.95 | 0.1228064 | 0.07327079 | 66 | NA |
| cpu | technical | block12_preln | C1 | 0.95 | 0.08688497 | 1 | 0.05761628 | 0.05887498 | 14 | NA |
| cpu | repetitive | block12_preln | C1 | 0.7 | 0.3076498 | 0.7 | 0.3377927 | 0.04950283 | 64 | NA |
| cpu | narrative | block12_preln | C4 | 0.65 | 0.3617879 | 1 | 0.04670154 | 0.2100156 | 74 | 0.006597306 |
| cpu | technical | block12_preln | C4 | 0.6 | 0.4629798 | 1 | 0.07196161 | 0.350317 | 0 | 0.350317 |
| cpu | repetitive | block12_preln | C4 | 0.65 | 0.3441873 | 0.65 | 0.359769 | 0.04427984 | 32 | 1.314177e-05 |
| cpu | narrative | final_ln | C0 | 0.8 | 0.1461831 | 1 | 0.05005686 | 0.1290569 | 66 | 0.07469191 |
| cpu | technical | final_ln | C0 | 0.85 | 0.1869758 | 0.95 | 0.09552374 | 0.1257593 | 0 | 0.1257593 |
| cpu | repetitive | final_ln | C0 | 0.65 | 0.2152494 | 0.65 | 0.2642084 | 0.08386372 | 58 | 0.004925562 |
| cpu | narrative | final_ln | C1 | 0.75 | 0.199966 | 1 | 0.05144308 | 0.1406128 | 66 | NA |
| cpu | technical | final_ln | C1 | 0.95 | 0.09329344 | 0.95 | 0.09121002 | 0.07737546 | 14 | NA |
| cpu | repetitive | final_ln | C1 | 0.65 | 0.2155424 | 0.6 | 0.2645407 | 0.08479689 | 58 | NA |
| cpu | narrative | final_ln | C4 | 0.5 | 0.2876189 | 0.95 | 0.06979695 | 0.416677 | 66 | 0.02809801 |
| cpu | technical | final_ln | C4 | 0.9 | 0.1498331 | 1 | 0.07750369 | 0.08514311 | 14 | 0.08064185 |
| cpu | repetitive | final_ln | C4 | 0.4 | 0.3642005 | 0.4 | 0.3695132 | 0.07863319 | 58 | 0.003792465 |
| cuda | narrative | block06 | C0 | 0.1 | 0.9838956 | 0.9 | 0.1574687 | 0.989483 | 0 | 0.989483 |
| cuda | technical | block06 | C0 | 0.1 | 0.9852295 | 0.9 | 0.1035948 | 0.9877736 | 0 | 0.9877736 |
| cuda | repetitive | block06 | C0 | 0.1 | 0.9906163 | 0.95 | 0.08963418 | 0.9858213 | 0 | 0.9858213 |
| cuda | narrative | block06 | C1 | 0.95 | 0.06638854 | 0.95 | 0.06421724 | 0.09024985 | 2 | NA |
| cuda | technical | block06 | C1 | 0.95 | 0.09821199 | 1 | 0.04736268 | 0.07178237 | 13 | NA |
| cuda | repetitive | block06 | C1 | 0.7 | 0.3234944 | 0.65 | 0.3662182 | 0.046898 | 6 | NA |
| cuda | narrative | block06 | C4 | 1 | 0.0502934 | 1 | 0.05877981 | 0.03097269 | 6 | 0.0005655255 |
| cuda | technical | block06 | C4 | 1 | 0.08081273 | 0.9 | 0.09600151 | 0.04593873 | 84 | 0.0007234345 |
| cuda | repetitive | block06 | C4 | 0.6 | 0.4733026 | 0.55 | 0.5052328 | 0.0280453 | 34 | 0.006223833 |
| cuda | narrative | block07 | C0 | 0.1 | 0.9822424 | 0.85 | 0.1339903 | 0.9893788 | 0 | 0.9893788 |
| cuda | technical | block07 | C0 | 0.1 | 0.9852378 | 0.95 | 0.07721872 | 0.9876786 | 0 | 0.9876786 |
| cuda | repetitive | block07 | C0 | 0.1 | 0.9885722 | 1 | 0.05612307 | 0.9856691 | 0 | 0.9856691 |
| cuda | narrative | block07 | C1 | 0.95 | 0.1248973 | 1 | 0.06571773 | 0.09018893 | 2 | NA |
| cuda | technical | block07 | C1 | 0.9 | 0.0822888 | 0.9 | 0.1105819 | 0.1080115 | 13 | NA |
| cuda | repetitive | block07 | C1 | 0.7 | 0.183288 | 0.75 | 0.1981375 | 0.04663152 | 6 | NA |
| cuda | narrative | block07 | C4 | 0.95 | 0.07254893 | 1 | 0.07667414 | 0.05797039 | 84 | 0.009520318 |
| cuda | technical | block07 | C4 | 1 | 0.06165005 | 0.95 | 0.07231565 | 0.07051421 | 13 | 0.009309581 |
| cuda | repetitive | block07 | C4 | 0.8 | 0.1319273 | 0.8 | 0.1432382 | 0.04378063 | 32 | 0.0004148547 |
| cuda | narrative | block12_preln | C0 | 0.75 | 0.2156378 | 1 | 0.09279363 | 0.124612 | 0 | 0.124612 |
| cuda | technical | block12_preln | C0 | 0.7 | 0.3366579 | 1 | 0.06186954 | 0.1838716 | 0 | 0.1838716 |
| cuda | repetitive | block12_preln | C0 | 0.75 | 0.2617648 | 0.75 | 0.2873129 | 0.04923782 | 64 | 0.000972275 |
| cuda | narrative | block12_preln | C1 | 0.9 | 0.1084153 | 0.95 | 0.1228065 | 0.07327081 | 66 | NA |
| cuda | technical | block12_preln | C1 | 0.95 | 0.08688472 | 1 | 0.05761608 | 0.05887497 | 14 | NA |
| cuda | repetitive | block12_preln | C1 | 0.7 | 0.3076493 | 0.7 | 0.3377922 | 0.0495028 | 64 | NA |
| cuda | narrative | block12_preln | C4 | 0.65 | 0.3617876 | 1 | 0.04670788 | 0.2100153 | 74 | 0.006597275 |
| cuda | technical | block12_preln | C4 | 0.6 | 0.4629793 | 1 | 0.07196079 | 0.3503165 | 0 | 0.3503165 |
| cuda | repetitive | block12_preln | C4 | 0.65 | 0.3441869 | 0.65 | 0.3597685 | 0.04427982 | 32 | 1.314194e-05 |
| cuda | narrative | final_ln | C0 | 0.8 | 0.146183 | 1 | 0.05005609 | 0.129057 | 66 | 0.07469169 |
| cuda | technical | final_ln | C0 | 0.85 | 0.1869755 | 0.95 | 0.09552429 | 0.1257591 | 0 | 0.1257591 |
| cuda | repetitive | final_ln | C0 | 0.65 | 0.2152491 | 0.65 | 0.264208 | 0.08386371 | 58 | 0.004925557 |
| cuda | narrative | final_ln | C1 | 0.75 | 0.1999661 | 1 | 0.05144236 | 0.1406127 | 66 | NA |
| cuda | technical | final_ln | C1 | 0.95 | 0.09329426 | 0.95 | 0.09121065 | 0.07737543 | 14 | NA |
| cuda | repetitive | final_ln | C1 | 0.65 | 0.2155421 | 0.6 | 0.2645403 | 0.08479688 | 58 | NA |
| cuda | narrative | final_ln | C4 | 0.5 | 0.2876191 | 0.95 | 0.06979668 | 0.416677 | 66 | 0.02809799 |
| cuda | technical | final_ln | C4 | 0.9 | 0.149833 | 1 | 0.07750391 | 0.08514311 | 14 | 0.08064182 |
| cuda | repetitive | final_ln | C4 | 0.4 | 0.3641949 | 0.4 | 0.3695075 | 0.07863313 | 58 | 0.003792468 |

## 2Dの曲線

| 実行先 | 文章 | 点 | 処理 | Q_line_2d | Q_curve_2d | curve_gain_2d | Q_insample_2d | graph_2d | curved_graph_2d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | narrative | block06 | C0 | 0.8104415 | -433.4379 | -434.2484 | 0.9979698 | False | False |
| cpu | technical | block06 | C0 | -0.1342936 | -1411.926 | -1411.792 | 0.9985025 | False | False |
| cpu | repetitive | block06 | C0 | 0.9949394 | -194374.4 | -194375.4 | 0.9967974 | False | False |
| cpu | narrative | block06 | C1 | 0.3691373 | 0.3364195 | -0.03271779 | 0.5346027 | False | False |
| cpu | technical | block06 | C1 | 0.3395086 | 0.3712883 | 0.0317797 | 0.5596983 | False | False |
| cpu | repetitive | block06 | C1 | 0.5512399 | 0.5750575 | 0.0238176 | 0.5858212 | False | False |
| cpu | narrative | block06 | C4 | 0.2421394 | 0.233593 | -0.008546448 | 0.5140675 | False | False |
| cpu | technical | block06 | C4 | 0.4249177 | 0.4652822 | 0.04036448 | 0.5561993 | False | False |
| cpu | repetitive | block06 | C4 | 0.5034978 | 0.6144587 | 0.1109609 | 0.6304517 | False | False |
| cpu | narrative | block07 | C0 | 0.6568833 | -1141.913 | -1142.57 | 0.9979771 | False | False |
| cpu | technical | block07 | C0 | -0.1246183 | -106.0922 | -105.9676 | 0.9982013 | False | False |
| cpu | repetitive | block07 | C0 | 0.8913494 | -21056.87 | -21057.76 | 0.9959267 | False | False |
| cpu | narrative | block07 | C1 | 0.4008427 | 0.4038252 | 0.002982491 | 0.5499491 | False | False |
| cpu | technical | block07 | C1 | 0.3520514 | 0.4372441 | 0.08519269 | 0.5887122 | False | False |
| cpu | repetitive | block07 | C1 | 0.541878 | 0.5663693 | 0.02449122 | 0.573421 | False | False |
| cpu | narrative | block07 | C4 | 0.349693 | 0.3248691 | -0.02482392 | 0.5133938 | False | False |
| cpu | technical | block07 | C4 | 0.4629202 | 0.6476957 | 0.1847755 | 0.6952637 | False | False |
| cpu | repetitive | block07 | C4 | 0.5003016 | 0.8025089 | 0.3022073 | 0.809608 | False | False |
| cpu | narrative | block12_preln | C0 | 0.5648896 | 0.642948 | 0.07805843 | 0.6843237 | False | False |
| cpu | technical | block12_preln | C0 | 0.7503982 | 0.8216662 | 0.071268 | 0.8346824 | False | False |
| cpu | repetitive | block12_preln | C0 | 0.810133 | 0.8462003 | 0.0360673 | 0.8580859 | False | False |
| cpu | narrative | block12_preln | C1 | 0.5914152 | 0.6041642 | 0.01274903 | 0.6550365 | False | False |
| cpu | technical | block12_preln | C1 | 0.7576327 | 0.7687251 | 0.01109245 | 0.7861486 | False | False |
| cpu | repetitive | block12_preln | C1 | 0.8150428 | 0.8666155 | 0.05157267 | 0.8763499 | False | False |
| cpu | narrative | block12_preln | C4 | 0.6138509 | 0.6345904 | 0.02073941 | 0.6730025 | False | False |
| cpu | technical | block12_preln | C4 | 0.208648 | 0.8765676 | 0.6679197 | 0.9651163 | False | False |
| cpu | repetitive | block12_preln | C4 | 0.8316243 | 0.9669313 | 0.135307 | 0.9695222 | False | False |
| cpu | narrative | final_ln | C0 | 0.8748373 | 0.8716052 | -0.003232081 | 0.8981333 | False | False |
| cpu | technical | final_ln | C0 | 0.8881836 | 0.888353 | 0.0001693867 | 0.8991285 | False | False |
| cpu | repetitive | final_ln | C0 | 0.930999 | 0.9837625 | 0.05276349 | 0.9849113 | False | False |
| cpu | narrative | final_ln | C1 | 0.8732226 | 0.8484677 | -0.02475488 | 0.8908365 | False | False |
| cpu | technical | final_ln | C1 | 0.8738181 | 0.8678928 | -0.005925295 | 0.8860454 | False | False |
| cpu | repetitive | final_ln | C1 | 0.9296786 | 0.985629 | 0.05595043 | 0.9867607 | False | False |
| cpu | narrative | final_ln | C4 | 0.6792598 | 0.6708137 | -0.00844611 | 0.6889252 | False | False |
| cpu | technical | final_ln | C4 | 0.7072146 | 0.7029819 | -0.00423277 | 0.7315115 | False | False |
| cpu | repetitive | final_ln | C4 | 0.8632148 | 0.9678468 | 0.104632 | 0.9744997 | False | False |
| cuda | narrative | block06 | C0 | 0.8104415 | -433.4365 | -434.247 | 0.9979698 | False | False |
| cuda | technical | block06 | C0 | -0.1342936 | -1411.934 | -1411.8 | 0.9985025 | False | False |
| cuda | repetitive | block06 | C0 | 0.9949393 | -194374.3 | -194375.3 | 0.9967974 | False | False |
| cuda | narrative | block06 | C1 | 0.3691374 | 0.3364196 | -0.03271778 | 0.5346027 | False | False |
| cuda | technical | block06 | C1 | 0.3395085 | 0.3712883 | 0.03177986 | 0.5596983 | False | False |
| cuda | repetitive | block06 | C1 | 0.5512399 | 0.5750577 | 0.02381781 | 0.5858214 | False | False |
| cuda | narrative | block06 | C4 | 0.2421395 | 0.2335931 | -0.008546411 | 0.5140674 | False | False |
| cuda | technical | block06 | C4 | 0.4249177 | 0.4652826 | 0.04036488 | 0.5561996 | False | False |
| cuda | repetitive | block06 | C4 | 0.5034978 | 0.6144562 | 0.1109584 | 0.6304493 | False | False |
| cuda | narrative | block07 | C0 | 0.6568836 | -1141.908 | -1142.565 | 0.9979771 | False | False |
| cuda | technical | block07 | C0 | -0.1246166 | -106.0927 | -105.9681 | 0.9982013 | False | False |
| cuda | repetitive | block07 | C0 | 0.89135 | -21056.83 | -21057.73 | 0.9959267 | False | False |
| cuda | narrative | block07 | C1 | 0.4008426 | 0.4038249 | 0.002982294 | 0.549949 | False | False |
| cuda | technical | block07 | C1 | 0.3520512 | 0.4372441 | 0.08519285 | 0.5887122 | False | False |
| cuda | repetitive | block07 | C1 | 0.541878 | 0.566369 | 0.02449097 | 0.5734208 | False | False |
| cuda | narrative | block07 | C4 | 0.3496931 | 0.3248692 | -0.02482394 | 0.5133938 | False | False |
| cuda | technical | block07 | C4 | 0.46292 | 0.6476959 | 0.1847759 | 0.6952638 | False | False |
| cuda | repetitive | block07 | C4 | 0.5003016 | 0.8025089 | 0.3022073 | 0.809608 | False | False |
| cuda | narrative | block12_preln | C0 | 0.5648897 | 0.6429476 | 0.07805797 | 0.6843234 | False | False |
| cuda | technical | block12_preln | C0 | 0.7503982 | 0.8216658 | 0.07126766 | 0.8346821 | False | False |
| cuda | repetitive | block12_preln | C0 | 0.8101331 | 0.8462004 | 0.03606727 | 0.8580859 | False | False |
| cuda | narrative | block12_preln | C1 | 0.5914153 | 0.6041641 | 0.01274884 | 0.6550363 | False | False |
| cuda | technical | block12_preln | C1 | 0.7576325 | 0.768725 | 0.0110925 | 0.7861485 | False | False |
| cuda | repetitive | block12_preln | C1 | 0.8150429 | 0.8666155 | 0.05157263 | 0.8763499 | False | False |
| cuda | narrative | block12_preln | C4 | 0.6138513 | 0.6345908 | 0.02073946 | 0.6730026 | False | False |
| cuda | technical | block12_preln | C4 | 0.2086484 | 0.876567 | 0.6679187 | 0.9651162 | False | False |
| cuda | repetitive | block12_preln | C4 | 0.8316243 | 0.9669313 | 0.135307 | 0.9695222 | False | False |
| cuda | narrative | final_ln | C0 | 0.8748373 | 0.8716053 | -0.003232009 | 0.8981333 | False | False |
| cuda | technical | final_ln | C0 | 0.8881836 | 0.8883529 | 0.0001693572 | 0.8991285 | False | False |
| cuda | repetitive | final_ln | C0 | 0.930999 | 0.9837625 | 0.05276349 | 0.9849113 | False | False |
| cuda | narrative | final_ln | C1 | 0.8732227 | 0.8484681 | -0.02475462 | 0.8908366 | False | False |
| cuda | technical | final_ln | C1 | 0.8738181 | 0.8678928 | -0.005925305 | 0.8860454 | False | False |
| cuda | repetitive | final_ln | C1 | 0.9296786 | 0.985629 | 0.05595043 | 0.9867607 | False | False |
| cuda | narrative | final_ln | C4 | 0.67926 | 0.6708138 | -0.008446103 | 0.6889253 | False | False |
| cuda | technical | final_ln | C4 | 0.7072143 | 0.7029813 | -0.00423296 | 0.7315111 | False | False |
| cuda | repetitive | final_ln | C4 | 0.8632149 | 0.967847 | 0.1046321 | 0.9744998 | False | False |

## 3Dの曲線

| 実行先 | 文章 | 点 | 処理 | Q_line_3d | Q_curve_3d | curve_gain_3d | Q_insample_3d | graph_3d | curved_graph_3d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | narrative | block06 | C0 | 0.8080016 | -11502.63 | -11503.44 | 0.9957716 | False | False |
| cpu | technical | block06 | C0 | -0.2236313 | -1807.323 | -1807.099 | 0.9966025 | False | False |
| cpu | repetitive | block06 | C0 | -4.853671 | -196301.3 | -196296.5 | 0.9947827 | False | False |
| cpu | narrative | block06 | C1 | 0.2432028 | 0.1372249 | -0.1059779 | 0.4015671 | False | False |
| cpu | technical | block06 | C1 | 0.1927159 | 0.2244136 | 0.0316977 | 0.4202943 | False | False |
| cpu | repetitive | block06 | C1 | 0.4133326 | 0.5527177 | 0.1393851 | 0.5669486 | False | False |
| cpu | narrative | block06 | C4 | 0.1538476 | 0.1513005 | -0.002547101 | 0.3839082 | False | False |
| cpu | technical | block06 | C4 | 0.2129419 | 0.2771699 | 0.06422798 | 0.4248541 | False | False |
| cpu | repetitive | block06 | C4 | 0.3813297 | 0.4694288 | 0.08809912 | 0.4827822 | False | False |
| cpu | narrative | block07 | C0 | 0.6523889 | -2929.67 | -2930.323 | 0.9954536 | False | False |
| cpu | technical | block07 | C0 | -0.138365 | -915.1803 | -915.042 | 0.9959855 | False | False |
| cpu | repetitive | block07 | C0 | -2.31264 | -99325.55 | -99323.24 | 0.9929098 | False | False |
| cpu | narrative | block07 | C1 | 0.2477653 | 0.1918291 | -0.05593617 | 0.406775 | False | False |
| cpu | technical | block07 | C1 | 0.1845359 | 0.2288559 | 0.04432002 | 0.4359403 | False | False |
| cpu | repetitive | block07 | C1 | 0.4120486 | 0.5799207 | 0.1678721 | 0.5952114 | False | False |
| cpu | narrative | block07 | C4 | 0.2236046 | 0.1880306 | -0.03557401 | 0.368186 | False | False |
| cpu | technical | block07 | C4 | 0.2029521 | 0.333573 | 0.1306209 | 0.5056021 | False | False |
| cpu | repetitive | block07 | C4 | 0.3831855 | 0.6739452 | 0.2907596 | 0.6828581 | False | False |
| cpu | narrative | block12_preln | C0 | 0.4499321 | 0.5160549 | 0.06612279 | 0.5762364 | False | False |
| cpu | technical | block12_preln | C0 | 0.6405309 | 0.7081016 | 0.06757063 | 0.7247147 | False | False |
| cpu | repetitive | block12_preln | C0 | 0.7818286 | 0.8378813 | 0.05605277 | 0.8507975 | False | False |
| cpu | narrative | block12_preln | C1 | 0.4731127 | 0.4831275 | 0.01001482 | 0.5414793 | False | False |
| cpu | technical | block12_preln | C1 | 0.6323218 | 0.6417287 | 0.009406933 | 0.666904 | False | False |
| cpu | repetitive | block12_preln | C1 | 0.7890257 | 0.8562685 | 0.06724284 | 0.8671914 | False | False |
| cpu | narrative | block12_preln | C4 | 0.4748436 | 0.4934513 | 0.01860773 | 0.5400387 | False | False |
| cpu | technical | block12_preln | C4 | 0.1234551 | 0.6692934 | 0.5458383 | 0.7911367 | False | False |
| cpu | repetitive | block12_preln | C4 | 0.7991184 | 0.935177 | 0.1360586 | 0.9394911 | False | False |
| cpu | narrative | final_ln | C0 | 0.8086366 | 0.8106669 | 0.002030287 | 0.8400139 | False | False |
| cpu | technical | final_ln | C0 | 0.8619536 | 0.8690783 | 0.007124738 | 0.8825207 | False | False |
| cpu | repetitive | final_ln | C0 | 0.9236075 | 0.9757084 | 0.05210081 | 0.9784817 | False | False |
| cpu | narrative | final_ln | C1 | 0.8045203 | 0.778461 | -0.02605924 | 0.8312447 | False | False |
| cpu | technical | final_ln | C1 | 0.85037 | 0.8449246 | -0.005445383 | 0.8635793 | False | False |
| cpu | repetitive | final_ln | C1 | 0.9227205 | 0.9777651 | 0.05504459 | 0.9805614 | False | False |
| cpu | narrative | final_ln | C4 | 0.5773043 | 0.5546333 | -0.022671 | 0.6330923 | False | False |
| cpu | technical | final_ln | C4 | 0.599026 | 0.6386297 | 0.03960369 | 0.6784883 | False | False |
| cpu | repetitive | final_ln | C4 | 0.8466846 | 0.9488389 | 0.1021543 | 0.9597937 | False | False |
| cuda | narrative | block06 | C0 | 0.8080017 | -11502.63 | -11503.43 | 0.9957716 | False | False |
| cuda | technical | block06 | C0 | -0.2236309 | -1807.332 | -1807.108 | 0.9966025 | False | False |
| cuda | repetitive | block06 | C0 | -4.853682 | -196301.3 | -196296.4 | 0.9947827 | False | False |
| cuda | narrative | block06 | C1 | 0.2432028 | 0.1372251 | -0.1059777 | 0.4015671 | False | False |
| cuda | technical | block06 | C1 | 0.1927158 | 0.2244136 | 0.0316978 | 0.4202944 | False | False |
| cuda | repetitive | block06 | C1 | 0.4133326 | 0.5527178 | 0.1393852 | 0.5669487 | False | False |
| cuda | narrative | block06 | C4 | 0.1538476 | 0.1513006 | -0.002547062 | 0.3839082 | False | False |
| cuda | technical | block06 | C4 | 0.2129419 | 0.2771702 | 0.06422827 | 0.4248543 | False | False |
| cuda | repetitive | block06 | C4 | 0.3813297 | 0.4694268 | 0.08809709 | 0.4827803 | False | False |
| cuda | narrative | block07 | C0 | 0.6523891 | -2929.67 | -2930.323 | 0.9954536 | False | False |
| cuda | technical | block07 | C0 | -0.1383635 | -915.1763 | -915.0379 | 0.9959855 | False | False |
| cuda | repetitive | block07 | C0 | -2.312643 | -99325.26 | -99322.95 | 0.9929098 | False | False |
| cuda | narrative | block07 | C1 | 0.2477652 | 0.1918289 | -0.05593626 | 0.4067749 | False | False |
| cuda | technical | block07 | C1 | 0.1845358 | 0.2288558 | 0.04432002 | 0.4359403 | False | False |
| cuda | repetitive | block07 | C1 | 0.4120487 | 0.5799206 | 0.167872 | 0.5952113 | False | False |
| cuda | narrative | block07 | C4 | 0.2236047 | 0.1880306 | -0.03557404 | 0.368186 | False | False |
| cuda | technical | block07 | C4 | 0.202952 | 0.3335732 | 0.1306212 | 0.5056022 | False | False |
| cuda | repetitive | block07 | C4 | 0.3831855 | 0.6739452 | 0.2907596 | 0.6828581 | False | False |
| cuda | narrative | block12_preln | C0 | 0.4499322 | 0.5160546 | 0.06612241 | 0.5762361 | False | False |
| cuda | technical | block12_preln | C0 | 0.640531 | 0.7081012 | 0.06757026 | 0.7247145 | False | False |
| cuda | repetitive | block12_preln | C0 | 0.7818287 | 0.8378814 | 0.05605276 | 0.8507975 | False | False |
| cuda | narrative | block12_preln | C1 | 0.4731128 | 0.4831274 | 0.01001466 | 0.5414791 | False | False |
| cuda | technical | block12_preln | C1 | 0.6323216 | 0.6417286 | 0.009406958 | 0.6669039 | False | False |
| cuda | repetitive | block12_preln | C1 | 0.7890258 | 0.8562686 | 0.06724281 | 0.8671914 | False | False |
| cuda | narrative | block12_preln | C4 | 0.4748438 | 0.4934516 | 0.01860776 | 0.5400388 | False | False |
| cuda | technical | block12_preln | C4 | 0.1234554 | 0.6692928 | 0.5458374 | 0.7911366 | False | False |
| cuda | repetitive | block12_preln | C4 | 0.7991185 | 0.935177 | 0.1360585 | 0.9394911 | False | False |
| cuda | narrative | final_ln | C0 | 0.8086367 | 0.810667 | 0.002030297 | 0.8400139 | False | False |
| cuda | technical | final_ln | C0 | 0.8619535 | 0.8690783 | 0.007124733 | 0.8825207 | False | False |
| cuda | repetitive | final_ln | C0 | 0.9236075 | 0.9757083 | 0.05210081 | 0.9784817 | False | False |
| cuda | narrative | final_ln | C1 | 0.8045204 | 0.7784613 | -0.02605907 | 0.8312447 | False | False |
| cuda | technical | final_ln | C1 | 0.85037 | 0.8449246 | -0.005445388 | 0.8635793 | False | False |
| cuda | repetitive | final_ln | C1 | 0.9227205 | 0.977765 | 0.05504458 | 0.9805614 | False | False |
| cuda | narrative | final_ln | C4 | 0.5773044 | 0.5546334 | -0.02267096 | 0.6330925 | False | False |
| cuda | technical | final_ln | C4 | 0.5990258 | 0.6386293 | 0.03960354 | 0.6784879 | False | False |
| cuda | repetitive | final_ln | C4 | 0.8466847 | 0.9488391 | 0.1021544 | 0.9597937 | False | False |

## 全空間の曲線

| 実行先 | 文章 | 点 | 処理 | Q_line_full | Q_curve_full | curve_gain_full | Q_insample_full | graph_full | curved_graph_full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | narrative | block06 | C0 | -0.829329 | -63230.14 | -63229.31 | 0.9681189 | False | False |
| cpu | technical | block06 | C0 | -1.618077 | -75371.77 | -75370.15 | 0.9721526 | False | False |
| cpu | repetitive | block06 | C0 | -6.092524 | -522480.2 | -522474.1 | 0.9826951 | False | False |
| cpu | narrative | block06 | C1 | -0.03632737 | -0.09482134 | -0.05849397 | 0.08863523 | False | False |
| cpu | technical | block06 | C1 | -0.01832083 | -0.01951545 | -0.001194614 | 0.09947052 | False | False |
| cpu | repetitive | block06 | C1 | 0.09292027 | 0.1470687 | 0.05414847 | 0.2419639 | False | False |
| cpu | narrative | block06 | C4 | -0.0181456 | -0.01529055 | 0.002855045 | 0.09121374 | False | False |
| cpu | technical | block06 | C4 | -0.01529647 | -0.01969632 | -0.004399856 | 0.09216103 | False | False |
| cpu | repetitive | block06 | C4 | 0.0867225 | 0.113633 | 0.02691054 | 0.2112783 | False | False |
| cpu | narrative | block07 | C0 | -1.11903 | -52536.92 | -52535.8 | 0.9620329 | False | False |
| cpu | technical | block07 | C0 | -1.67597 | -84591.77 | -84590.09 | 0.9671207 | False | False |
| cpu | repetitive | block07 | C0 | -3.514588 | -266432.9 | -266429.4 | 0.9775969 | False | False |
| cpu | narrative | block07 | C1 | -0.04453539 | -0.09992223 | -0.05538684 | 0.08663685 | False | False |
| cpu | technical | block07 | C1 | -0.03206498 | -0.04009722 | -0.00803224 | 0.1000583 | False | False |
| cpu | repetitive | block07 | C1 | 0.09743868 | 0.1631497 | 0.06571106 | 0.250931 | False | False |
| cpu | narrative | block07 | C4 | -0.04578734 | -0.07534835 | -0.02956102 | 0.07920427 | False | False |
| cpu | technical | block07 | C4 | -0.03737752 | -0.03412676 | 0.003250759 | 0.1058333 | False | False |
| cpu | repetitive | block07 | C4 | 0.09795861 | 0.1978218 | 0.09986322 | 0.2855738 | False | False |
| cpu | narrative | block12_preln | C0 | 0.1213306 | 0.1226042 | 0.001273636 | 0.2429456 | False | False |
| cpu | technical | block12_preln | C0 | 0.2160079 | 0.1242263 | -0.09178165 | 0.3532534 | False | False |
| cpu | repetitive | block12_preln | C0 | 0.6720134 | 0.7176726 | 0.04565924 | 0.7439668 | False | False |
| cpu | narrative | block12_preln | C1 | 0.1264374 | 0.1310078 | 0.004570412 | 0.2247919 | False | False |
| cpu | technical | block12_preln | C1 | 0.2061505 | 0.2065554 | 0.0004049765 | 0.3103892 | False | False |
| cpu | repetitive | block12_preln | C1 | 0.6904093 | 0.7462587 | 0.05584943 | 0.7707832 | False | False |
| cpu | narrative | block12_preln | C4 | 0.09880975 | -0.01141718 | -0.1102269 | 0.2050135 | False | False |
| cpu | technical | block12_preln | C4 | -0.02636758 | -0.6193828 | -0.5930152 | 0.2798529 | False | False |
| cpu | repetitive | block12_preln | C4 | 0.6822882 | 0.7948673 | 0.1125791 | 0.8144511 | False | False |
| cpu | narrative | final_ln | C0 | 0.7096161 | 0.7138995 | 0.004283413 | 0.7537266 | False | False |
| cpu | technical | final_ln | C0 | 0.7729624 | 0.7770744 | 0.004111993 | 0.8010384 | False | False |
| cpu | repetitive | final_ln | C0 | 0.9184835 | 0.9701535 | 0.05167002 | 0.9734164 | False | False |
| cpu | narrative | final_ln | C1 | 0.7019285 | 0.6737657 | -0.02816276 | 0.742616 | False | False |
| cpu | technical | final_ln | C1 | 0.7549499 | 0.7495625 | -0.005387401 | 0.7808129 | False | False |
| cpu | repetitive | final_ln | C1 | 0.9177133 | 0.9722955 | 0.0545822 | 0.9755973 | False | False |
| cpu | narrative | final_ln | C4 | 0.3972318 | 0.3796796 | -0.01755221 | 0.4995409 | False | False |
| cpu | technical | final_ln | C4 | 0.4059575 | 0.4204785 | 0.01452099 | 0.5051142 | False | False |
| cpu | repetitive | final_ln | C4 | 0.8375513 | 0.938706 | 0.1011547 | 0.9504477 | False | False |
| cuda | narrative | block06 | C0 | -0.829328 | -63230.05 | -63229.22 | 0.9681188 | False | False |
| cuda | technical | block06 | C0 | -1.618077 | -75371.78 | -75370.16 | 0.9721526 | False | False |
| cuda | repetitive | block06 | C0 | -6.092532 | -522480.4 | -522474.3 | 0.9826952 | False | False |
| cuda | narrative | block06 | C1 | -0.03632735 | -0.09482126 | -0.05849391 | 0.08863523 | False | False |
| cuda | technical | block06 | C1 | -0.01832087 | -0.01951552 | -0.001194646 | 0.09947051 | False | False |
| cuda | repetitive | block06 | C1 | 0.09292029 | 0.1470688 | 0.05414848 | 0.241964 | False | False |
| cuda | narrative | block06 | C4 | -0.01814559 | -0.01529051 | 0.002855075 | 0.09121374 | False | False |
| cuda | technical | block06 | C4 | -0.01529648 | -0.01969633 | -0.004399846 | 0.09216105 | False | False |
| cuda | repetitive | block06 | C4 | 0.08672247 | 0.1136323 | 0.02690983 | 0.2112775 | False | False |
| cuda | narrative | block07 | C0 | -1.119029 | -52536.83 | -52535.71 | 0.9620329 | False | False |
| cuda | technical | block07 | C0 | -1.67597 | -84591.71 | -84590.04 | 0.9671207 | False | False |
| cuda | repetitive | block07 | C0 | -3.51459 | -266432.8 | -266429.3 | 0.9775969 | False | False |
| cuda | narrative | block07 | C1 | -0.04453536 | -0.09992214 | -0.05538678 | 0.08663684 | False | False |
| cuda | technical | block07 | C1 | -0.03206503 | -0.04009733 | -0.0080323 | 0.1000583 | False | False |
| cuda | repetitive | block07 | C1 | 0.09743868 | 0.1631497 | 0.06571101 | 0.2509309 | False | False |
| cuda | narrative | block07 | C4 | -0.04578734 | -0.07534836 | -0.02956102 | 0.07920428 | False | False |
| cuda | technical | block07 | C4 | -0.03737753 | -0.03412678 | 0.003250751 | 0.1058333 | False | False |
| cuda | repetitive | block07 | C4 | 0.0979586 | 0.1978218 | 0.09986322 | 0.2855738 | False | False |
| cuda | narrative | block12_preln | C0 | 0.1213307 | 0.1226045 | 0.001273765 | 0.2429455 | False | False |
| cuda | technical | block12_preln | C0 | 0.216008 | 0.1242268 | -0.09178126 | 0.3532533 | False | False |
| cuda | repetitive | block12_preln | C0 | 0.6720135 | 0.7176727 | 0.04565924 | 0.7439668 | False | False |
| cuda | narrative | block12_preln | C1 | 0.1264374 | 0.1310078 | 0.004570402 | 0.2247918 | False | False |
| cuda | technical | block12_preln | C1 | 0.2061504 | 0.2065554 | 0.0004049895 | 0.3103892 | False | False |
| cuda | repetitive | block12_preln | C1 | 0.6904094 | 0.7462588 | 0.05584943 | 0.7707832 | False | False |
| cuda | narrative | block12_preln | C4 | 0.09880983 | -0.01141654 | -0.1102264 | 0.2050135 | False | False |
| cuda | technical | block12_preln | C4 | -0.02636743 | -0.6193792 | -0.5930118 | 0.2798529 | False | False |
| cuda | repetitive | block12_preln | C4 | 0.6822882 | 0.7948673 | 0.1125791 | 0.8144511 | False | False |
| cuda | narrative | final_ln | C0 | 0.7096162 | 0.7138996 | 0.004283425 | 0.7537266 | False | False |
| cuda | technical | final_ln | C0 | 0.7729624 | 0.7770744 | 0.004111988 | 0.8010384 | False | False |
| cuda | repetitive | final_ln | C0 | 0.9184835 | 0.9701535 | 0.05167002 | 0.9734164 | False | False |
| cuda | narrative | final_ln | C1 | 0.7019286 | 0.673766 | -0.02816257 | 0.742616 | False | False |
| cuda | technical | final_ln | C1 | 0.7549499 | 0.7495625 | -0.005387409 | 0.7808129 | False | False |
| cuda | repetitive | final_ln | C1 | 0.9177133 | 0.9722955 | 0.0545822 | 0.9755972 | False | False |
| cuda | narrative | final_ln | C4 | 0.3972319 | 0.3796798 | -0.01755215 | 0.499541 | False | False |
| cuda | technical | final_ln | C4 | 0.4059574 | 0.4204783 | 0.01452092 | 0.505114 | False | False |
| cuda | repetitive | final_ln | C4 | 0.8375514 | 0.9387062 | 0.1011548 | 0.9504477 | False | False |

## 厚みの全相関

二次平均を引いたPC2〜8。surprisalだけ位置0を除く。有意性判定は行わない。

| 実行先 | 文章 | 点 | 処理 | 特徴 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | narrative | block06 | C0 | position | -0.679357 | -0.567614 | -0.074328 | +0.287681 | +0.026768 | +0.042135 | +0.081576 |
| cpu | narrative | block06 | C0 | log_norm | +0.142889 | -0.150702 | -0.030035 | -0.049069 | -0.041886 | +0.012798 | -0.048079 |
| cpu | narrative | block06 | C0 | log_chars | +0.487616 | -0.359356 | +0.028954 | -0.210441 | -0.278246 | +0.170851 | -0.040551 |
| cpu | narrative | block06 | C0 | surprisal | +0.406214 | -0.260124 | -0.012169 | -0.234419 | -0.204807 | +0.114068 | -0.023401 |
| cpu | technical | block06 | C0 | position | -0.154027 | -0.779808 | -0.454301 | -0.123012 | +0.015629 | -0.143203 | -0.234793 |
| cpu | technical | block06 | C0 | log_norm | -0.131340 | +0.141083 | -0.084955 | -0.018464 | +0.056726 | -0.017770 | -0.004044 |
| cpu | technical | block06 | C0 | log_chars | -0.592810 | +0.025948 | -0.071628 | -0.119174 | +0.195139 | -0.015864 | +0.119546 |
| cpu | technical | block06 | C0 | surprisal | -0.422895 | +0.338653 | -0.126076 | -0.103803 | +0.150059 | -0.113583 | +0.118656 |
| cpu | repetitive | block06 | C0 | position | -0.026543 | -0.240952 | -0.063046 | -0.861955 | +0.042496 | +0.118989 | -0.019539 |
| cpu | repetitive | block06 | C0 | log_norm | +0.059935 | -0.075615 | +0.045674 | +0.011290 | +0.015972 | -0.056901 | +0.021098 |
| cpu | repetitive | block06 | C0 | log_chars | -0.189136 | -0.630456 | +0.517405 | -0.125987 | +0.079883 | -0.404173 | +0.087518 |
| cpu | repetitive | block06 | C0 | surprisal | -0.369425 | -0.463574 | +0.041131 | +0.003824 | -0.150725 | +0.208422 | -0.146816 |
| cpu | narrative | block06 | C1 | position | -0.563615 | +0.226432 | -0.173904 | -0.065502 | +0.016001 | +0.019275 | +0.003603 |
| cpu | narrative | block06 | C1 | log_norm | -0.521839 | +0.170932 | +0.161084 | -0.095332 | -0.095180 | -0.118161 | +0.142193 |
| cpu | narrative | block06 | C1 | log_chars | -0.367766 | -0.115909 | +0.136927 | -0.152286 | -0.281138 | -0.080902 | +0.016870 |
| cpu | narrative | block06 | C1 | surprisal | -0.269010 | -0.086341 | +0.166529 | -0.059140 | -0.200613 | -0.066644 | +0.057915 |
| cpu | technical | block06 | C1 | position | -0.743770 | -0.490698 | -0.130307 | +0.025606 | -0.147304 | -0.285218 | -0.023733 |
| cpu | technical | block06 | C1 | log_norm | +0.516097 | -0.078605 | -0.045698 | +0.292715 | -0.123204 | +0.161028 | -0.106287 |
| cpu | technical | block06 | C1 | log_chars | +0.103786 | +0.015284 | -0.128681 | +0.193811 | -0.003224 | +0.066331 | -0.189332 |
| cpu | technical | block06 | C1 | surprisal | +0.385777 | +0.024436 | -0.110364 | +0.165178 | -0.114274 | +0.117721 | +0.007262 |
| cpu | repetitive | block06 | C1 | position | -0.039350 | +0.006487 | -0.949853 | -0.077993 | +0.107070 | -0.004287 | -0.107807 |
| cpu | repetitive | block06 | C1 | log_norm | -0.592303 | -0.042013 | +0.254555 | +0.230509 | -0.170563 | +0.051902 | -0.129415 |
| cpu | repetitive | block06 | C1 | log_chars | -0.519336 | +0.176455 | -0.096800 | -0.011818 | -0.376627 | +0.066358 | +0.204901 |
| cpu | repetitive | block06 | C1 | surprisal | -0.557761 | -0.046410 | +0.087353 | +0.301456 | +0.140497 | -0.133240 | +0.056676 |
| cpu | narrative | block06 | C4 | position | -0.907801 | +0.182290 | +0.158535 | +0.071646 | -0.079242 | +0.036995 | -0.042796 |
| cpu | narrative | block06 | C4 | log_norm | +0.135339 | +0.274958 | +0.149411 | +0.166847 | +0.100671 | +0.004822 | +0.144623 |
| cpu | narrative | block06 | C4 | log_chars | +0.075109 | -0.231171 | +0.095173 | +0.204445 | +0.011514 | -0.081207 | +0.157887 |
| cpu | narrative | block06 | C4 | surprisal | +0.128634 | -0.058778 | -0.015236 | +0.128504 | +0.055282 | -0.145305 | +0.077025 |
| cpu | technical | block06 | C4 | position | +0.425335 | -0.649716 | +0.054242 | -0.109814 | +0.204404 | -0.217858 | -0.106765 |
| cpu | technical | block06 | C4 | log_norm | +0.121342 | +0.247846 | +0.112081 | +0.093489 | +0.194054 | -0.056011 | +0.254748 |
| cpu | technical | block06 | C4 | log_chars | +0.596008 | +0.127361 | +0.123830 | +0.049185 | -0.208026 | +0.168034 | -0.226604 |
| cpu | technical | block06 | C4 | surprisal | +0.338903 | +0.175329 | +0.125937 | +0.088378 | -0.041644 | +0.097100 | +0.051503 |
| cpu | repetitive | block06 | C4 | position | +0.061099 | +0.041055 | -0.952684 | -0.080506 | +0.099091 | -0.177686 | -0.003749 |
| cpu | repetitive | block06 | C4 | log_norm | -0.222774 | +0.018583 | +0.165118 | +0.341007 | +0.093789 | -0.199085 | +0.043588 |
| cpu | repetitive | block06 | C4 | log_chars | -0.497921 | +0.592700 | -0.122442 | -0.139145 | -0.132687 | +0.146522 | +0.087157 |
| cpu | repetitive | block06 | C4 | surprisal | -0.445743 | +0.057169 | +0.058184 | +0.249312 | +0.095473 | -0.047077 | +0.049484 |
| cpu | narrative | block07 | C0 | position | -0.750178 | -0.415762 | -0.311938 | +0.188825 | +0.024229 | +0.032847 | -0.039061 |
| cpu | narrative | block07 | C0 | log_norm | +0.097521 | -0.156953 | -0.025486 | -0.060385 | -0.043051 | +0.031118 | -0.028978 |
| cpu | narrative | block07 | C0 | log_chars | +0.359366 | -0.440782 | +0.115732 | -0.107006 | -0.162487 | +0.316028 | -0.050892 |
| cpu | narrative | block07 | C0 | surprisal | +0.323988 | -0.305603 | +0.090363 | -0.190566 | -0.171909 | +0.237093 | -0.007410 |
| cpu | technical | block07 | C0 | position | +0.644503 | +0.523403 | -0.541830 | -0.099028 | +0.020372 | +0.141041 | +0.151937 |
| cpu | technical | block07 | C0 | log_norm | -0.003883 | -0.163604 | -0.064326 | -0.017367 | -0.039037 | +0.014691 | -0.012027 |
| cpu | technical | block07 | C0 | log_chars | +0.454427 | -0.330500 | -0.235467 | -0.199499 | -0.257053 | -0.058403 | -0.183962 |
| cpu | technical | block07 | C0 | surprisal | +0.097212 | -0.486955 | -0.232601 | -0.134715 | -0.174134 | +0.063993 | -0.160138 |
| cpu | repetitive | block07 | C0 | position | -0.006806 | +0.176421 | -0.124754 | -0.829333 | +0.088248 | +0.019000 | -0.102070 |
| cpu | repetitive | block07 | C0 | log_norm | +0.059695 | +0.057898 | -0.049988 | +0.041674 | +0.002714 | -0.041649 | -0.013344 |
| cpu | repetitive | block07 | C0 | log_chars | -0.056630 | +0.590377 | -0.564234 | -0.009760 | -0.170232 | -0.424455 | +0.109668 |
| cpu | repetitive | block07 | C0 | surprisal | -0.301991 | +0.469694 | -0.111665 | -0.049504 | -0.089406 | +0.251892 | +0.014089 |
| cpu | narrative | block07 | C1 | position | -0.408343 | +0.332325 | +0.163251 | -0.071491 | +0.015197 | +0.014787 | -0.025759 |
| cpu | narrative | block07 | C1 | log_norm | -0.561468 | +0.170442 | -0.220693 | -0.161862 | +0.152536 | -0.133753 | -0.041990 |
| cpu | narrative | block07 | C1 | log_chars | -0.439606 | -0.093614 | -0.092464 | -0.082763 | +0.347689 | -0.085621 | -0.025622 |
| cpu | narrative | block07 | C1 | surprisal | -0.320796 | -0.117611 | -0.182365 | -0.079973 | +0.263434 | -0.070008 | -0.002936 |
| cpu | technical | block07 | C1 | position | -0.658985 | -0.568818 | -0.151657 | +0.008418 | +0.150378 | +0.174710 | -0.112310 |
| cpu | technical | block07 | C1 | log_norm | +0.582865 | +0.028939 | -0.051824 | -0.219464 | +0.093793 | -0.261292 | +0.041732 |
| cpu | technical | block07 | C1 | log_chars | +0.300798 | -0.063685 | -0.225903 | -0.271573 | -0.052773 | -0.216025 | -0.098902 |
| cpu | technical | block07 | C1 | surprisal | +0.497371 | +0.004347 | -0.162884 | -0.200758 | +0.080069 | -0.217035 | +0.091057 |
| cpu | repetitive | block07 | C1 | position | +0.014793 | -0.410168 | -0.922255 | -0.080696 | +0.054461 | -0.107880 | -0.003320 |
| cpu | repetitive | block07 | C1 | log_norm | +0.555679 | -0.151618 | +0.393039 | +0.304972 | -0.027668 | -0.095891 | -0.116356 |
| cpu | repetitive | block07 | C1 | log_chars | +0.711465 | -0.530829 | -0.043350 | +0.067628 | -0.406209 | +0.097945 | +0.051794 |
| cpu | repetitive | block07 | C1 | surprisal | +0.554569 | -0.366191 | +0.109314 | +0.314555 | +0.205870 | +0.016095 | +0.135910 |
| cpu | narrative | block07 | C4 | position | +0.573325 | +0.221139 | +0.141503 | -0.108700 | -0.059286 | -0.014930 | -0.003523 |
| cpu | narrative | block07 | C4 | log_norm | -0.293781 | +0.231694 | +0.031919 | -0.117060 | +0.141792 | +0.022307 | +0.169794 |
| cpu | narrative | block07 | C4 | log_chars | -0.614458 | -0.146199 | -0.043483 | -0.282842 | +0.123881 | +0.034234 | +0.210589 |
| cpu | narrative | block07 | C4 | surprisal | -0.534085 | -0.020337 | -0.135729 | -0.172712 | +0.113451 | +0.099722 | +0.055828 |
| cpu | technical | block07 | C4 | position | -0.212391 | -0.753807 | +0.022830 | +0.089593 | +0.150612 | +0.002199 | -0.233028 |
| cpu | technical | block07 | C4 | log_norm | -0.194976 | +0.192225 | +0.177683 | -0.119375 | +0.167786 | +0.204595 | +0.127270 |
| cpu | technical | block07 | C4 | log_chars | -0.498960 | +0.045262 | +0.269323 | -0.170752 | -0.115363 | -0.118559 | -0.106003 |
| cpu | technical | block07 | C4 | surprisal | -0.436459 | +0.188865 | +0.219260 | -0.117843 | +0.059095 | +0.091960 | +0.065046 |
| cpu | repetitive | block07 | C4 | position | -0.144609 | -0.175812 | -0.917914 | -0.206911 | +0.050087 | -0.181313 | -0.046667 |
| cpu | repetitive | block07 | C4 | log_norm | +0.187070 | -0.090262 | +0.167889 | +0.384342 | +0.103344 | -0.167702 | -0.040827 |
| cpu | repetitive | block07 | C4 | log_chars | +0.708265 | -0.769386 | -0.070830 | +0.021548 | -0.270639 | +0.160464 | -0.004972 |
| cpu | repetitive | block07 | C4 | surprisal | +0.358243 | -0.343696 | +0.091597 | +0.322888 | +0.034898 | -0.108375 | -0.018124 |
| cpu | narrative | block12_preln | C0 | position | +0.015386 | +0.237284 | +0.199225 | -0.765882 | -0.159506 | -0.074580 | +0.110066 |
| cpu | narrative | block12_preln | C0 | log_norm | +0.426207 | +0.041826 | -0.055924 | +0.091397 | -0.028156 | +0.161432 | +0.219688 |
| cpu | narrative | block12_preln | C0 | log_chars | -0.110078 | +0.259354 | -0.172433 | +0.118936 | +0.144119 | -0.069717 | -0.154615 |
| cpu | narrative | block12_preln | C0 | surprisal | -0.159825 | -0.045928 | -0.069808 | +0.048794 | -0.044436 | +0.081167 | -0.242191 |
| cpu | technical | block12_preln | C0 | position | -0.250851 | +0.211143 | +0.300947 | -0.248687 | +0.471139 | -0.428266 | +0.000166 |
| cpu | technical | block12_preln | C0 | log_norm | -0.201639 | +0.010360 | +0.127941 | +0.164223 | -0.023068 | +0.100696 | -0.007520 |
| cpu | technical | block12_preln | C0 | log_chars | -0.049779 | +0.378346 | +0.425113 | -0.095508 | -0.207283 | -0.116721 | -0.362457 |
| cpu | technical | block12_preln | C0 | surprisal | +0.055501 | +0.211674 | +0.059656 | -0.021072 | -0.055747 | +0.003813 | -0.140476 |
| cpu | repetitive | block12_preln | C0 | position | -0.051124 | +0.106572 | +0.085082 | -0.365508 | -0.574379 | -0.315925 | -0.238841 |
| cpu | repetitive | block12_preln | C0 | log_norm | +0.698042 | -0.372826 | +0.144883 | +0.213161 | -0.147861 | +0.010379 | +0.043128 |
| cpu | repetitive | block12_preln | C0 | log_chars | +0.034348 | +0.147002 | +0.363047 | -0.231282 | -0.190021 | +0.239730 | -0.349682 |
| cpu | repetitive | block12_preln | C0 | surprisal | -0.092326 | +0.261799 | +0.022352 | -0.226385 | +0.087161 | -0.083588 | -0.330871 |
| cpu | narrative | block12_preln | C1 | position | -0.031148 | +0.170692 | +0.059856 | -0.767053 | -0.110880 | +0.226883 | -0.053055 |
| cpu | narrative | block12_preln | C1 | log_norm | +0.595114 | +0.190794 | +0.074169 | +0.018360 | -0.012548 | +0.071710 | -0.138621 |
| cpu | narrative | block12_preln | C1 | log_chars | -0.208039 | +0.231996 | -0.253366 | +0.174295 | +0.159356 | +0.014670 | -0.310618 |
| cpu | narrative | block12_preln | C1 | surprisal | -0.239360 | -0.092121 | -0.128436 | +0.079418 | -0.037278 | -0.209649 | +0.089773 |
| cpu | technical | block12_preln | C1 | position | -0.265588 | +0.260107 | +0.364492 | +0.401460 | -0.491234 | -0.018131 | -0.096511 |
| cpu | technical | block12_preln | C1 | log_norm | -0.231126 | +0.048119 | -0.148517 | +0.023796 | +0.036159 | +0.026493 | +0.042021 |
| cpu | technical | block12_preln | C1 | log_chars | -0.040560 | +0.408503 | +0.196394 | -0.185420 | -0.186427 | -0.336226 | +0.097304 |
| cpu | technical | block12_preln | C1 | surprisal | +0.076829 | +0.206617 | +0.055603 | -0.071465 | +0.014525 | -0.145341 | +0.015230 |
| cpu | repetitive | block12_preln | C1 | position | -0.118639 | +0.227732 | +0.171701 | -0.631842 | -0.351919 | -0.253354 | +0.047448 |
| cpu | repetitive | block12_preln | C1 | log_norm | +0.741002 | -0.553012 | +0.009812 | -0.112750 | -0.004733 | +0.036879 | +0.068062 |
| cpu | repetitive | block12_preln | C1 | log_chars | +0.015354 | +0.040118 | +0.408275 | -0.247487 | +0.234099 | -0.340439 | +0.075554 |
| cpu | repetitive | block12_preln | C1 | surprisal | -0.127545 | +0.250172 | +0.095807 | +0.069652 | -0.080086 | -0.333585 | +0.027347 |
| cpu | narrative | block12_preln | C4 | position | -0.221451 | +0.240274 | -0.323032 | +0.660240 | +0.036506 | +0.217184 | +0.049035 |
| cpu | narrative | block12_preln | C4 | log_norm | -0.311879 | -0.131233 | -0.290461 | -0.057089 | -0.186622 | +0.165162 | +0.472462 |
| cpu | narrative | block12_preln | C4 | log_chars | +0.024432 | +0.276513 | +0.242061 | -0.225888 | -0.340100 | -0.013943 | -0.336101 |
| cpu | narrative | block12_preln | C4 | surprisal | +0.260759 | +0.020949 | +0.292631 | -0.050386 | +0.017256 | -0.143311 | -0.195937 |
| cpu | technical | block12_preln | C4 | position | -0.420469 | +0.489694 | -0.443746 | -0.049850 | +0.377514 | -0.158875 | +0.057953 |
| cpu | technical | block12_preln | C4 | log_norm | -0.140607 | +0.180836 | -0.298233 | -0.393548 | -0.085315 | -0.035527 | -0.109815 |
| cpu | technical | block12_preln | C4 | log_chars | +0.134636 | +0.394490 | +0.059867 | +0.136594 | -0.413245 | -0.209312 | +0.332337 |
| cpu | technical | block12_preln | C4 | surprisal | +0.234262 | +0.120757 | +0.234598 | +0.203121 | -0.127267 | -0.010233 | +0.258409 |
| cpu | repetitive | block12_preln | C4 | position | +0.138993 | -0.105771 | -0.197102 | -0.525092 | +0.177501 | -0.437649 | +0.306351 |
| cpu | repetitive | block12_preln | C4 | log_norm | +0.338514 | -0.192913 | -0.212666 | +0.025331 | +0.709133 | -0.020540 | -0.173234 |
| cpu | repetitive | block12_preln | C4 | log_chars | -0.008024 | +0.075716 | -0.409071 | -0.342199 | -0.238942 | +0.274138 | -0.039785 |
| cpu | repetitive | block12_preln | C4 | surprisal | +0.051456 | +0.129258 | +0.006337 | +0.000624 | -0.426874 | -0.136819 | +0.386997 |
| cpu | narrative | final_ln | C0 | position | -0.124655 | +0.218214 | +0.106026 | -0.191298 | +0.010272 | +0.137330 | -0.212522 |
| cpu | narrative | final_ln | C0 | log_norm | -0.002495 | +0.025667 | +0.049095 | -0.038241 | +0.009287 | -0.003145 | -0.048299 |
| cpu | narrative | final_ln | C0 | log_chars | +0.209836 | -0.366247 | -0.133009 | -0.282063 | +0.033870 | -0.052827 | +0.115801 |
| cpu | narrative | final_ln | C0 | surprisal | +0.215916 | +0.123085 | -0.112868 | -0.020989 | -0.025562 | -0.094638 | -0.006931 |
| cpu | technical | final_ln | C0 | position | -0.211543 | +0.121178 | -0.028747 | -0.184903 | +0.057650 | +0.207858 | +0.177934 |
| cpu | technical | final_ln | C0 | log_norm | +0.056475 | +0.060520 | -0.032434 | +0.035266 | +0.023153 | +0.022094 | -0.011736 |
| cpu | technical | final_ln | C0 | log_chars | -0.254543 | -0.310429 | -0.160964 | -0.270985 | -0.369100 | -0.019824 | +0.216888 |
| cpu | technical | final_ln | C0 | surprisal | +0.146598 | -0.281637 | -0.051763 | -0.028043 | -0.186701 | -0.044987 | +0.164575 |
| cpu | repetitive | final_ln | C0 | position | +0.004990 | -0.424971 | +0.353350 | +0.050433 | -0.021862 | -0.182146 | +0.012814 |
| cpu | repetitive | final_ln | C0 | log_norm | +0.363799 | -0.107829 | -0.165989 | +0.026156 | +0.125655 | -0.107358 | -0.053165 |
| cpu | repetitive | final_ln | C0 | log_chars | +0.518446 | -0.198802 | -0.366382 | -0.265896 | +0.367375 | +0.108821 | -0.067668 |
| cpu | repetitive | final_ln | C0 | surprisal | +0.445197 | +0.122658 | -0.171990 | -0.016371 | +0.241673 | -0.251574 | -0.151502 |
| cpu | narrative | final_ln | C1 | position | -0.081780 | +0.212477 | +0.072431 | -0.170850 | +0.012609 | +0.116874 | -0.186761 |
| cpu | narrative | final_ln | C1 | log_norm | +0.004645 | +0.021813 | +0.042444 | -0.029283 | +0.010228 | -0.014528 | -0.040673 |
| cpu | narrative | final_ln | C1 | log_chars | +0.203430 | -0.419064 | -0.168766 | -0.214078 | +0.030279 | -0.076474 | +0.158962 |
| cpu | narrative | final_ln | C1 | surprisal | +0.230463 | +0.094573 | -0.120319 | -0.008964 | -0.025047 | -0.102696 | +0.005940 |
| cpu | technical | final_ln | C1 | position | -0.217762 | +0.002452 | -0.020133 | -0.220144 | +0.026681 | +0.172973 | +0.183052 |
| cpu | technical | final_ln | C1 | log_norm | +0.070198 | -0.020274 | +0.058621 | +0.011019 | +0.006583 | +0.003709 | -0.013427 |
| cpu | technical | final_ln | C1 | log_chars | -0.249563 | -0.209068 | -0.327225 | -0.149789 | -0.363423 | +0.034430 | +0.216570 |
| cpu | technical | final_ln | C1 | surprisal | +0.148909 | -0.083122 | -0.241898 | +0.072509 | -0.204696 | -0.012362 | +0.165218 |
| cpu | repetitive | final_ln | C1 | position | -0.060591 | -0.403618 | +0.357536 | +0.049501 | -0.058907 | -0.182673 | -0.099504 |
| cpu | repetitive | final_ln | C1 | log_norm | +0.335759 | -0.077834 | -0.180556 | +0.034692 | +0.090300 | -0.107386 | +0.066818 |
| cpu | repetitive | final_ln | C1 | log_chars | +0.561683 | -0.215704 | -0.364614 | -0.267668 | +0.408054 | -0.045884 | -0.055798 |
| cpu | repetitive | final_ln | C1 | surprisal | +0.432734 | +0.125583 | -0.175386 | -0.010953 | +0.192705 | -0.336183 | -0.014519 |
| cpu | narrative | final_ln | C4 | position | +0.134691 | -0.116199 | +0.098164 | +0.092054 | -0.026625 | -0.222638 | -0.061355 |
| cpu | narrative | final_ln | C4 | log_norm | -0.365262 | +0.044010 | +0.206006 | -0.001114 | -0.026633 | -0.244050 | -0.065364 |
| cpu | narrative | final_ln | C4 | log_chars | +0.055781 | -0.312271 | -0.428465 | -0.172504 | +0.016025 | +0.145063 | -0.065842 |
| cpu | narrative | final_ln | C4 | surprisal | +0.361021 | +0.072764 | -0.178132 | +0.054906 | -0.055413 | +0.150929 | -0.070752 |
| cpu | technical | final_ln | C4 | position | -0.168399 | +0.005504 | -0.184727 | +0.083492 | -0.003716 | +0.054787 | -0.269671 |
| cpu | technical | final_ln | C4 | log_norm | +0.222154 | -0.130509 | +0.012815 | +0.304584 | +0.224619 | +0.139925 | +0.170298 |
| cpu | technical | final_ln | C4 | log_chars | -0.033793 | +0.239768 | -0.208094 | -0.155201 | -0.325727 | +0.281931 | -0.102135 |
| cpu | technical | final_ln | C4 | surprisal | +0.052067 | +0.057346 | -0.003862 | -0.057803 | -0.121347 | +0.158995 | -0.048807 |
| cpu | repetitive | final_ln | C4 | position | -0.358029 | -0.541226 | +0.159188 | -0.185769 | -0.011143 | -0.087919 | -0.230888 |
| cpu | repetitive | final_ln | C4 | log_norm | -0.286590 | -0.234045 | -0.089714 | +0.021178 | +0.086146 | +0.140116 | -0.007333 |
| cpu | repetitive | final_ln | C4 | log_chars | +0.254929 | +0.040193 | -0.536566 | +0.027264 | -0.036286 | +0.240121 | -0.376382 |
| cpu | repetitive | final_ln | C4 | surprisal | +0.483717 | +0.357248 | -0.187201 | -0.401277 | -0.168018 | -0.190438 | -0.276533 |
| cuda | narrative | block06 | C0 | position | -0.679356 | -0.567615 | -0.074329 | +0.287681 | +0.026768 | +0.042135 | +0.081576 |
| cuda | narrative | block06 | C0 | log_norm | +0.142889 | -0.150702 | -0.030035 | -0.049069 | -0.041886 | +0.012798 | -0.048079 |
| cuda | narrative | block06 | C0 | log_chars | +0.487617 | -0.359356 | +0.028954 | -0.210440 | -0.278246 | +0.170850 | -0.040550 |
| cuda | narrative | block06 | C0 | surprisal | +0.406214 | -0.260122 | -0.012167 | -0.234420 | -0.204808 | +0.114069 | -0.023398 |
| cuda | technical | block06 | C0 | position | -0.154027 | -0.779808 | -0.454301 | -0.123012 | +0.015629 | -0.143203 | -0.234793 |
| cuda | technical | block06 | C0 | log_norm | -0.131340 | +0.141083 | -0.084955 | -0.018464 | +0.056726 | -0.017770 | -0.004044 |
| cuda | technical | block06 | C0 | log_chars | -0.592810 | +0.025948 | -0.071628 | -0.119174 | +0.195138 | -0.015864 | +0.119546 |
| cuda | technical | block06 | C0 | surprisal | -0.422896 | +0.338652 | -0.126075 | -0.103803 | +0.150059 | -0.113582 | +0.118655 |
| cuda | repetitive | block06 | C0 | position | -0.026543 | -0.240952 | -0.063046 | -0.861956 | +0.042496 | +0.118989 | -0.019539 |
| cuda | repetitive | block06 | C0 | log_norm | +0.059935 | -0.075615 | +0.045674 | +0.011290 | +0.015972 | -0.056901 | +0.021098 |
| cuda | repetitive | block06 | C0 | log_chars | -0.189136 | -0.630456 | +0.517405 | -0.125987 | +0.079883 | -0.404173 | +0.087518 |
| cuda | repetitive | block06 | C0 | surprisal | -0.369426 | -0.463574 | +0.041131 | +0.003823 | -0.150726 | +0.208422 | -0.146817 |
| cuda | narrative | block06 | C1 | position | -0.563615 | +0.226433 | -0.173904 | -0.065502 | +0.016001 | +0.019275 | +0.003603 |
| cuda | narrative | block06 | C1 | log_norm | -0.521839 | +0.170932 | +0.161084 | -0.095332 | -0.095181 | -0.118161 | +0.142193 |
| cuda | narrative | block06 | C1 | log_chars | -0.367766 | -0.115909 | +0.136927 | -0.152286 | -0.281138 | -0.080902 | +0.016870 |
| cuda | narrative | block06 | C1 | surprisal | -0.269009 | -0.086343 | +0.166529 | -0.059140 | -0.200616 | -0.066642 | +0.057916 |
| cuda | technical | block06 | C1 | position | -0.743770 | -0.490698 | -0.130307 | +0.025606 | -0.147304 | -0.285219 | -0.023733 |
| cuda | technical | block06 | C1 | log_norm | +0.516097 | -0.078605 | -0.045698 | +0.292715 | -0.123204 | +0.161028 | -0.106287 |
| cuda | technical | block06 | C1 | log_chars | +0.103786 | +0.015284 | -0.128681 | +0.193811 | -0.003224 | +0.066331 | -0.189332 |
| cuda | technical | block06 | C1 | surprisal | +0.385776 | +0.024436 | -0.110365 | +0.165179 | -0.114272 | +0.117720 | +0.007262 |
| cuda | repetitive | block06 | C1 | position | -0.039350 | +0.006486 | -0.949853 | -0.077993 | +0.107070 | -0.004286 | -0.107807 |
| cuda | repetitive | block06 | C1 | log_norm | -0.592303 | -0.042013 | +0.254555 | +0.230509 | -0.170563 | +0.051902 | -0.129416 |
| cuda | repetitive | block06 | C1 | log_chars | -0.519335 | +0.176455 | -0.096800 | -0.011818 | -0.376627 | +0.066358 | +0.204901 |
| cuda | repetitive | block06 | C1 | surprisal | -0.557761 | -0.046409 | +0.087353 | +0.301457 | +0.140496 | -0.133241 | +0.056678 |
| cuda | narrative | block06 | C4 | position | -0.907801 | +0.182290 | +0.158535 | +0.071646 | -0.079242 | +0.036995 | -0.042796 |
| cuda | narrative | block06 | C4 | log_norm | +0.135339 | +0.274958 | +0.149411 | +0.166847 | +0.100671 | +0.004822 | +0.144623 |
| cuda | narrative | block06 | C4 | log_chars | +0.075109 | -0.231172 | +0.095173 | +0.204445 | +0.011514 | -0.081207 | +0.157887 |
| cuda | narrative | block06 | C4 | surprisal | +0.128635 | -0.058780 | -0.015236 | +0.128504 | +0.055282 | -0.145306 | +0.077028 |
| cuda | technical | block06 | C4 | position | +0.425334 | -0.649716 | +0.054242 | -0.109814 | +0.204404 | -0.217858 | -0.106765 |
| cuda | technical | block06 | C4 | log_norm | +0.121342 | +0.247845 | +0.112081 | +0.093490 | +0.194054 | -0.056011 | +0.254747 |
| cuda | technical | block06 | C4 | log_chars | +0.596007 | +0.127361 | +0.123830 | +0.049185 | -0.208026 | +0.168034 | -0.226604 |
| cuda | technical | block06 | C4 | surprisal | +0.338904 | +0.175329 | +0.125937 | +0.088377 | -0.041645 | +0.097099 | +0.051503 |
| cuda | repetitive | block06 | C4 | position | +0.061098 | +0.041055 | -0.952684 | -0.080506 | +0.099091 | -0.177685 | -0.003749 |
| cuda | repetitive | block06 | C4 | log_norm | -0.222774 | +0.018583 | +0.165118 | +0.341007 | +0.093788 | -0.199085 | +0.043588 |
| cuda | repetitive | block06 | C4 | log_chars | -0.497922 | +0.592699 | -0.122443 | -0.139146 | -0.132687 | +0.146523 | +0.087157 |
| cuda | repetitive | block06 | C4 | surprisal | -0.445744 | +0.057168 | +0.058183 | +0.249312 | +0.095472 | -0.047076 | +0.049484 |
| cuda | narrative | block07 | C0 | position | -0.750178 | -0.415763 | -0.311938 | +0.188825 | +0.024229 | +0.032847 | -0.039061 |
| cuda | narrative | block07 | C0 | log_norm | +0.097521 | -0.156953 | -0.025486 | -0.060385 | -0.043051 | +0.031118 | -0.028978 |
| cuda | narrative | block07 | C0 | log_chars | +0.359367 | -0.440781 | +0.115732 | -0.107006 | -0.162487 | +0.316027 | -0.050892 |
| cuda | narrative | block07 | C0 | surprisal | +0.323989 | -0.305601 | +0.090365 | -0.190566 | -0.171910 | +0.237093 | -0.007410 |
| cuda | technical | block07 | C0 | position | +0.644503 | +0.523402 | -0.541830 | -0.099028 | +0.020372 | +0.141041 | +0.151938 |
| cuda | technical | block07 | C0 | log_norm | -0.003884 | -0.163604 | -0.064326 | -0.017367 | -0.039037 | +0.014691 | -0.012027 |
| cuda | technical | block07 | C0 | log_chars | +0.454426 | -0.330501 | -0.235467 | -0.199499 | -0.257053 | -0.058403 | -0.183962 |
| cuda | technical | block07 | C0 | surprisal | +0.097213 | -0.486955 | -0.232601 | -0.134715 | -0.174134 | +0.063992 | -0.160137 |
| cuda | repetitive | block07 | C0 | position | -0.006806 | +0.176421 | -0.124754 | -0.829334 | +0.088247 | +0.018999 | -0.102070 |
| cuda | repetitive | block07 | C0 | log_norm | +0.059695 | +0.057898 | -0.049988 | +0.041674 | +0.002714 | -0.041649 | -0.013344 |
| cuda | repetitive | block07 | C0 | log_chars | -0.056630 | +0.590377 | -0.564234 | -0.009760 | -0.170232 | -0.424455 | +0.109669 |
| cuda | repetitive | block07 | C0 | surprisal | -0.301992 | +0.469694 | -0.111665 | -0.049504 | -0.089408 | +0.251891 | +0.014090 |
| cuda | narrative | block07 | C1 | position | -0.408343 | +0.332325 | +0.163252 | -0.071491 | +0.015197 | +0.014787 | -0.025759 |
| cuda | narrative | block07 | C1 | log_norm | -0.561468 | +0.170442 | -0.220693 | -0.161862 | +0.152536 | -0.133754 | -0.041989 |
| cuda | narrative | block07 | C1 | log_chars | -0.439606 | -0.093614 | -0.092464 | -0.082763 | +0.347688 | -0.085622 | -0.025622 |
| cuda | narrative | block07 | C1 | surprisal | -0.320794 | -0.117613 | -0.182364 | -0.079973 | +0.263435 | -0.070006 | -0.002937 |
| cuda | technical | block07 | C1 | position | -0.658985 | -0.568818 | -0.151657 | +0.008419 | +0.150378 | +0.174711 | -0.112310 |
| cuda | technical | block07 | C1 | log_norm | +0.582865 | +0.028939 | -0.051823 | -0.219464 | +0.093794 | -0.261292 | +0.041732 |
| cuda | technical | block07 | C1 | log_chars | +0.300798 | -0.063685 | -0.225903 | -0.271573 | -0.052772 | -0.216025 | -0.098902 |
| cuda | technical | block07 | C1 | surprisal | +0.497370 | +0.004347 | -0.162884 | -0.200759 | +0.080067 | -0.217033 | +0.091058 |
| cuda | repetitive | block07 | C1 | position | +0.014794 | -0.410168 | -0.922255 | -0.080695 | +0.054461 | -0.107880 | -0.003320 |
| cuda | repetitive | block07 | C1 | log_norm | +0.555679 | -0.151618 | +0.393039 | +0.304972 | -0.027668 | -0.095891 | -0.116356 |
| cuda | repetitive | block07 | C1 | log_chars | +0.711465 | -0.530828 | -0.043350 | +0.067628 | -0.406209 | +0.097945 | +0.051794 |
| cuda | repetitive | block07 | C1 | surprisal | +0.554570 | -0.366191 | +0.109315 | +0.314557 | +0.205869 | +0.016095 | +0.135911 |
| cuda | narrative | block07 | C4 | position | +0.573325 | +0.221139 | +0.141503 | -0.108700 | -0.059286 | -0.014930 | -0.003523 |
| cuda | narrative | block07 | C4 | log_norm | -0.293781 | +0.231694 | +0.031919 | -0.117060 | +0.141792 | +0.022307 | +0.169794 |
| cuda | narrative | block07 | C4 | log_chars | -0.614458 | -0.146199 | -0.043483 | -0.282842 | +0.123881 | +0.034234 | +0.210589 |
| cuda | narrative | block07 | C4 | surprisal | -0.534083 | -0.020339 | -0.135729 | -0.172712 | +0.113452 | +0.099724 | +0.055829 |
| cuda | technical | block07 | C4 | position | -0.212391 | -0.753807 | +0.022830 | +0.089594 | +0.150612 | +0.002198 | -0.233028 |
| cuda | technical | block07 | C4 | log_norm | -0.194975 | +0.192225 | +0.177683 | -0.119375 | +0.167786 | +0.204595 | +0.127269 |
| cuda | technical | block07 | C4 | log_chars | -0.498959 | +0.045262 | +0.269323 | -0.170752 | -0.115362 | -0.118560 | -0.106002 |
| cuda | technical | block07 | C4 | surprisal | -0.436460 | +0.188865 | +0.219260 | -0.117842 | +0.059093 | +0.091960 | +0.065045 |
| cuda | repetitive | block07 | C4 | position | -0.144609 | -0.175811 | -0.917914 | -0.206911 | +0.050087 | -0.181313 | -0.046667 |
| cuda | repetitive | block07 | C4 | log_norm | +0.187070 | -0.090262 | +0.167889 | +0.384342 | +0.103344 | -0.167702 | -0.040827 |
| cuda | repetitive | block07 | C4 | log_chars | +0.708265 | -0.769386 | -0.070830 | +0.021548 | -0.270639 | +0.160464 | -0.004972 |
| cuda | repetitive | block07 | C4 | surprisal | +0.358243 | -0.343697 | +0.091596 | +0.322890 | +0.034898 | -0.108374 | -0.018124 |
| cuda | narrative | block12_preln | C0 | position | +0.015386 | +0.237284 | +0.199224 | -0.765883 | -0.159504 | -0.074586 | +0.110063 |
| cuda | narrative | block12_preln | C0 | log_norm | +0.426207 | +0.041827 | -0.055922 | +0.091395 | -0.028156 | +0.161427 | +0.219687 |
| cuda | narrative | block12_preln | C0 | log_chars | -0.110077 | +0.259354 | -0.172433 | +0.118937 | +0.144118 | -0.069715 | -0.154617 |
| cuda | narrative | block12_preln | C0 | surprisal | -0.159823 | -0.045930 | -0.069808 | +0.048794 | -0.044438 | +0.081172 | -0.242190 |
| cuda | technical | block12_preln | C0 | position | -0.250852 | +0.211142 | +0.300949 | -0.248684 | +0.471139 | -0.428265 | +0.000166 |
| cuda | technical | block12_preln | C0 | log_norm | -0.201639 | +0.010361 | +0.127938 | +0.164225 | -0.023068 | +0.100696 | -0.007520 |
| cuda | technical | block12_preln | C0 | log_chars | -0.049780 | +0.378346 | +0.425112 | -0.095503 | -0.207282 | -0.116723 | -0.362457 |
| cuda | technical | block12_preln | C0 | surprisal | +0.055501 | +0.211675 | +0.059657 | -0.021071 | -0.055747 | +0.003813 | -0.140478 |
| cuda | repetitive | block12_preln | C0 | position | -0.051124 | +0.106572 | +0.085082 | -0.365508 | -0.574379 | -0.315925 | -0.238841 |
| cuda | repetitive | block12_preln | C0 | log_norm | +0.698042 | -0.372827 | +0.144883 | +0.213160 | -0.147860 | +0.010379 | +0.043128 |
| cuda | repetitive | block12_preln | C0 | log_chars | +0.034348 | +0.147002 | +0.363047 | -0.231282 | -0.190021 | +0.239730 | -0.349682 |
| cuda | repetitive | block12_preln | C0 | surprisal | -0.092325 | +0.261800 | +0.022352 | -0.226385 | +0.087161 | -0.083589 | -0.330873 |
| cuda | narrative | block12_preln | C1 | position | -0.031148 | +0.170692 | +0.059856 | -0.767053 | -0.110879 | +0.226883 | -0.053055 |
| cuda | narrative | block12_preln | C1 | log_norm | +0.595114 | +0.190794 | +0.074169 | +0.018360 | -0.012548 | +0.071709 | -0.138621 |
| cuda | narrative | block12_preln | C1 | log_chars | -0.208039 | +0.231996 | -0.253366 | +0.174295 | +0.159356 | +0.014670 | -0.310617 |
| cuda | narrative | block12_preln | C1 | surprisal | -0.239358 | -0.092124 | -0.128437 | +0.079417 | -0.037279 | -0.209650 | +0.089775 |
| cuda | technical | block12_preln | C1 | position | -0.265589 | +0.260107 | +0.364492 | +0.401460 | -0.491233 | -0.018130 | -0.096512 |
| cuda | technical | block12_preln | C1 | log_norm | -0.231126 | +0.048119 | -0.148517 | +0.023796 | +0.036159 | +0.026493 | +0.042021 |
| cuda | technical | block12_preln | C1 | log_chars | -0.040561 | +0.408503 | +0.196393 | -0.185420 | -0.186428 | -0.336226 | +0.097303 |
| cuda | technical | block12_preln | C1 | surprisal | +0.076829 | +0.206618 | +0.055603 | -0.071465 | +0.014526 | -0.145343 | +0.015229 |
| cuda | repetitive | block12_preln | C1 | position | -0.118639 | +0.227731 | +0.171701 | -0.631843 | -0.351919 | -0.253354 | +0.047448 |
| cuda | repetitive | block12_preln | C1 | log_norm | +0.741002 | -0.553012 | +0.009811 | -0.112749 | -0.004732 | +0.036879 | +0.068062 |
| cuda | repetitive | block12_preln | C1 | log_chars | +0.015354 | +0.040117 | +0.408275 | -0.247487 | +0.234100 | -0.340439 | +0.075554 |
| cuda | repetitive | block12_preln | C1 | surprisal | -0.127544 | +0.250172 | +0.095808 | +0.069651 | -0.080086 | -0.333587 | +0.027346 |
| cuda | narrative | block12_preln | C4 | position | -0.221451 | +0.240274 | -0.323033 | +0.660239 | +0.036506 | +0.217184 | +0.049035 |
| cuda | narrative | block12_preln | C4 | log_norm | -0.311881 | -0.131232 | -0.290460 | -0.057090 | -0.186621 | +0.165163 | +0.472461 |
| cuda | narrative | block12_preln | C4 | log_chars | +0.024433 | +0.276513 | +0.242062 | -0.225887 | -0.340100 | -0.013943 | -0.336101 |
| cuda | narrative | block12_preln | C4 | surprisal | +0.260759 | +0.020947 | +0.292631 | -0.050384 | +0.017257 | -0.143312 | -0.195937 |
| cuda | technical | block12_preln | C4 | position | -0.420470 | +0.489695 | -0.443745 | -0.049850 | +0.377514 | -0.158875 | +0.057953 |
| cuda | technical | block12_preln | C4 | log_norm | -0.140607 | +0.180837 | -0.298233 | -0.393548 | -0.085315 | -0.035527 | -0.109815 |
| cuda | technical | block12_preln | C4 | log_chars | +0.134636 | +0.394490 | +0.059867 | +0.136594 | -0.413245 | -0.209312 | +0.332337 |
| cuda | technical | block12_preln | C4 | surprisal | +0.234263 | +0.120758 | +0.234599 | +0.203121 | -0.127268 | -0.010233 | +0.258409 |
| cuda | repetitive | block12_preln | C4 | position | +0.138993 | -0.105771 | -0.197102 | -0.525092 | +0.177500 | -0.437649 | +0.306351 |
| cuda | repetitive | block12_preln | C4 | log_norm | +0.338514 | -0.192913 | -0.212667 | +0.025329 | +0.709133 | -0.020540 | -0.173234 |
| cuda | repetitive | block12_preln | C4 | log_chars | -0.008024 | +0.075717 | -0.409071 | -0.342198 | -0.238943 | +0.274138 | -0.039786 |
| cuda | repetitive | block12_preln | C4 | surprisal | +0.051458 | +0.129258 | +0.006337 | +0.000625 | -0.426873 | -0.136820 | +0.386997 |
| cuda | narrative | final_ln | C0 | position | -0.124655 | +0.218214 | +0.106026 | -0.191298 | +0.010272 | +0.137330 | -0.212522 |
| cuda | narrative | final_ln | C0 | log_norm | -0.002495 | +0.025667 | +0.049095 | -0.038241 | +0.009287 | -0.003145 | -0.048299 |
| cuda | narrative | final_ln | C0 | log_chars | +0.209836 | -0.366247 | -0.133008 | -0.282063 | +0.033870 | -0.052827 | +0.115802 |
| cuda | narrative | final_ln | C0 | surprisal | +0.215915 | +0.123085 | -0.112868 | -0.020986 | -0.025564 | -0.094639 | -0.006930 |
| cuda | technical | final_ln | C0 | position | -0.211543 | +0.121177 | -0.028747 | -0.184903 | +0.057649 | +0.207858 | +0.177933 |
| cuda | technical | final_ln | C0 | log_norm | +0.056475 | +0.060519 | -0.032435 | +0.035266 | +0.023153 | +0.022094 | -0.011736 |
| cuda | technical | final_ln | C0 | log_chars | -0.254543 | -0.310430 | -0.160962 | -0.270986 | -0.369100 | -0.019824 | +0.216889 |
| cuda | technical | final_ln | C0 | surprisal | +0.146597 | -0.281638 | -0.051764 | -0.028044 | -0.186700 | -0.044989 | +0.164575 |
| cuda | repetitive | final_ln | C0 | position | +0.004990 | -0.424971 | +0.353351 | +0.050433 | -0.021862 | -0.182147 | +0.012814 |
| cuda | repetitive | final_ln | C0 | log_norm | +0.363799 | -0.107829 | -0.165989 | +0.026155 | +0.125655 | -0.107358 | -0.053165 |
| cuda | repetitive | final_ln | C0 | log_chars | +0.518445 | -0.198802 | -0.366382 | -0.265896 | +0.367376 | +0.108820 | -0.067667 |
| cuda | repetitive | final_ln | C0 | surprisal | +0.445197 | +0.122658 | -0.171990 | -0.016371 | +0.241673 | -0.251575 | -0.151501 |
| cuda | narrative | final_ln | C1 | position | -0.081780 | +0.212477 | +0.072431 | -0.170850 | +0.012609 | +0.116873 | -0.186760 |
| cuda | narrative | final_ln | C1 | log_norm | +0.004645 | +0.021813 | +0.042444 | -0.029283 | +0.010228 | -0.014528 | -0.040673 |
| cuda | narrative | final_ln | C1 | log_chars | +0.203431 | -0.419064 | -0.168765 | -0.214078 | +0.030279 | -0.076474 | +0.158963 |
| cuda | narrative | final_ln | C1 | surprisal | +0.230462 | +0.094574 | -0.120319 | -0.008962 | -0.025049 | -0.102697 | +0.005941 |
| cuda | technical | final_ln | C1 | position | -0.217762 | +0.002452 | -0.020133 | -0.220144 | +0.026681 | +0.172973 | +0.183051 |
| cuda | technical | final_ln | C1 | log_norm | +0.070197 | -0.020274 | +0.058621 | +0.011019 | +0.006584 | +0.003709 | -0.013427 |
| cuda | technical | final_ln | C1 | log_chars | -0.249563 | -0.209068 | -0.327225 | -0.149790 | -0.363423 | +0.034429 | +0.216570 |
| cuda | technical | final_ln | C1 | surprisal | +0.148909 | -0.083125 | -0.241899 | +0.072507 | -0.204695 | -0.012363 | +0.165219 |
| cuda | repetitive | final_ln | C1 | position | -0.060591 | -0.403618 | +0.357537 | +0.049501 | -0.058907 | -0.182674 | -0.099504 |
| cuda | repetitive | final_ln | C1 | log_norm | +0.335759 | -0.077834 | -0.180556 | +0.034692 | +0.090300 | -0.107386 | +0.066818 |
| cuda | repetitive | final_ln | C1 | log_chars | +0.561682 | -0.215704 | -0.364614 | -0.267668 | +0.408055 | -0.045884 | -0.055798 |
| cuda | repetitive | final_ln | C1 | surprisal | +0.432734 | +0.125583 | -0.175386 | -0.010953 | +0.192704 | -0.336184 | -0.014521 |
| cuda | narrative | final_ln | C4 | position | +0.134691 | -0.116199 | +0.098164 | +0.092054 | -0.026625 | -0.222639 | -0.061355 |
| cuda | narrative | final_ln | C4 | log_norm | -0.365262 | +0.044009 | +0.206006 | -0.001114 | -0.026632 | -0.244050 | -0.065365 |
| cuda | narrative | final_ln | C4 | log_chars | +0.055782 | -0.312271 | -0.428465 | -0.172504 | +0.016025 | +0.145063 | -0.065843 |
| cuda | narrative | final_ln | C4 | surprisal | +0.361021 | +0.072767 | -0.178131 | +0.054908 | -0.055412 | +0.150929 | -0.070751 |
| cuda | technical | final_ln | C4 | position | -0.168399 | +0.005505 | -0.184728 | +0.083492 | -0.003716 | +0.054787 | -0.269671 |
| cuda | technical | final_ln | C4 | log_norm | +0.222155 | -0.130509 | +0.012816 | +0.304584 | +0.224619 | +0.139925 | +0.170298 |
| cuda | technical | final_ln | C4 | log_chars | -0.033793 | +0.239768 | -0.208095 | -0.155201 | -0.325727 | +0.281931 | -0.102135 |
| cuda | technical | final_ln | C4 | surprisal | +0.052067 | +0.057348 | -0.003864 | -0.057804 | -0.121347 | +0.158995 | -0.048807 |
| cuda | repetitive | final_ln | C4 | position | -0.358029 | -0.541225 | +0.159188 | -0.185769 | -0.011143 | -0.087919 | -0.230888 |
| cuda | repetitive | final_ln | C4 | log_norm | -0.286589 | -0.234045 | -0.089715 | +0.021178 | +0.086146 | +0.140115 | -0.007333 |
| cuda | repetitive | final_ln | C4 | log_chars | +0.254928 | +0.040194 | -0.536566 | +0.027264 | -0.036286 | +0.240121 | -0.376383 |
| cuda | repetitive | final_ln | C4 | surprisal | +0.483717 | +0.357248 | -0.187200 | -0.401277 | -0.168019 | -0.190439 | -0.276533 |

## CPU/CUDAの生状態差

| 文章 | 点 | 最大絶対差 | 相対Frobenius差 | 固定許容差内 | 超過座標数 |
| --- | --- | --- | --- | --- | --- |
| narrative | block06 | 0.000244140625 | 2.06166054e-07 | True | 0 |
| narrative | block07 | 0.00048828125 | 2.59991267e-07 | True | 0 |
| narrative | block12_preln | 0.00131225586 | 5.49955852e-07 | True | 0 |
| narrative | final_ln | 0.000442504883 | 4.39504233e-07 | True | 0 |
| technical | block06 | 0.000244140625 | 1.8976982e-07 | True | 0 |
| technical | block07 | 0.000244140625 | 2.03146375e-07 | True | 0 |
| technical | block12_preln | 0.000701904297 | 5.17427739e-07 | True | 0 |
| technical | final_ln | 0.000183105469 | 3.00225571e-07 | True | 0 |
| repetitive | block06 | 0.00048828125 | 2.42281327e-07 | True | 0 |
| repetitive | block07 | 0.00048828125 | 2.68137739e-07 | True | 0 |
| repetitive | block12_preln | 0.000553131104 | 7.72186983e-07 | False | 1 |
| repetitive | final_ln | 0.000457763672 | 6.18876692e-07 | True | 0 |
