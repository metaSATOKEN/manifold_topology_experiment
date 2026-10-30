# P5 全図

全３文章を表示。各処理でPCAを再計算しており、異なる図のPC軸は同一ではない。
線付き図の線は補間。色はtoken順（紫→黄）、赤△は先頭。2Dの青▽は末尾。
3Dは３軸とも同じ単位・同じ幅。CPUとGPUの全条件を掲載する。

## 01_historical_comparison

旧図原画像と同じ３文章の再現。下段の線は隣接token間の補間。

![旧図原画像と同じ３文章の再現。下段の線は隣接token間の補間。](figures/01_historical_comparison.png)

## 02_points_zoom_and_C1

線なし全点、同一PCAの中央90%拡大、先頭除外後の再PCA。外側点は図中に計数。

![線なし全点、同一PCAの中央90%拡大、先頭除外後の再PCA。外側点は図中に計数。](figures/02_points_zoom_and_C1.png)

## atlas_cpu_block06_2d

cpu block06の全３文章×３処理、2D等縮尺。

![cpu block06の全３文章×３処理、2D等縮尺。](figures/atlas_cpu_block06_2d.png)

## atlas_cpu_block06_3d

cpu block06の全３文章×３処理、3D等縮尺。

![cpu block06の全３文章×３処理、3D等縮尺。](figures/atlas_cpu_block06_3d.png)

## atlas_cpu_block07_2d

cpu block07の全３文章×３処理、2D等縮尺。

![cpu block07の全３文章×３処理、2D等縮尺。](figures/atlas_cpu_block07_2d.png)

## atlas_cpu_block07_3d

cpu block07の全３文章×３処理、3D等縮尺。

![cpu block07の全３文章×３処理、3D等縮尺。](figures/atlas_cpu_block07_3d.png)

## atlas_cpu_block12_preln_2d

cpu block12_prelnの全３文章×３処理、2D等縮尺。

![cpu block12_prelnの全３文章×３処理、2D等縮尺。](figures/atlas_cpu_block12_preln_2d.png)

## atlas_cpu_block12_preln_3d

cpu block12_prelnの全３文章×３処理、3D等縮尺。

![cpu block12_prelnの全３文章×３処理、3D等縮尺。](figures/atlas_cpu_block12_preln_3d.png)

## atlas_cpu_final_ln_2d

cpu final_lnの全３文章×３処理、2D等縮尺。

![cpu final_lnの全３文章×３処理、2D等縮尺。](figures/atlas_cpu_final_ln_2d.png)

## atlas_cpu_final_ln_3d

cpu final_lnの全３文章×３処理、3D等縮尺。

![cpu final_lnの全３文章×３処理、3D等縮尺。](figures/atlas_cpu_final_ln_3d.png)

## atlas_cuda_block06_2d

cuda block06の全３文章×３処理、2D等縮尺。

![cuda block06の全３文章×３処理、2D等縮尺。](figures/atlas_cuda_block06_2d.png)

## atlas_cuda_block06_3d

cuda block06の全３文章×３処理、3D等縮尺。

![cuda block06の全３文章×３処理、3D等縮尺。](figures/atlas_cuda_block06_3d.png)

## atlas_cuda_block07_2d

cuda block07の全３文章×３処理、2D等縮尺。

![cuda block07の全３文章×３処理、2D等縮尺。](figures/atlas_cuda_block07_2d.png)

## atlas_cuda_block07_3d

cuda block07の全３文章×３処理、3D等縮尺。

![cuda block07の全３文章×３処理、3D等縮尺。](figures/atlas_cuda_block07_3d.png)

## atlas_cuda_block12_preln_2d

cuda block12_prelnの全３文章×３処理、2D等縮尺。

![cuda block12_prelnの全３文章×３処理、2D等縮尺。](figures/atlas_cuda_block12_preln_2d.png)

## atlas_cuda_block12_preln_3d

cuda block12_prelnの全３文章×３処理、3D等縮尺。

![cuda block12_prelnの全３文章×３処理、3D等縮尺。](figures/atlas_cuda_block12_preln_3d.png)

## atlas_cuda_final_ln_2d

cuda final_lnの全３文章×３処理、2D等縮尺。

![cuda final_lnの全３文章×３処理、2D等縮尺。](figures/atlas_cuda_final_ln_2d.png)

## atlas_cuda_final_ln_3d

cuda final_lnの全３文章×３処理、3D等縮尺。

![cuda final_lnの全３文章×３処理、3D等縮尺。](figures/atlas_cuda_final_ln_3d.png)

## 03_thickness_all_axes

主条件の残差PC2〜8×全４特徴×３文章。軸を選別しない。

![主条件の残差PC2〜8×全４特徴×３文章。軸を選別しない。](figures/03_thickness_all_axes.png)
