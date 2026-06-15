# 第 5b/6b 阶段 PD-only 特征扩展与权重敏感性报告

## 1. 方法学动机
Healthy/PD 判别特征是为区分病例与健康对照而排序的，可能主要反映疾病有无、性别或全局声学差异；PD 内部分型需要捕捉 188 名 PD 受试者内部的异质性。因此第 5b/6b 阶段将特征筛选限定在 PD-only 样本内，并保留第 4 阶段 Top20/Top50 仅作为对照。

## 2. 新增 PD-only 特征方案
| 特征方案 | 原始特征数 | 是否为 SparsePCA 表示 | SparsePCA 维数 | 聚类表示 | 筛选规则 |
| --- | --- | --- | --- | --- | --- |
| top20_biomarkers | 20 | False |  | scaled_only;pca_5;pca_10 | 封版第 4 阶段 Top20；仅作对照；不加评分奖励 |
| top50_biomarkers | 50 | False |  | scaled_only;pca_5;pca_10 | 封版第 4 阶段 Top50；仅作对照；不加评分奖励 |
| all_acoustic | 752 | False |  | pca_10;pca_20;pca_90 | 全部 752 个声学特征；聚类前必须先做 PCA |
| pd_corr_reduced_all_090 | 310 | False |  | pca_10;pca_20;pca_90 | 基于 PD-only 样本，从全部声学特征出发按 |r| > 0.90 做相关性去冗余 |
| pd_corr_reduced_all_095 | 425 | False |  | pca_10;pca_20;pca_90 | 基于 PD-only 样本，从全部声学特征出发按 |r| > 0.95 做相关性去冗余 |
| pd_corr_reduced_top50_090 | 31 | False |  | scaled_only;pca_5;pca_10 | 基于 PD-only 样本，对第 4 阶段 Top50 按 |r| > 0.90 做相关性去冗余 |
| pd_high_iqr_top50 | 50 | False |  | scaled_only;pca_5;pca_10 | 基于 PD-only 样本，在 752 个声学特征中保留 IQR 最大的 50 个 |
| pd_high_iqr_top100 | 100 | False |  | pca_10;pca_20;pca_90 | 基于 PD-only 样本，在 752 个声学特征中保留 IQR 最大的 100 个 |
| pd_high_mad_top50 | 50 | False |  | scaled_only;pca_5;pca_10 | 基于 PD-only 样本，在 752 个声学特征中保留 MAD 最大的 50 个 |
| pd_high_mad_top100 | 100 | False |  | pca_10;pca_20;pca_90 | 基于 PD-only 样本，在 752 个声学特征中保留 MAD 最大的 100 个 |
| pd_sparse_pca_10 | 752 | True | 10.0000 | sparse_pca_10 | 对 752 个 PD-only 声学特征先标准化，再提取 10 维 SparsePCA 表示 |
| pd_sparse_pca_20 | 752 | True | 20.0000 | sparse_pca_20 | 对 752 个 PD-only 声学特征先标准化，再提取 20 维 SparsePCA 表示 |

## 3. 取消人为奖励
第 5b/6b 阶段不对 Top20、Top50 或第 4 阶段 biomarker 特征集加解释性奖励；第 6b 阶段不使用第 5 阶段 NMI 参与主推荐评分；图像好看程度不进入任何评分。解释性只在论文讨论中单独说明。

## 4. 第 5b 阶段全部候选方案客观指标
| 特征方案 | 表示方式 | 聚类方法 | 簇人数 | 较小簇比例 | 轮廓系数 | Calinski-Harabasz | Davies-Bouldin | 稳定性 ARI 均值 | 是否可推荐 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top20_biomarkers | scaled_only | KMeans | 0:62;1:126 | 0.3298 | 0.2950 | 78.4693 | 1.3274 | 0.9870 | True |
| top20_biomarkers | scaled_only | GaussianMixture | 0:106;1:82 | 0.4362 | 0.1908 | 50.3604 | 1.7521 | 0.8032 | True |
| top20_biomarkers | scaled_only | AgglomerativeClustering | 0:133;1:55 | 0.2926 | 0.2972 | 74.9374 | 1.3023 | 0.8956 | True |
| top20_biomarkers | pca_5 | KMeans | 0:62;1:126 | 0.3298 | 0.3674 | 109.9050 | 1.0711 | 0.9897 | True |
| top20_biomarkers | pca_5 | GaussianMixture | 0:143;1:45 | 0.2394 | 0.2324 | 30.0584 | 2.1576 | 0.4425 | True |
| top20_biomarkers | pca_5 | AgglomerativeClustering | 0:135;1:53 | 0.2819 | 0.3713 | 104.7317 | 1.0343 | 0.8677 | True |
| top20_biomarkers | pca_10 | KMeans | 0:62;1:126 | 0.3298 | 0.3164 | 87.1264 | 1.2437 | 0.9881 | True |
| top20_biomarkers | pca_10 | GaussianMixture | 0:136;1:52 | 0.2766 | 0.1792 | 38.4521 | 1.8398 | 0.5144 | True |
| top20_biomarkers | pca_10 | AgglomerativeClustering | 0:134;1:54 | 0.2872 | 0.3187 | 82.5403 | 1.2165 | 0.8778 | True |
| top50_biomarkers | scaled_only | KMeans | 0:94;1:94 | 0.5000 | 0.2147 | 56.5849 | 1.6488 | 0.9211 | True |
| top50_biomarkers | scaled_only | GaussianMixture | 0:97;1:91 | 0.4840 | 0.2104 | 55.0911 | 1.6732 | 0.7792 | True |
| top50_biomarkers | scaled_only | AgglomerativeClustering | 0:139;1:49 | 0.2606 | 0.2280 | 49.0916 | 1.4839 | 0.8858 | True |
| top50_biomarkers | pca_5 | KMeans | 0:88;1:100 | 0.4681 | 0.3073 | 88.3439 | 1.2422 | 0.9224 | True |
| top50_biomarkers | pca_5 | GaussianMixture | 0:155;1:33 | 0.1755 | 0.3418 | 47.3357 | 1.5950 | 0.4328 | True |
| top50_biomarkers | pca_5 | AgglomerativeClustering | 0:134;1:54 | 0.2872 | 0.2943 | 71.7972 | 1.1734 | 0.6994 | True |
| top50_biomarkers | pca_10 | KMeans | 0:97;1:91 | 0.4840 | 0.2581 | 70.0834 | 1.4360 | 0.9250 | True |
| top50_biomarkers | pca_10 | GaussianMixture | 0:100;1:88 | 0.4681 | 0.2258 | 58.2064 | 1.5884 | 0.7503 | True |
| top50_biomarkers | pca_10 | AgglomerativeClustering | 0:138;1:50 | 0.2660 | 0.2612 | 57.8719 | 1.3177 | 0.6893 | True |
| all_acoustic | pca_10 | KMeans | 0:96;1:92 | 0.4894 | 0.2035 | 39.7925 | 1.8597 | 0.9518 | True |
| all_acoustic | pca_10 | GaussianMixture | 0:19;1:169 | 0.1011 | 0.3650 | 19.0565 | 2.1652 | 0.3304 | True |
| all_acoustic | pca_10 | AgglomerativeClustering | 0:106;1:82 | 0.4362 | 0.1796 | 33.6770 | 1.9968 | 0.7563 | True |
| all_acoustic | pca_20 | KMeans | 0:95;1:93 | 0.4947 | 0.1657 | 31.5744 | 2.1189 | 0.9502 | True |
| all_acoustic | pca_20 | GaussianMixture | 0:90;1:98 | 0.4787 | 0.1551 | 29.7123 | 2.1797 | 0.4651 | True |
| all_acoustic | pca_20 | AgglomerativeClustering | 0:105;1:83 | 0.4415 | 0.1505 | 27.1848 | 2.2701 | 0.6978 | True |
| all_acoustic | pca_90 | KMeans | 0:95;1:93 | 0.4947 | 0.1239 | 24.1713 | 2.5093 | 0.9498 | True |
| all_acoustic | pca_90 | GaussianMixture | 0:93;1:95 | 0.4947 | 0.1234 | 24.0801 | 2.5137 | 0.5851 | True |
| all_acoustic | pca_90 | AgglomerativeClustering | 0:98;1:90 | 0.4787 | 0.1064 | 19.2744 | 2.7809 | 0.6566 | True |
| pd_corr_reduced_all_090 | pca_10 | KMeans | 0:119;1:69 | 0.3670 | 0.1914 | 34.8956 | 2.0007 | 0.8762 | True |
| pd_corr_reduced_all_090 | pca_10 | GaussianMixture | 0:14;1:174 | 0.0745 | 0.3946 | 11.8853 | 2.7037 | 0.4570 | False |
| pd_corr_reduced_all_090 | pca_10 | AgglomerativeClustering | 0:90;1:98 | 0.4787 | 0.1479 | 29.5352 | 2.2114 | 0.4588 | True |
| pd_corr_reduced_all_090 | pca_20 | KMeans | 0:72;1:116 | 0.3830 | 0.1378 | 24.1380 | 2.4402 | 0.8802 | True |
| pd_corr_reduced_all_090 | pca_20 | GaussianMixture | 0:23;1:165 | 0.1223 | 0.2912 | 12.6153 | 2.8305 | 0.3902 | True |
| pd_corr_reduced_all_090 | pca_20 | AgglomerativeClustering | 0:87;1:101 | 0.4628 | 0.1072 | 20.0649 | 2.7161 | 0.5142 | True |
| pd_corr_reduced_all_090 | pca_90 | KMeans | 0:119;1:69 | 0.3670 | 0.0705 | 13.4726 | 3.3786 | 0.8361 | True |
| pd_corr_reduced_all_090 | pca_90 | GaussianMixture | 0:85;1:103 | 0.4521 | 0.0649 | 13.0566 | 3.5429 | 0.3557 | True |
| pd_corr_reduced_all_090 | pca_90 | AgglomerativeClustering | 0:55;1:133 | 0.2926 | 0.0687 | 11.4009 | 3.4793 | 0.3968 | True |
| pd_corr_reduced_all_095 | pca_10 | KMeans | 0:97;1:91 | 0.4840 | 0.2004 | 42.3950 | 1.8208 | 0.8968 | True |
| pd_corr_reduced_all_095 | pca_10 | GaussianMixture | 0:19;1:169 | 0.1011 | 0.3479 | 22.3577 | 1.9345 | 0.5015 | True |
| pd_corr_reduced_all_095 | pca_10 | AgglomerativeClustering | 0:106;1:82 | 0.4362 | 0.1832 | 35.6731 | 1.9721 | 0.7251 | True |
| pd_corr_reduced_all_095 | pca_20 | KMeans | 0:98;1:90 | 0.4787 | 0.1555 | 31.0875 | 2.1494 | 0.8865 | True |
| pd_corr_reduced_all_095 | pca_20 | GaussianMixture | 0:83;1:105 | 0.4415 | 0.1416 | 28.7010 | 2.2290 | 0.5615 | True |
| pd_corr_reduced_all_095 | pca_20 | AgglomerativeClustering | 0:92;1:96 | 0.4894 | 0.1398 | 25.3074 | 2.3774 | 0.6092 | True |
| pd_corr_reduced_all_095 | pca_90 | KMeans | 0:96;1:92 | 0.4894 | 0.0950 | 19.4158 | 2.8700 | 0.8905 | True |
| pd_corr_reduced_all_095 | pca_90 | GaussianMixture | 0:98;1:90 | 0.4787 | 0.1003 | 19.3555 | 2.8862 | 0.5408 | True |
| pd_corr_reduced_all_095 | pca_90 | AgglomerativeClustering | 0:88;1:100 | 0.4681 | 0.0785 | 15.6307 | 3.1724 | 0.6453 | True |
| pd_corr_reduced_top50_090 | scaled_only | KMeans | 0:76;1:112 | 0.4043 | 0.1930 | 47.4417 | 1.8023 | 0.9541 | True |
| pd_corr_reduced_top50_090 | scaled_only | GaussianMixture | 0:99;1:89 | 0.4734 | 0.1749 | 42.9394 | 1.9353 | 0.6744 | True |
| pd_corr_reduced_top50_090 | scaled_only | AgglomerativeClustering | 0:135;1:53 | 0.2819 | 0.2042 | 39.7451 | 1.8241 | 0.5470 | True |
| pd_corr_reduced_top50_090 | pca_5 | KMeans | 0:76;1:112 | 0.4043 | 0.2860 | 80.2525 | 1.3316 | 0.9575 | True |
| pd_corr_reduced_top50_090 | pca_5 | GaussianMixture | 0:106;1:82 | 0.4362 | 0.2652 | 64.1377 | 1.5109 | 0.6132 | True |
| pd_corr_reduced_top50_090 | pca_5 | AgglomerativeClustering | 0:114;1:74 | 0.3936 | 0.2715 | 74.7946 | 1.3689 | 0.4720 | True |
| pd_corr_reduced_top50_090 | pca_10 | KMeans | 0:76;1:112 | 0.4043 | 0.2347 | 60.0256 | 1.5690 | 0.9551 | True |
| pd_corr_reduced_top50_090 | pca_10 | GaussianMixture | 0:83;1:105 | 0.4415 | 0.1982 | 48.0455 | 1.7841 | 0.5162 | True |
| pd_corr_reduced_top50_090 | pca_10 | AgglomerativeClustering | 0:132;1:56 | 0.2979 | 0.2045 | 44.9818 | 1.6313 | 0.5418 | True |
| pd_high_iqr_top50 | scaled_only | KMeans | 0:42;1:146 | 0.2234 | 0.3894 | 100.6160 | 1.0147 | 0.9236 | True |
| pd_high_iqr_top50 | scaled_only | GaussianMixture | 0:145;1:43 | 0.2287 | 0.3873 | 100.5914 | 1.0238 | 0.8341 | True |
| pd_high_iqr_top50 | scaled_only | AgglomerativeClustering | 0:157;1:31 | 0.1649 | 0.4249 | 95.5011 | 0.9557 | 0.8823 | True |
| pd_high_iqr_top50 | pca_5 | KMeans | 0:145;1:43 | 0.2287 | 0.4133 | 110.8783 | 0.9622 | 0.9182 | True |
| pd_high_iqr_top50 | pca_5 | GaussianMixture | 0:151;1:37 | 0.1968 | 0.2007 | 10.8840 | 3.1554 | 0.6116 | True |
| pd_high_iqr_top50 | pca_5 | AgglomerativeClustering | 0:157;1:31 | 0.1649 | 0.4491 | 104.9483 | 0.9011 | 0.8799 | True |
| pd_high_iqr_top50 | pca_10 | KMeans | 0:42;1:146 | 0.2234 | 0.3953 | 102.5897 | 1.0000 | 0.9236 | True |
| pd_high_iqr_top50 | pca_10 | GaussianMixture | 0:146;1:42 | 0.2234 | 0.3957 | 102.1926 | 1.0056 | 0.7249 | True |
| pd_high_iqr_top50 | pca_10 | AgglomerativeClustering | 0:157;1:31 | 0.1649 | 0.4305 | 97.3225 | 0.9426 | 0.8361 | True |
| pd_high_iqr_top100 | pca_10 | KMeans | 0:139;1:49 | 0.2606 | 0.3227 | 70.9000 | 1.2782 | 0.8918 | True |
| pd_high_iqr_top100 | pca_10 | GaussianMixture | 0:134;1:54 | 0.2872 | 0.2789 | 65.5741 | 1.3427 | 0.2444 | True |
| pd_high_iqr_top100 | pca_10 | AgglomerativeClustering | 0:162;1:26 | 0.1383 | 0.3548 | 57.6856 | 1.0981 | 0.6821 | True |
| pd_high_iqr_top100 | pca_20 | KMeans | 0:139;1:49 | 0.2606 | 0.3047 | 66.0153 | 1.3345 | 0.8799 | True |
| pd_high_iqr_top100 | pca_20 | GaussianMixture | 0:38;1:150 | 0.2021 | 0.1587 | 10.2462 | 3.3405 | 0.0575 | True |
| pd_high_iqr_top100 | pca_20 | AgglomerativeClustering | 0:162;1:26 | 0.1383 | 0.3375 | 53.8661 | 1.1462 | 0.6915 | True |
| pd_high_iqr_top100 | pca_90 | KMeans | 0:139;1:49 | 0.2606 | 0.3309 | 73.4568 | 1.2485 | 0.8916 | True |
| pd_high_iqr_top100 | pca_90 | GaussianMixture | 0:162;1:26 | 0.1383 | 0.3360 | 44.0682 | 1.3778 | 0.2722 | True |
| pd_high_iqr_top100 | pca_90 | AgglomerativeClustering | 0:158;1:30 | 0.1596 | 0.3557 | 64.2719 | 1.0906 | 0.7152 | True |
| pd_high_mad_top50 | scaled_only | KMeans | 0:146;1:42 | 0.2234 | 0.3905 | 106.0877 | 0.9962 | 0.9241 | True |
| pd_high_mad_top50 | scaled_only | GaussianMixture | 0:146;1:42 | 0.2234 | 0.3905 | 106.0877 | 0.9962 | 0.7983 | True |
| pd_high_mad_top50 | scaled_only | AgglomerativeClustering | 0:157;1:31 | 0.1649 | 0.4243 | 100.3805 | 0.9330 | 0.9092 | True |
| pd_high_mad_top50 | pca_5 | KMeans | 0:146;1:42 | 0.2234 | 0.4095 | 113.4490 | 0.9523 | 0.9257 | True |
| pd_high_mad_top50 | pca_5 | GaussianMixture | 0:40;1:148 | 0.2128 | 0.2703 | 12.6789 | 3.3042 | 0.5314 | True |
| pd_high_mad_top50 | pca_5 | AgglomerativeClustering | 0:156;1:32 | 0.1702 | 0.4373 | 106.1373 | 0.9107 | 0.8819 | True |
| pd_high_mad_top50 | pca_10 | KMeans | 0:146;1:42 | 0.2234 | 0.3954 | 107.8128 | 0.9844 | 0.9254 | True |
| pd_high_mad_top50 | pca_10 | GaussianMixture | 0:149;1:39 | 0.2074 | 0.4007 | 105.9008 | 0.9734 | 0.6674 | True |
| pd_high_mad_top50 | pca_10 | AgglomerativeClustering | 0:157;1:31 | 0.1649 | 0.4289 | 101.9691 | 0.9227 | 0.8821 | True |
| pd_high_mad_top100 | pca_10 | KMeans | 0:49;1:139 | 0.2606 | 0.3191 | 72.6621 | 1.2965 | 0.8952 | True |
| pd_high_mad_top100 | pca_10 | GaussianMixture | 0:165;1:23 | 0.1223 | 0.3360 | 42.7850 | 1.3835 | 0.3476 | True |
| pd_high_mad_top100 | pca_10 | AgglomerativeClustering | 0:154;1:34 | 0.1809 | 0.3252 | 63.6541 | 1.1848 | 0.7056 | True |
| pd_high_mad_top100 | pca_20 | KMeans | 0:49;1:139 | 0.2606 | 0.2992 | 66.8402 | 1.3651 | 0.8885 | True |
| pd_high_mad_top100 | pca_20 | GaussianMixture | 0:132;1:56 | 0.2979 | 0.2688 | 65.2384 | 1.4106 | 0.6886 | True |
| pd_high_mad_top100 | pca_20 | AgglomerativeClustering | 0:83;1:105 | 0.4415 | 0.2020 | 48.3940 | 1.7286 | 0.1278 | True |
| pd_high_mad_top100 | pca_90 | KMeans | 0:49;1:139 | 0.2606 | 0.3229 | 73.8368 | 1.2845 | 0.8922 | True |
| pd_high_mad_top100 | pca_90 | GaussianMixture | 0:171;1:17 | 0.0904 | 0.4114 | 38.0184 | 1.4100 | 0.3246 | False |
| pd_high_mad_top100 | pca_90 | AgglomerativeClustering | 0:154;1:34 | 0.1809 | 0.3291 | 64.6949 | 1.1735 | 0.6002 | True |
| pd_sparse_pca_10 | sparse_pca_10 | KMeans | 0:95;1:93 | 0.4947 | 0.1846 | 34.1032 | 1.9918 | 0.6746 | True |
| pd_sparse_pca_10 | sparse_pca_10 | GaussianMixture | 0:37;1:151 | 0.1968 | 0.2059 | 14.3250 | 2.8310 | 0.4254 | True |
| pd_sparse_pca_10 | sparse_pca_10 | AgglomerativeClustering | 0:87;1:101 | 0.4628 | 0.1611 | 28.3709 | 2.1881 | 0.4406 | True |
| pd_sparse_pca_20 | sparse_pca_20 | KMeans | 0:87;1:101 | 0.4628 | 0.1384 | 26.3112 | 2.3391 | 0.6678 | True |
| pd_sparse_pca_20 | sparse_pca_20 | GaussianMixture | 0:44;1:144 | 0.2340 | 0.1657 | 10.5889 | 3.5313 | 0.4299 | True |
| pd_sparse_pca_20 | sparse_pca_20 | AgglomerativeClustering | 0:99;1:89 | 0.4734 | 0.1321 | 22.6970 | 2.4924 | 0.4881 | True |

## 5. 第 6b 阶段全部候选方案客观指标
| 特征方案 | 表示方式 | 聚类方法 | 簇人数 | 最小簇人数 | 最小簇比例 | 轮廓系数 | Calinski-Harabasz | Davies-Bouldin | 稳定性 ARI 均值 | 是否可推荐 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top20_biomarkers | scaled_only | KMeans | 0:17;1:52;2:30;3:7;4:36;5:46 | 7 | 0.0372 | 0.1425 | 43.6599 | 1.6722 | 0.6669 | False |
| top20_biomarkers | scaled_only | GaussianMixture | 0:42;1:7;2:51;3:37;4:34;5:17 | 7 | 0.0372 | 0.1329 | 43.1719 | 1.7215 | 0.5536 | False |
| top20_biomarkers | scaled_only | AgglomerativeClustering | 0:59;1:61;2:16;3:2;4:39;5:11 | 2 | 0.0106 | 0.1724 | 40.3351 | 1.6817 | 0.6094 | False |
| top20_biomarkers | pca_5 | KMeans | 0:38;1:39;2:53;3:6;4:17;5:35 | 6 | 0.0319 | 0.2650 | 75.7029 | 1.3416 | 0.6998 | False |
| top20_biomarkers | pca_5 | GaussianMixture | 0:35;1:13;2:5;3:85;4:18;5:32 | 5 | 0.0266 | 0.0926 | 43.6401 | 1.7294 | 0.3679 | False |
| top20_biomarkers | pca_5 | AgglomerativeClustering | 0:13;1:55;2:34;3:28;4:25;5:33 | 13 | 0.0691 | 0.1823 | 63.9737 | 1.4268 | 0.4582 | True |
| top20_biomarkers | pca_10 | KMeans | 0:30;1:36;2:17;3:47;4:7;5:51 | 7 | 0.0372 | 0.1685 | 51.2075 | 1.5510 | 0.6824 | False |
| top20_biomarkers | pca_10 | GaussianMixture | 0:56;1:9;2:40;3:36;4:31;5:16 | 9 | 0.0479 | 0.1256 | 43.7200 | 1.9423 | 0.4267 | False |
| top20_biomarkers | pca_10 | AgglomerativeClustering | 0:65;1:56;2:21;3:11;4:33;5:2 | 2 | 0.0106 | 0.1852 | 46.5434 | 1.5609 | 0.6154 | False |
| top50_biomarkers | scaled_only | KMeans | 0:22;1:2;2:65;3:16;4:53;5:30 | 2 | 0.0106 | 0.1487 | 36.8790 | 1.7525 | 0.7106 | False |
| top50_biomarkers | scaled_only | GaussianMixture | 0:53;1:9;2:50;3:49;4:1;5:26 | 1 | 0.0053 | 0.1494 | 33.2302 | 1.6224 | 0.6498 | False |
| top50_biomarkers | scaled_only | AgglomerativeClustering | 0:49;1:9;2:62;3:2;4:53;5:13 | 2 | 0.0106 | 0.1656 | 34.6387 | 1.6352 | 0.7311 | False |
| top50_biomarkers | pca_5 | KMeans | 0:17;1:28;2:26;3:50;4:2;5:65 | 2 | 0.0106 | 0.2727 | 77.7494 | 1.1815 | 0.6975 | False |
| top50_biomarkers | pca_5 | GaussianMixture | 0:10;1:24;2:2;3:50;4:18;5:84 | 2 | 0.0106 | 0.1691 | 48.6146 | 1.3445 | 0.5479 | False |
| top50_biomarkers | pca_5 | AgglomerativeClustering | 0:54;1:50;2:67;3:9;4:2;5:6 | 2 | 0.0106 | 0.2828 | 71.1246 | 0.9944 | 0.6676 | False |
| top50_biomarkers | pca_10 | KMeans | 0:36;1:53;2:49;3:37;4:2;5:11 | 2 | 0.0106 | 0.2054 | 51.4603 | 1.3959 | 0.7212 | False |
| top50_biomarkers | pca_10 | GaussianMixture | 0:35;1:58;2:2;3:54;4:25;5:14 | 2 | 0.0106 | 0.1989 | 46.3986 | 1.4682 | 0.6091 | False |
| top50_biomarkers | pca_10 | AgglomerativeClustering | 0:75;1:14;2:48;3:47;4:2;5:2 | 2 | 0.0106 | 0.2137 | 47.3598 | 1.2569 | 0.6926 | False |
| all_acoustic | pca_10 | KMeans | 0:58;1:28;2:10;3:26;4:1;5:65 | 1 | 0.0053 | 0.1990 | 32.4766 | 1.2946 | 0.7647 | False |
| all_acoustic | pca_10 | GaussianMixture | 0:23;1:35;2:37;3:23;4:1;5:69 | 1 | 0.0053 | 0.1491 | 27.4992 | 1.4278 | 0.4014 | False |
| all_acoustic | pca_10 | AgglomerativeClustering | 0:13;1:56;2:73;3:26;4:19;5:1 | 1 | 0.0053 | 0.1878 | 30.7466 | 1.2908 | 0.6775 | False |
| all_acoustic | pca_20 | KMeans | 0:36;1:24;2:1;3:58;4:58;5:11 | 1 | 0.0053 | 0.1413 | 23.3088 | 1.5767 | 0.6634 | False |
| all_acoustic | pca_20 | GaussianMixture | 0:6;1:39;2:44;3:25;4:9;5:65 | 6 | 0.0319 | 0.1607 | 20.9891 | 1.6931 | 0.4881 | False |
| all_acoustic | pca_20 | AgglomerativeClustering | 0:2;1:17;2:33;3:66;4:16;5:54 | 2 | 0.0106 | 0.1310 | 22.2759 | 1.6571 | 0.6380 | False |
| all_acoustic | pca_90 | KMeans | 0:38;1:57;2:12;3:21;4:59;5:1 | 1 | 0.0053 | 0.0864 | 16.4367 | 1.8854 | 0.6505 | False |
| all_acoustic | pca_90 | GaussianMixture | 0:14;1:22;2:27;3:60;4:24;5:41 | 14 | 0.0745 | 0.0898 | 14.2187 | 2.3991 | 0.4682 | True |
| all_acoustic | pca_90 | AgglomerativeClustering | 0:2;1:65;2:73;3:12;4:25;5:11 | 2 | 0.0106 | 0.0794 | 15.5082 | 1.9598 | 0.6113 | False |
| pd_corr_reduced_all_090 | pca_10 | KMeans | 0:62;1:40;2:49;3:1;4:1;5:35 | 1 | 0.0053 | 0.1473 | 26.7650 | 1.3249 | 0.5853 | False |
| pd_corr_reduced_all_090 | pca_10 | GaussianMixture | 0:33;1:98;2:12;3:1;4:43;5:1 | 1 | 0.0053 | 0.1014 | 17.1082 | 1.6165 | 0.3002 | False |
| pd_corr_reduced_all_090 | pca_10 | AgglomerativeClustering | 0:59;1:39;2:1;3:58;4:30;5:1 | 1 | 0.0053 | 0.0909 | 23.6314 | 1.3798 | 0.4572 | False |
| pd_corr_reduced_all_090 | pca_20 | KMeans | 0:43;1:1;2:1;3:69;4:65;5:9 | 1 | 0.0053 | 0.1167 | 16.9583 | 1.5377 | 0.5051 | False |
| pd_corr_reduced_all_090 | pca_20 | GaussianMixture | 0:33;1:19;2:56;3:1;4:42;5:37 | 1 | 0.0053 | 0.0682 | 14.9712 | 2.1424 | 0.2623 | False |
| pd_corr_reduced_all_090 | pca_20 | AgglomerativeClustering | 0:77;1:9;2:1;3:70;4:30;5:1 | 1 | 0.0053 | 0.1127 | 16.1253 | 1.5137 | 0.4583 | False |
| pd_corr_reduced_all_090 | pca_90 | KMeans | 0:57;1:8;2:35;3:42;4:1;5:45 | 1 | 0.0053 | 0.0201 | 8.4993 | 2.6454 | 0.4054 | False |
| pd_corr_reduced_all_090 | pca_90 | GaussianMixture | 0:29;1:29;2:53;3:2;4:31;5:44 | 2 | 0.0106 | 0.0174 | 6.6183 | 3.0829 | 0.2920 | False |
| pd_corr_reduced_all_090 | pca_90 | AgglomerativeClustering | 0:61;1:10;2:44;3:19;4:53;5:1 | 1 | 0.0053 | 0.0061 | 7.9469 | 2.7679 | 0.4135 | False |
| pd_corr_reduced_all_095 | pca_10 | KMeans | 0:62;1:11;2:40;3:1;4:24;5:50 | 1 | 0.0053 | 0.1609 | 32.3098 | 1.3809 | 0.7715 | False |
| pd_corr_reduced_all_095 | pca_10 | GaussianMixture | 0:17;1:31;2:8;3:1;4:73;5:58 | 1 | 0.0053 | 0.1479 | 25.8806 | 1.6677 | 0.4987 | False |
| pd_corr_reduced_all_095 | pca_10 | AgglomerativeClustering | 0:22;1:75;2:8;3:49;4:33;5:1 | 1 | 0.0053 | 0.1933 | 30.0004 | 1.3495 | 0.5994 | False |
| pd_corr_reduced_all_095 | pca_20 | KMeans | 0:44;1:50;2:9;3:41;4:1;5:43 | 1 | 0.0053 | 0.1168 | 21.1733 | 1.6655 | 0.7003 | False |
| pd_corr_reduced_all_095 | pca_20 | GaussianMixture | 0:64;1:33;2:22;3:37;4:1;5:31 | 1 | 0.0053 | 0.0856 | 18.7867 | 2.0755 | 0.4763 | False |
| pd_corr_reduced_all_095 | pca_20 | AgglomerativeClustering | 0:14;1:82;2:9;3:57;4:25;5:1 | 1 | 0.0053 | 0.1533 | 20.2795 | 1.5926 | 0.5968 | False |
| pd_corr_reduced_all_095 | pca_90 | KMeans | 0:52;1:41;2:10;3:55;4:1;5:29 | 1 | 0.0053 | 0.0452 | 11.8002 | 2.3208 | 0.6205 | False |
| pd_corr_reduced_all_095 | pca_90 | GaussianMixture | 0:21;1:26;2:1;3:17;4:80;5:43 | 1 | 0.0053 | 0.0647 | 8.4420 | 2.6756 | 0.2925 | False |
| pd_corr_reduced_all_095 | pca_90 | AgglomerativeClustering | 0:2;1:31;2:77;3:53;4:16;5:9 | 2 | 0.0106 | 0.0737 | 11.1182 | 2.4465 | 0.5481 | False |
| pd_corr_reduced_top50_090 | scaled_only | KMeans | 0:27;1:18;2:39;3:38;4:57;5:9 | 9 | 0.0479 | 0.1423 | 26.9809 | 1.8480 | 0.6849 | False |
| pd_corr_reduced_top50_090 | scaled_only | GaussianMixture | 0:25;1:18;2:55;3:45;4:7;5:38 | 7 | 0.0372 | 0.1222 | 24.8309 | 1.9811 | 0.5484 | False |
| pd_corr_reduced_top50_090 | scaled_only | AgglomerativeClustering | 0:64;1:10;2:61;3:17;4:34;5:2 | 2 | 0.0106 | 0.1525 | 25.4406 | 1.7002 | 0.7152 | False |
| pd_corr_reduced_top50_090 | pca_5 | KMeans | 0:38;1:25;2:52;3:16;4:8;5:49 | 8 | 0.0426 | 0.2660 | 56.0657 | 1.2174 | 0.7551 | False |
| pd_corr_reduced_top50_090 | pca_5 | GaussianMixture | 0:47;1:62;2:10;3:20;4:8;5:41 | 8 | 0.0426 | 0.2241 | 46.2512 | 1.3146 | 0.5970 | False |
| pd_corr_reduced_top50_090 | pca_5 | AgglomerativeClustering | 0:13;1:64;2:2;3:37;4:35;5:37 | 2 | 0.0106 | 0.2323 | 50.7382 | 1.2747 | 0.6268 | False |
| pd_corr_reduced_top50_090 | pca_10 | KMeans | 0:40;1:19;2:56;3:9;4:47;5:17 | 9 | 0.0479 | 0.1930 | 36.9509 | 1.5112 | 0.7032 | False |
| pd_corr_reduced_top50_090 | pca_10 | GaussianMixture | 0:30;1:58;2:67;3:12;4:5;5:16 | 5 | 0.0266 | 0.1647 | 29.4653 | 1.5037 | 0.4957 | False |
| pd_corr_reduced_top50_090 | pca_10 | AgglomerativeClustering | 0:28;1:63;2:55;3:9;4:28;5:5 | 5 | 0.0266 | 0.2001 | 34.3639 | 1.5038 | 0.7298 | False |
| pd_high_iqr_top50 | scaled_only | KMeans | 0:57;1:29;2:33;3:9;4:55;5:5 | 5 | 0.0266 | 0.2531 | 83.2891 | 1.0770 | 0.7298 | False |
| pd_high_iqr_top50 | scaled_only | GaussianMixture | 0:9;1:30;2:31;3:64;4:8;5:46 | 8 | 0.0426 | 0.2352 | 80.8288 | 1.1490 | 0.5724 | False |
| pd_high_iqr_top50 | scaled_only | AgglomerativeClustering | 0:72;1:24;2:53;3:4;4:27;5:8 | 4 | 0.0213 | 0.2471 | 77.1161 | 1.1423 | 0.5660 | False |
| pd_high_iqr_top50 | pca_5 | KMeans | 0:57;1:33;2:5;3:9;4:29;5:55 | 5 | 0.0266 | 0.2971 | 100.8515 | 0.9448 | 0.7237 | False |
| pd_high_iqr_top50 | pca_5 | GaussianMixture | 0:7;1:31;2:87;3:4;4:47;5:12 | 4 | 0.0213 | 0.1530 | 30.1856 | 2.0614 | 0.3836 | False |
| pd_high_iqr_top50 | pca_5 | AgglomerativeClustering | 0:70;1:28;2:51;3:5;4:26;5:8 | 5 | 0.0266 | 0.2570 | 88.4517 | 1.0466 | 0.4719 | False |
| pd_high_iqr_top50 | pca_10 | KMeans | 0:57;1:29;2:9;3:53;4:5;5:35 | 5 | 0.0266 | 0.2641 | 86.7580 | 1.0504 | 0.7144 | False |
| pd_high_iqr_top50 | pca_10 | GaussianMixture | 0:23;1:10;2:72;3:5;4:67;5:11 | 5 | 0.0266 | 0.1812 | 44.2363 | 1.6619 | 0.4133 | False |
| pd_high_iqr_top50 | pca_10 | AgglomerativeClustering | 0:10;1:55;2:4;3:51;4:27;5:41 | 4 | 0.0213 | 0.2460 | 80.6009 | 1.1582 | 0.6214 | False |
| pd_high_iqr_top100 | pca_10 | KMeans | 0:37;1:81;2:8;3:4;4:53;5:5 | 4 | 0.0213 | 0.2554 | 58.7138 | 1.1487 | 0.7178 | False |
| pd_high_iqr_top100 | pca_10 | GaussianMixture | 0:9;1:15;2:9;3:127;4:9;5:19 | 9 | 0.0479 | 0.0107 | 18.5880 | 2.4420 | 0.3966 | False |
| pd_high_iqr_top100 | pca_10 | AgglomerativeClustering | 0:89;1:60;2:4;3:5;4:8;5:22 | 4 | 0.0213 | 0.2529 | 53.8247 | 1.1790 | 0.6849 | False |
| pd_high_iqr_top100 | pca_20 | KMeans | 0:38;1:4;2:72;3:57;4:9;5:8 | 4 | 0.0213 | 0.2198 | 51.9295 | 1.2891 | 0.7048 | False |
| pd_high_iqr_top100 | pca_20 | GaussianMixture | 0:50;1:60;2:16;3:23;4:30;5:9 | 9 | 0.0479 | 0.1904 | 47.1066 | 1.3830 | 0.4912 | False |
| pd_high_iqr_top100 | pca_20 | AgglomerativeClustering | 0:26;1:55;2:28;3:67;4:4;5:8 | 4 | 0.0213 | 0.2018 | 45.3491 | 1.3341 | 0.4731 | False |
| pd_high_iqr_top100 | pca_90 | KMeans | 0:60;1:35;2:8;3:74;4:6;5:5 | 5 | 0.0266 | 0.2690 | 61.1759 | 1.1371 | 0.6973 | False |
| pd_high_iqr_top100 | pca_90 | GaussianMixture | 0:8;1:17;2:120;3:9;4:10;5:24 | 8 | 0.0426 | 0.1240 | 30.9595 | 1.5626 | 0.5365 | False |
| pd_high_iqr_top100 | pca_90 | AgglomerativeClustering | 0:75;1:70;2:4;3:5;4:8;5:26 | 4 | 0.0213 | 0.2308 | 55.0990 | 1.2179 | 0.5614 | False |
| pd_high_mad_top50 | scaled_only | KMeans | 0:58;1:34;2:5;3:9;4:29;5:53 | 5 | 0.0266 | 0.2684 | 97.8025 | 1.0251 | 0.8260 | False |
| pd_high_mad_top50 | scaled_only | GaussianMixture | 0:32;1:9;2:44;3:5;4:71;5:27 | 5 | 0.0266 | 0.2586 | 96.2570 | 1.0621 | 0.5790 | False |
| pd_high_mad_top50 | scaled_only | AgglomerativeClustering | 0:53;1:47;2:9;3:27;4:4;5:48 | 4 | 0.0213 | 0.2401 | 90.8038 | 1.1167 | 0.5738 | False |
| pd_high_mad_top50 | pca_5 | KMeans | 0:29;1:54;2:60;3:31;4:5;5:9 | 5 | 0.0266 | 0.3009 | 112.1657 | 0.9235 | 0.8216 | False |
| pd_high_mad_top50 | pca_5 | GaussianMixture | 0:12;1:21;2:48;3:4;4:98;5:5 | 4 | 0.0213 | 0.1589 | 34.7134 | 1.9761 | 0.4311 | False |
| pd_high_mad_top50 | pca_5 | AgglomerativeClustering | 0:53;1:52;2:28;3:9;4:4;5:42 | 4 | 0.0213 | 0.2687 | 104.1187 | 1.0331 | 0.6766 | False |
| pd_high_mad_top50 | pca_10 | KMeans | 0:9;1:29;2:55;3:5;4:32;5:58 | 5 | 0.0266 | 0.2772 | 101.5311 | 0.9921 | 0.8208 | False |
| pd_high_mad_top50 | pca_10 | GaussianMixture | 0:9;1:34;2:68;3:5;4:52;5:20 | 5 | 0.0266 | 0.2081 | 84.6953 | 1.3120 | 0.5122 | False |
| pd_high_mad_top50 | pca_10 | AgglomerativeClustering | 0:78;1:51;2:27;3:9;4:4;5:19 | 4 | 0.0213 | 0.2600 | 94.1998 | 0.9738 | 0.6085 | False |
| pd_high_mad_top100 | pca_10 | KMeans | 0:29;1:30;2:52;3:58;4:10;5:9 | 9 | 0.0479 | 0.2257 | 58.1959 | 1.2134 | 0.7471 | False |
| pd_high_mad_top100 | pca_10 | GaussianMixture | 0:19;1:28;2:16;3:9;4:55;5:61 | 9 | 0.0479 | 0.1325 | 38.7280 | 1.4753 | 0.2455 | False |
| pd_high_mad_top100 | pca_10 | AgglomerativeClustering | 0:75;1:61;2:10;3:4;4:30;5:8 | 4 | 0.0213 | 0.2276 | 51.7513 | 1.2772 | 0.5671 | False |
| pd_high_mad_top100 | pca_20 | KMeans | 0:58;1:9;2:10;3:29;4:53;5:29 | 9 | 0.0479 | 0.2007 | 50.7239 | 1.3125 | 0.7491 | False |
| pd_high_mad_top100 | pca_20 | GaussianMixture | 0:36;1:57;2:16;3:9;4:37;5:33 | 9 | 0.0479 | 0.1861 | 48.2645 | 1.4145 | 0.4763 | False |
| pd_high_mad_top100 | pca_20 | AgglomerativeClustering | 0:40;1:49;2:35;3:8;4:48;5:8 | 8 | 0.0426 | 0.1799 | 41.6958 | 1.4615 | 0.4279 | False |
| pd_high_mad_top100 | pca_90 | KMeans | 0:27;1:4;2:67;3:10;4:53;5:27 | 4 | 0.0213 | 0.2409 | 59.7726 | 1.1406 | 0.7523 | False |
| pd_high_mad_top100 | pca_90 | GaussianMixture | 0:8;1:12;2:12;3:9;4:72;5:75 | 8 | 0.0426 | 0.1368 | 32.9655 | 1.7932 | 0.2873 | False |
| pd_high_mad_top100 | pca_90 | AgglomerativeClustering | 0:69;1:4;2:30;3:8;4:47;5:30 | 4 | 0.0213 | 0.2027 | 52.0861 | 1.2368 | 0.4336 | False |
| pd_sparse_pca_10 | sparse_pca_10 | KMeans | 0:36;1:30;2:69;3:1;4:51;5:1 | 1 | 0.0053 | 0.1882 | 30.7289 | 1.1602 | 0.7174 | False |
| pd_sparse_pca_10 | sparse_pca_10 | GaussianMixture | 0:33;1:96;2:15;3:1;4:34;5:9 | 1 | 0.0053 | 0.1058 | 19.8829 | 1.5569 | 0.3636 | False |
| pd_sparse_pca_10 | sparse_pca_10 | AgglomerativeClustering | 0:12;1:37;2:74;3:22;4:1;5:42 | 1 | 0.0053 | 0.1713 | 28.5179 | 1.3989 | 0.6071 | False |
| pd_sparse_pca_20 | sparse_pca_20 | KMeans | 0:41;1:67;2:8;3:29;4:1;5:42 | 1 | 0.0053 | 0.1311 | 19.6490 | 1.6173 | 0.5637 | False |
| pd_sparse_pca_20 | sparse_pca_20 | GaussianMixture | 0:29;1:38;2:27;3:1;4:38;5:55 | 1 | 0.0053 | 0.0989 | 16.6752 | 1.8286 | 0.2694 | False |
| pd_sparse_pca_20 | sparse_pca_20 | AgglomerativeClustering | 0:87;1:17;2:52;3:1;4:1;5:30 | 1 | 0.0053 | 0.1284 | 18.6471 | 1.4471 | 0.4966 | False |

## 6. 权重敏感性分析方法
对通过硬约束过滤的候选方案，先对 silhouette、Calinski-Harabasz、反向 Davies-Bouldin、stability ARI 和 balance 五个指标做 min-max 归一化。随后从 Dirichlet(1,1,1,1,1) 随机生成 10000 组非负且和为 1 的权重，逐次计算综合分并记录第一名和前三名频率。稳定性分析使用 50 次 80% PD 受试者重采样；SparsePCA 方案在已拟合的 SparsePCA 表示空间内重聚类以避免重复拟合高耗时表示。

## 7. 第 5b 阶段第一名与前三名频率
判定：相对推荐方案。
| 特征方案 | 表示方式 | 聚类方法 | 第一名频率 | 前三名频率 | 轮廓系数 | Davies-Bouldin | 稳定性 ARI 均值 | 较小簇比例 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top20_biomarkers | pca_5 | KMeans | 0.3731 | 0.6322 | 0.3674 | 1.0711 | 0.9897 | 0.3298 |
| top50_biomarkers | pca_5 | KMeans | 0.2235 | 0.3523 | 0.3073 | 1.2422 | 0.9224 | 0.4681 |
| pd_high_iqr_top50 | pca_5 | AgglomerativeClustering | 0.2099 | 0.2889 | 0.4491 | 0.9011 | 0.8799 | 0.1649 |
| pd_high_mad_top50 | pca_5 | KMeans | 0.1812 | 0.5972 | 0.4095 | 0.9523 | 0.9257 | 0.2234 |
| top50_biomarkers | scaled_only | KMeans | 0.0093 | 0.0892 | 0.2147 | 1.6488 | 0.9211 | 0.5000 |
| pd_high_iqr_top50 | pca_5 | KMeans | 0.0014 | 0.5215 | 0.4133 | 0.9622 | 0.9182 | 0.2287 |
| top50_biomarkers | pca_10 | KMeans | 0.0008 | 0.2181 | 0.2581 | 1.4360 | 0.9250 | 0.4840 |
| all_acoustic | pca_10 | KMeans | 0.0007 | 0.0028 | 0.2035 | 1.8597 | 0.9518 | 0.4894 |
| all_acoustic | pca_20 | KMeans | 0.0001 | 0.0011 | 0.1657 | 2.1189 | 0.9502 | 0.4947 |
| pd_high_mad_top50 | pca_5 | AgglomerativeClustering | 0.0000 | 0.1887 | 0.4373 | 0.9107 | 0.8819 | 0.1702 |
| pd_high_mad_top50 | pca_10 | AgglomerativeClustering | 0.0000 | 0.0540 | 0.4289 | 0.9227 | 0.8821 | 0.1649 |
| top20_biomarkers | pca_10 | KMeans | 0.0000 | 0.0297 | 0.3164 | 1.2437 | 0.9881 | 0.3298 |
| pd_high_mad_top50 | scaled_only | AgglomerativeClustering | 0.0000 | 0.0107 | 0.4243 | 0.9330 | 0.9092 | 0.1649 |
| top20_biomarkers | scaled_only | KMeans | 0.0000 | 0.0068 | 0.2950 | 1.3274 | 0.9870 | 0.3298 |
| top20_biomarkers | pca_5 | AgglomerativeClustering | 0.0000 | 0.0027 | 0.3713 | 1.0343 | 0.8677 | 0.2819 |
| pd_corr_reduced_top50_090 | pca_5 | KMeans | 0.0000 | 0.0024 | 0.2860 | 1.3316 | 0.9575 | 0.4043 |
| pd_high_iqr_top50 | pca_10 | AgglomerativeClustering | 0.0000 | 0.0017 | 0.4305 | 0.9426 | 0.8361 | 0.1649 |
| pd_corr_reduced_top50_090 | pca_10 | KMeans | 0.0000 | 0.0000 | 0.2347 | 1.5690 | 0.9551 | 0.4043 |
| pd_corr_reduced_top50_090 | scaled_only | KMeans | 0.0000 | 0.0000 | 0.1930 | 1.8023 | 0.9541 | 0.4043 |
| all_acoustic | pca_90 | KMeans | 0.0000 | 0.0000 | 0.1239 | 2.5093 | 0.9498 | 0.4947 |

## 8. 第 6b 阶段第一名与前三名频率
判定：稳健主推荐方案。
| 特征方案 | 表示方式 | 聚类方法 | 第一名频率 | 前三名频率 | 最小簇人数 | 最小簇比例 | 轮廓系数 | Davies-Bouldin | 稳定性 ARI 均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top20_biomarkers | pca_5 | AgglomerativeClustering | 0.8714 | 1.0000 | 13 | 0.0691 | 0.1823 | 1.4268 | 0.4582 |
| all_acoustic | pca_90 | GaussianMixture | 0.1286 | 1.0000 | 14 | 0.0745 | 0.0898 | 2.3991 | 0.4682 |

## 9. 原第 5/6 阶段主方案稳健性
第 5 阶段 原主方案 `top20_biomarkers / pca_5 / KMeans` 在无奖励权重敏感性分析中的第一名频率为 37.3%，只能视为相对稳健，应同时报告其他可接受方案。
第 6 阶段 原主方案 `top20_biomarkers / pca_5 / AgglomerativeClustering` 在无奖励权重敏感性分析中的第一名频率为 87.1%，仍可视为稳健。

## 10. 论文表述建议
若原方案第一名频率低于 60%，论文中不应继续写作唯一“最佳方案”，应改为“在若干客观指标和权重设定下表现较好的候选方案”或“多个可接受候选方案”。权重敏感性不是外部临床标签验证，只能说明聚类推荐对主观权重的稳健性。

## 11. 最终建议
第 5b 阶段：相对推荐方案，首位候选为 `top20_biomarkers / pca_5 / KMeans`，第一名频率 37.3%。
第 6b 阶段：稳健主推荐方案，首位候选为 `top20_biomarkers / pca_5 / AgglomerativeClustering`，第一名频率 87.1%。
所有结论仅为 PD 受试者内部语音特征聚类的稳健性补充，不写作真实临床亚型或六症状诊断。