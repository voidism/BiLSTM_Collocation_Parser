# BiLSTM Collocation Parser

用 BiLSTM＋ELMo 搭建的 Collocation Parser，可以抓取句子中特定的Collocation Pairs，如名詞<->量詞、動詞<->名詞等等，用以建構Ontology及其他應用。雖直接使用Dependency Parser也能有差不多效果，唯Dependency Parser一次要預測許多不同詞類的arcs，易受其他詞的arcs連結干擾影響正確率，且中文的Dependency Parser表現較差錯誤率稍高。此Project專注抓取Collocation Pairs，先使用unsupervised [word2vec-based簡易方法](<https://github.com/voidism/Chinese_Sentence_Dependency_Analyzer>)有效的得到許多有關連的詞類pairs，透過少量標註(1000句)找出正確率夠大的頻率區間，拿來train後續的BiLSTM Collocation Parser。

## Requirements:

- python >= 3.6 (do not use 3.5)
- pytorch 0.4
- overrides
- matplotlib
- 先到[這裡](https://drive.google.com/drive/folders/1oGEvrpkquWP6vcLtE7XQXDKGrxu62Mne)下載ELMo 的 Pretrain model，放在主資料夾中

## Usage:

### Testing

#### 使用範例：

```bash
python colloc_parser.py write -model_name vn_model.ckpt -in_file in.txt -out_file out.txt  -prefix v_n -timing True -threshold 0.5 -threshold_2 0.6 -show_score True -batch_size 256 -max_len 70
```

#### 說明：

```bash
python colloc_parser.py [mode(執行模式)"print" or "write" -> print是全部印出來，write是寫進去檔案裡] 
        -model_name [model檔案名稱] 
        -in_file [要parse的檔案名稱] 
        -out_file [要write的檔案名稱，若選print模式可不填] 
        -prefix [pair的詞性1及詞性2，用"_"隔開, ex:"nf_n"] 
        -timing [要不要幫你算預計執行時間(True or False)，default: True，
        選True會跑出Time Bar, 只有在write mode有效，否則會跟print出來的東西混在一起] 
        -threshold [抓pair時預測分數的閾值，default: 0.4]
        -threshold_2 [如果希望pair中第二個詞所使用閾值與第一個詞不同，可指定此argument為另一個閾值，
        沒指定則按照-threshold的值當作第二個字的閾值]
        -show_score [要不要顯示每個pair所預測的分數(True or False)，default: True，建議先印出來看看效果，
        再決定要不要提高閾值，可以濾掉一些抓錯的]
        -batch_size [跑程式時的batch size，如果GPU記憶體夠，大一點會跑比較快，如12G RAM最多可以設256 
        ，default: 128]
        -max_len [因ELMo跑的時候如果句子太長，會Out-of-memory，這裡設定一個最大的句子長度，default: 100，
        過長的句子會直接不處理，並在其句首加入"TOO LONG ==>"字樣標示，同時句子長度 1 or 0 的句子也會標上
        "TOO SHORT ==>"]
```

### Visualization

- 可以畫出如下的 score map，只要把 mode 改成 plot 即可
- 畫圖所耗時間較純testing大，建議取一小部分句子來畫圖就好，像是取1000句就會畫很久

![](https://i.imgur.com/i93cSI1.png)

#### 使用範例：

```bash
python colloc_parser.py plot -model_name vn_model.ckpt -in_file in.txt -out_folder ./temp/ -timing True -batch_size 256 -max_len 30
```

#### 說明：

```bash
python colloc_parser.py [mode(執行模式)"print" or "write" -> print是全部印出來，write是寫進去檔案裡] 
        -model_name [model檔案名稱] 
        -in_file [要parse的檔案名稱] 
        -out_folder [輸出圖檔的位置，檔名會自動用句子的內容命名]
        -timing [要不要幫你算預計執行時間(True or False)，default: True] 
        -batch_size [跑程式時的batch size，default: 128]
        -max_len [因ELMo跑的時候如果句子太長，會Out-of-memory，這裡設定一個最大的句子長度，default: 100，
        因為這裡要畫圖，建議句子長度不要超過30]
```

### Training

#### 使用範例：

```bash
python colloc_parser.py train -pretrain_name nf_model.ckpt -save_model_name math_model.ckpt -epochs 10 -batch_size 32 -train_file math_train_data.txt -prefix f_n
```

#### 說明：

```bash
python colloc_parser.py [mode(執行模式)請打"train"] 
        -pretrain_name [pretrain的model檔案名稱，如果沒有指定就重頭train] 
        -save_model_name [train完後model要儲存的檔案名稱]
        -epochs [epoch數量，default:10]
        -batch_size [跑training時的batch size，建議設32]
        -train_file [訓練資料檔案名稱，格式底下會詳述]
        -prefix [訓練資料使用的label標註字母，一定要打才會parse對data，等等底下會詳述]
```

#### 訓練過程：

```
 [ Epoch 1/10 ]
|=================================================>| 98.8%  	ALL: 15.0 s a: 0.6094 b: 0.625  c: 0.837  d: 0.8696
|====================================================>| 105.9% 	ETA: 6.0 s a: 0.4706 b: 0.4706 c: 1.0    d: 1.0
 [ Epoch 2/10 ]
|================================>                 | 63.7%  	ETA: 8.0 s a: 0.6406 b: 0.6562 c: 0.8587 d: 0.8696

```



數值 a, b, c, d分別代表的意思是

- a: pair前面字的precision          e.g. 如Nf-N pair，前面字的precision就是抓量詞Nf的precision

- b: pair前面字的recall

- c: pair後面字的precision

- d: pair後面字的recall

每個epoch第二個timing bar(會跑到105.9%的那個)，代表的是跑Testing set的數值  

由於這個資料是multi-pair，要預測比較難，假如一句話中兩個pair只要一個predict錯就變50%，所以算出來precision/recall會不如直接train single-pair的高。



## Data Format

訓練資料需符合以下格式：  

每行一句話，需斷好詞，並定義好prefix字母，用來標記pair，如例子中是Nf-N pair，就用f跟n當prefix。  

句中每一對pair中第一個字後面放第一個prefix＋數字編號，第二個字後面放第二個prefix＋數字編號。  

如：`有 一 個 邊長 7 公分 的 正 方體 粉筆盒`  
第一個pair是 "個->粉筆盒" 給上編號1號  
變成：`有 一 個f1 邊長 7 公分 的 正 方體 粉筆盒n1`  
第二個pair是 "公分->邊長" 給上編號2號  
變成：`有 一 個f1 邊長n2 7 公分f2 的 正 方體 粉筆盒n1`  

如果有共用的情形發生，就在字尾繼續附加上去即可，如下面的："市值n1n2"，編號順序也不會影響parse結果

```
有 77 枝f1 牛奶 冰棒n1 和 12 枝f2 花生 冰棒n2
有 一 個f1 邊長n2 7 公分f2 的 正 方體 粉筆盒n1
有 一 枝f1 筆n1 用 尺量 起來 是 10 大格 8 小 格
某 雞排 餐車 平均 一 天 可以 賣出 259 塊f1 雞排n1
每 一 個f1 鈍角 三角形n1 中 有 168 個f2 鈍角n2
比 上 週 市值n1n2 約 25兆4927億 元f1 增加 幾億 元f2
水果店 老闆 把 4 箱f1 的 水梨n1 分裝 成 小 盒 禮盒
活動 中心 到 籃球場 的 實際 直線 距離n1 約 幾 公尺f1
爸爸 買 了 一 支f1 刮鬍刀n1n2 付 了 2975 元f2
用 15 個f1 空 寶特瓶n1 可以 換 2 枝f2 棒棒糖n2
```

## Models

目前提供：
- 量詞名詞 (Nf - N) model	`models/NfN_model.ckpt`
- 動詞名詞 (V - N) model	`models/VN_model.ckpt`
- 形容詞名詞 (A - N) model	`models/AN_model.ckpt`
- 名詞名詞 (N - N) model	`models/NN_model.ckpt`

Nf - N 的手標data數量最多，品質較好，其餘手標資料data較少(1000句)，較不穩定

#### 小學數學題 model

位置：`models/math_model.ckpt`

由3000多句小學數學題標註名詞-量詞pair(Nf-N)的句子訓練而來，資料常有數詞量詞一句多pair及倒裝，有比較強的correference能力。

例句：若一簍魚有6條每公斤賣95元台幣

![](https://i.imgur.com/B4AIdgx.png)

----------------

Yung-Sung Chuang, 2019 © Intelligent Agent Systems Lab., Institute of Information Science, Academia Sinica.

