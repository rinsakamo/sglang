# SGLang RelayKV v0.6 devlog

Date: 2026-04-28  
Repo: `~/work/sglang-relaykv`  
Branch: `relaykv-v0`  
Fork remote: `mine https://github.com/rinsakamo/sglang.git`

## 目的

v0.5 では `code_probe_table_medium_spread` により、block 0〜9 に分散した lookup case で manual query-aware RelayKV が `expected_code / first_code` を 12/12 維持できることを確認した。

v0.6 ではさらに長い synthetic lookup case を追加し、target rows / recommended blocks がより後方 block に分散した場合に RelayKV v0 がどこまで維持できるかを確認した。

## 実装・改善

主な対象ファイル:

- `scripts/relaykv_compare_outputs.py`

### 1. `plan --group-by-blocks`

`plan --server-command` の出力を block set ごとにまとめる `--group-by-blocks` を追加。

目的:

- 同じ `RELAYKV_V0_RETRIEVAL_BLOCKS` を使う item をまとめる。
- SGLang サーバー再起動回数を減らす。
- multi-item ON 実行の作業負荷を下げる。

出力イメージ:

```bash
# blocks=3,4,5 item_ids=0150,0160
./scripts/start_relaykv_server.sh on 3,4,5
python scripts/relaykv_compare_outputs.py run --label on --case ... --item-id 0150 --relaykv-blocks 3,4,5
python scripts/relaykv_compare_outputs.py run --label on --case ... --item-id 0160 --relaykv-blocks 3,4,5
```

### 2. `code_probe_table_long_spread`

新規 case:

```text
code_probe_table_long_spread
```

目的:

- `medium_spread` よりさらに長い lookup case を作る。
- baseline-friendly を保ちつつ、target rows を block 10〜30 以降にも分散させる。
- manual query-aware RelayKV の block selection 限界を確認する。

対象 item:

```text
0100,0110,0120,0130,0140,0150,0160,0170,
0180,0190,0200,0210,0220,0230,0240,0250
```

## 実験条件

共通条件:

```text
model: Qwen/Qwen2.5-3B-Instruct
backend: TritonAttnBackend
attention-backend: triton
sampling-backend: pytorch
cuda graph: disabled
piecewise cuda graph: disabled
overlap schedule: disabled
temperature: 0.0
max_new_tokens: 32
case: code_probe_table_long_spread
```

サーバー起動は safe helper を使用。

OFF baseline:

```bash
./scripts/start_relaykv_server.sh off
```

RelayKV ON:

```bash
./scripts/start_relaykv_server.sh on <recommended_blocks>
```

## baseline scan

`code_probe_table_long_spread` では、16件すべてで OFF baseline が expected_code と一致した。

```text
baseline_correct_item_ids:
0100,0110,0120,0130,0140,0150,0160,0170,
0180,0190,0200,0210,0220,0230,0240,0250
```

baseline scan 結果:

| item_id | expected_code | off_first_code | off_matches_expected |
|---:|---|---|---|
| 0100 | KJQ-2382 | KJQ-2382 | True |
| 0110 | KJQ-1572 | KJQ-1572 | True |
| 0120 | KJQ-0762 | KJQ-0762 | True |
| 0130 | KJQ-9952 | KJQ-9952 | True |
| 0140 | KJQ-9142 | KJQ-9142 | True |
| 0150 | KJQ-8332 | KJQ-8332 | True |
| 0160 | KJQ-7522 | KJQ-7522 | True |
| 0170 | KJQ-6712 | KJQ-6712 | True |
| 0180 | KJQ-5902 | KJQ-5902 | True |
| 0190 | KJQ-5092 | KJQ-5092 | True |
| 0200 | KJQ-4282 | KJQ-4282 | True |
| 0210 | KJQ-3472 | KJQ-3472 | True |
| 0220 | KJQ-2662 | KJQ-2662 | True |
| 0230 | KJQ-1852 | KJQ-1852 | True |
| 0240 | KJQ-1042 | KJQ-1042 | True |
| 0250 | KJQ-0232 | KJQ-0232 | True |

## b0 / recommended 3-block 評価

`code_probe_table_long_spread` の 16件に対して、`recommend-blocks` の標準 recommended blocks を使って RelayKV ON を実行した。

結果 summary:

```json
{
  "total_items": 16,
  "off_matches_expected_true": 16,
  "on_matches_expected_true": 13,
  "same_first_code_true": 13,
  "same_output_ids_true": 5,
  "baseline_correct_items": 16,
  "baseline_correct_on_matches_expected_true": 13,
  "baseline_correct_same_first_code_true": 13,
  "baseline_correct_same_output_ids_true": 5
}
```

読み替え:

```text
baseline-correct: 16/16
RelayKV ON expected_code 維持: 13/16
same_first_code: 13/16
same_output_ids: 5/16
```

ここで初めて、baseline は全件正解しているが RelayKV selection により一部 item が落ちる、というきれいな評価が取れた。

b0 の失敗 item:

```text
0190
0210
0250
```

## neighbor-radius による回復確認

### r1: 失敗3件に neighbor-radius 1

対象:

```text
0190,0210,0250
```

結果:

| item_id | recommended_blocks | actual_relaykv_blocks | expected_code | on_first_code | on_matches_expected |
|---:|---|---|---|---|---|
| 0190 | 19,20,21 | 18,19,20,21,22 | KJQ-5092 | KJQ-5092 | True |
| 0210 | 24,25,26 | 23,24,25,26,27 | KJQ-3472 | KJQ-2662 | False |
| 0250 | 33,34,35 | 32,33,34,35,36 | KJQ-0232 | KJQ-2382 | False |

summary:

```text
r1:
  0190 回復
  0210 未回復
  0250 未回復
```

### r2: 0210 / 0250 に neighbor-radius 2

対象:

```text
0210,0250
```

結果:

| item_id | recommended_blocks | actual_relaykv_blocks | expected_code | on_first_code | on_matches_expected | same_output_ids |
|---:|---|---|---|---|---|---|
| 0210 | 24,25,26 | 22,23,24,25,26,27,28 | KJQ-3472 | KJQ-3472 | True | True |
| 0250 | 33,34,35 | 31,32,33,34,35,36,37 | KJQ-0232 | KJQ-2382 | False | False |

summary:

```text
r2:
  0210 回復
  0250 未回復
```

## 0250 block sweep

0250 は r2 でも回復しなかったため、target block をより限定した sweep を実施した。

0250 の情報:

```text
expected_code: KJQ-0232
block_range: [34,34]
recommended_blocks: 33,34,35
```

試した設定:

| tag | actual_relaykv_blocks | on_first_code | on_matches_expected |
|---|---|---|---|
| exact34 | 34 | KJQ-2382 | False |
| right3 | 34,35,36 | KJQ-2382 | False |
| right5 | 34,35,36,37,38 | KJQ-2382 | False |

0250 は target block 34 を単独で含めても、右側に寄せても回復しなかった。

これは、失敗原因が単純な「target block が未選択」ではないことを示している。

## v0.6 の集約結果

### b0

```text
long_spread 16件
baseline_correct: 16/16
RelayKV b0 on_matches_expected: 13/16
same_first_code: 13/16
same_output_ids: 5/16
```

### r1/r2 での回復

```text
0190:
  r1 で回復

0210:
  r2 で回復
  same_output_ids も True まで回復

0250:
  r1/r2/exact/right-heavy でも未回復
```

### best effort

```text
b0 成功: 13/16
r1 追加回復: 0190
r2 追加回復: 0210

best effort:
  15/16 on_matches_expected

未回復:
  0250
```

## 解釈

v0.6 では、v0.5 よりも後方 block に分散した case で、RelayKV v0 の品質限界が見え始めた。

重要な点:

```text
1. baseline は 16/16 正解。
2. recommended 3-block では 13/16 維持。
3. neighbor expansion により 0190 と 0210 は回復。
4. 0250 は target block を含めても回復しない。
```

特に 0250 は以下の挙動を示した。

```text
expected: KJQ-0232
ON output: KJQ-2382
```

`KJQ-2382` は先頭 item 0100 の code であり、後方 item 0250 では anchor / 先頭情報への引っ張られが疑われる。

## 現時点の仮説

0250 の失敗は、以下のどれか、または複合の可能性がある。

```text
A. anchor block の影響が強すぎる
B. retrieval block 34 は含まれているが、decode attention が正しい行に十分寄らない
C. selected context 内で先頭 item / anchor 情報が強く、後方 target が負ける
D. recent window / anchor / retrieval の三層構成バランスが後方 lookup に不利
```

単純な neighbor-radius 拡張や target block 単独指定では回復しないため、次の検証は anchor ablation が有力。

## 次の課題

### 1. anchor ablation

0250 について、anchor block を弱める / 無効化する実験を行う。

候補:

```text
anchor=1, retrieval=34
  現状: 失敗

anchor=0, retrieval=34
  先頭バイアスが消えるか確認

anchor=0, retrieval=33,34,35
  推奨3-blockで確認

anchor=0, retrieval=31..37
  r2相当で確認
```

必要な実装:

```text
RELAYKV_V0_ANCHOR_BLOCKS
```

または既存の anchor 設定を env で制御可能にする。

### 2. report に anchor setting を記録

今後の比較では、output JSON meta に以下を残したい。

```json
{
  "relaykv_v0_anchor_blocks": "0",
  "relaykv_v0_retrieval_blocks": "34"
}
```

### 3. server log と output JSON の対応強化

`actual_relaykv_blocks` はクライアント側で指定した intent の記録であり、サーバー内 env の直接取得ではない。

safe helper の起動ログでは env は確認できるが、将来的には run result と server config の対応をより機械可読にしたい。

### 4. request-wise blocks 指定

現状は `RELAYKV_V0_RETRIEVAL_BLOCKS` がサーバー起動時 env で固定される。

そのため blocks を変えるにはサーバー再起動が必要。

将来的には request metadata 経由で item ごとに blocks を渡せるようにしたい。

ただしこれは SGLang 内部の request metadata 経路を触るため、v0 実験結果を固めてから着手するのが安全。

## v0.6 の結論

`code_probe_table_long_spread` により、RelayKV v0 の評価は一段進んだ。

```text
- baseline-correct 16/16 の長め synthetic case を作れた。
- recommended 3-block RelayKV は 13/16 維持。
- neighbor expansion で 15/16 まで回復。
- 最後の 0250 は target block を含めても回復せず、anchor/三層構成の限界が疑われる。
```

次の最有力テーマは **anchor ablation**。

## 次回開始時の推奨コマンド

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate

git status
git log --oneline -5
```

次回は以下から始めるのがよい。

```bash
grep -R "ANCHOR\|anchor_blocks\|anchor" -n \
  python/sglang/srt/model_executor/model_runner.py \
  python/sglang/srt/layers/attention/triton_backend.py \
  scripts/start_relaykv_server.sh \
  scripts/relaykv_compare_outputs.py
```

目的:

```text
anchor_blocks が env で制御可能か確認し、
RELAYKV_V0_ANCHOR_BLOCKS=0 の ablation を可能にする。
```
