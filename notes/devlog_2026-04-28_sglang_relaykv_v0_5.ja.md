# SGLang RelayKV v0.5 devlog

Date: 2026-04-28  
Repo: `~/work/sglang-relaykv`  
Branch: `relaykv-v0`  
Fork remote: `mine https://github.com/rinsakamo/sglang.git`

## 目的

v0.4 では `baseline-scan` と `expected_code` 評価を導入し、`code_probe_table_easy` / `code_probe_table_medium` で RelayKV ON が baseline-correct item の `SECRET_CODE` を維持できることを確認した。

ただし v0.4 の `easy` / `medium` は retrieval blocks が先頭寄りだった。

```text
easy:
  recommended_blocks = 0,1

medium:
  recommended_blocks = 0,1 または 0,1,2
```

そのため v0.5 では、baseline-friendly でありながら target rows がより広い token block に分散する `code_probe_table_medium_spread` を追加し、manual query-aware RelayKV が block 0〜9 にまたがる retrieval selection でも主要評価値を維持できるかを確認した。

## 実装

主な対象ファイル:

- `scripts/relaykv_compare_outputs.py`

### 追加した case

```text
code_probe_table_medium_spread
```

狙い:

- `code_probe_table_medium` より target rows を広く分散させる。
- `code_probe_table_small` よりは baseline-friendly にする。
- `baseline-correct` item を十分に確保する。
- `recommended_blocks` が `0,1,2` だけに偏らないようにする。
- block 0〜9 程度に retrieval selection が分散する状態を作る。

## 実験条件

### 共通条件

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
case: code_probe_table_medium_spread
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

評価対象 item:

```text
0100,0110,0120,0130,0140,0150,0160,0170,0180,0190,0200,0210
```

## baseline scan

`code_probe_table_medium_spread` では、12件すべてで OFF baseline が expected_code と一致した。

```text
baseline_correct_item_ids:
0100,0110,0120,0130,0140,0150,0160,0170,0180,0190,0200,0210
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

## recommended blocks の分散

`plan --server-command` で確認した recommended blocks は以下の通り。

```text
0100,0110: 0,1
0120:      0,1,2
0130:      1,2,3
0140:      2,3,4
0150,0160: 3,4,5
0170:      4,5,6
0180:      5,6,7
0190:      6,7,8
0200,0210: 7,8,9
```

v0.4 の `medium` では `0,1` / `0,1,2` に偏っていたが、v0.5 の `medium_spread` では block 0〜9 に分散した。

これは、manual query-aware RelayKV の cold block retrieval 評価として v0.4 より有効な構成である。

## RelayKV ON 評価

各 item について、recommended blocks を `RELAYKV_V0_RETRIEVAL_BLOCKS` に設定して ON 実行した。

blocks ごとにまとめてサーバーを起動し、対応する item を実行した。

例:

```bash
./scripts/start_relaykv_server.sh on 3,4,5

python scripts/relaykv_compare_outputs.py run \
  --label on \
  --case code_probe_table_medium_spread \
  --out-dir /tmp/relaykv_compare \
  --item-id 0150 \
  --relaykv-blocks 3,4,5
```

最終 report:

```bash
python scripts/relaykv_compare_outputs.py report \
  --case code_probe_table_medium_spread \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0100,0110,0120,0130,0140,0150,0160,0170,0180,0190,0200,0210 \
  --out-json /tmp/relaykv_compare/report_code_probe_table_medium_spread_b0_12items.json \
  --out-md /tmp/relaykv_compare/report_code_probe_table_medium_spread_b0_12items.md
```

## 結果 summary

```json
{
  "total_items": 12,
  "off_matches_expected_true": 12,
  "on_matches_expected_true": 12,
  "same_first_code_true": 12,
  "same_output_ids_true": 11,
  "baseline_correct_items": 12,
  "baseline_correct_on_matches_expected_true": 12,
  "baseline_correct_same_first_code_true": 12,
  "baseline_correct_same_output_ids_true": 11
}
```

読み替え:

```text
total_items: 12
off_matches_expected: 12/12
on_matches_expected: 12/12
same_first_code: 12/12
same_output_ids: 11/12

baseline_correct_items: 12
baseline_correct_on_matches_expected: 12/12
baseline_correct_same_first_code: 12/12
baseline_correct_same_output_ids: 11/12
```

## 解釈

v0.5 の最重要結果:

```text
baseline が全件正解する medium_spread 12件に対して、
manual query-aware RelayKV は block 0〜9 に分散した retrieval selection でも
expected_code / first_code を 12/12 維持した。
```

`same_output_ids` は 11/12 で、1件は token列全体の完全一致ではなかった。

ただし、lookup タスクの主要評価値である `SECRET_CODE` は全件で維持できた。

## v0.4 からの進歩

v0.4:

```text
code_probe_table_easy:
  blocks = 0,1
  on_matches_expected = 10/10

code_probe_table_medium:
  blocks = 0,1 or 0,1,2
  on_matches_expected = 11/11
```

v0.5:

```text
code_probe_table_medium_spread:
  blocks = 0,1 〜 7,8,9
  on_matches_expected = 12/12
  same_first_code = 12/12
  same_output_ids = 11/12
```

v0.5 では、より広い block 範囲で RelayKV selection が機能することを確認できた。

## 現時点の結論

manual query-aware RelayKV v0.5 は、SGLang Triton backend 上で以下を満たした。

```text
1. safe launch 条件で安定して起動できる。
2. baseline-correct item を expected_code で切り出せる。
3. medium_spread 12件で baseline は 12/12 正解。
4. RelayKV ON でも expected_code / first_code を 12/12 維持。
5. retrieval blocks は 0〜9 に分散しており、v0.4 より retrieval 評価として強い。
```

これは、SGLang 内部の完全自動 request-aware selection に進む前段階として、かなり良い v0 実証結果である。

## 注意点

### 1. まだ manual query-aware

現時点では、blocks は外部スクリプトの `recommend-blocks` / `plan` で決めている。

SGLang 内部で request 内容を見て自動的に retrieval blocks を選ぶ段階にはまだ入っていない。

### 2. medium_spread は synthetic case

`code_probe_table_medium_spread` は評価用に設計した synthetic lookup case。

実プロンプトや自然文 long-context で同等に機能するかは未確認。

### 3. output_ids 完全一致は 11/12

主要評価値は維持できたが、token列全体は1件で完全一致していない。

今後は以下を分けて扱うべき。

```text
on_matches_expected:
  タスク正解

same_first_code:
  主要出力の一致

same_output_ids:
  完全な decode 再現性
```

## 次の課題

### 1. `medium_spread` より長い case

次は block 10〜20 以降にも target rows が出る case を作る。

候補:

```text
code_probe_table_long_spread
code_probe_table_medium_long
```

狙い:

```text
baseline-correct: 8件以上
recommended_blocks: block 0〜20 程度に分散
on_matches_expected: 高維持
```

### 2. plan の group 出力

現在は `plan --server-command` が item ごとの起動コマンドを出すが、同じ blocks の item を手動でまとめている。

改善案:

```text
plan --group-by-blocks
```

出力イメージ:

```bash
# blocks=3,4,5
./scripts/start_relaykv_server.sh on 3,4,5
python scripts/relaykv_compare_outputs.py run ... --item-id 0150 --relaykv-blocks 3,4,5
python scripts/relaykv_compare_outputs.py run ... --item-id 0160 --relaykv-blocks 3,4,5
```

### 3. request-aware selection への橋渡し

manual selection の成功条件が見えてきたため、次は SGLang 内部で以下を検討できる。

```text
- request/prompt から target identifier を抽出
- token span / block span を推定
- retrieval blocks を metadata に入れる
- Triton decode の kv_indices 差し替えに渡す
```

ただし、すぐに内部自動化へ進む前に、もう1段長い synthetic case で v0 の限界を確認するのが安全。

## コミット状況

v0.5 実装変更はコミット対象。

想定コミット:

```text
Add medium-spread RelayKV table case
```

devlog 追加の想定コミット:

```text
Add SGLang RelayKV v0.5 devlog
```

## 次回開始時の推奨コマンド

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate

git status
git log --oneline -5
```

次回は以下のどちらかから始めるのがよい。

```text
A. code_probe_table_long_spread / medium_long を追加して、より遠い block を評価する
B. plan --group-by-blocks を追加して、multi-item ON 実行の作業負荷を下げる
```
