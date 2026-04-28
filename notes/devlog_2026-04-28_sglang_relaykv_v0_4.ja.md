# SGLang RelayKV v0.4 devlog

Date: 2026-04-28  
Repo: `~/work/sglang-relaykv`  
Branch: `relaykv-v0`  
Fork remote: `mine https://github.com/rinsakamo/sglang.git`

## 目的

SGLang fork 上の manual query-aware RelayKV v0 実装について、評価の信頼性を上げるために以下を進めた。

- `same_first_code` だけではなく、`expected_code` との一致を評価する。
- OFF baseline が正解している item だけを RelayKV 劣化評価の主対象にする。
- baseline-correct item を増やすため、より簡単な lookup case を追加する。
- safe launch helper を標準条件として使い、ON/OFF の比較条件を揃える。

## 実装・スクリプト改善

主な対象ファイル:

- `scripts/relaykv_compare_outputs.py`
- `scripts/start_relaykv_server.sh`

### 1. `baseline-scan`

OFF baseline 結果を読み、`expected_code` と `off_first_code` の一致を確認するサブコマンドを追加。

主な出力:

- `item_id`
- `expected_code`
- `off_first_code`
- `off_matches_expected`
- `baseline_correct_item_ids`

これにより、RelayKV ON 評価前に「baseline 自体が正解している item」を切り出せるようになった。

### 2. `expected_code` ベースの report

`report` に以下の評価軸を追加。

- `expected_code`
- `off_matches_expected`
- `on_matches_expected`
- `same_first_code`
- `same_output_ids`
- `baseline_correct_items`
- baseline-correct subset の aggregate summary

重要な整理:

```text
off_matches_expected:
  baseline が評価対象として妥当かを見る指標

on_matches_expected:
  RelayKV ON がタスク正解を維持できたかを見る指標

same_first_code:
  OFF 出力の再現性を見る指標
```

### 3. `code_probe_table_easy`

baseline-correct item を増やすための easy lookup case を追加。

目的:

- 評価パイプラインの smoke test
- safe helper + report + expected_code 指標の健全性確認

結果として baseline も RelayKV ON も 10/10 で一致した。

### 4. `code_probe_table_medium`

`easy` より少し難しく、`small` より baseline-friendly な case として追加。

目的:

- baseline が高確率で正解する評価セットを作る。
- `code_probe_table_small` では baseline-correct が少なすぎる問題を回避する。
- RelayKV ON の expected_code 維持をより安定して評価する。

現状では recommended blocks が `0,1` または `0,1,2` に偏っているため、本格的な long-context cold block retrieval 評価としてはまだ弱い。ただし v0.4 の評価セットとしては十分に機能した。

### 5. safe launch helper

`./scripts/start_relaykv_server.sh` を標準起動方法として使用。

安全側オプション:

```bash
--trust-remote-code
--attention-backend triton
--sampling-backend pytorch
--disable-cuda-graph
--disable-piecewise-cuda-graph
--disable-overlap-schedule
```

RelayKV ON 例:

```bash
./scripts/start_relaykv_server.sh on 0,1,2
```

OFF baseline 例:

```bash
./scripts/start_relaykv_server.sh off
```

## 実験結果

### `code_probe_table_easy`

対象 item:

```text
0100,0110,0120,0130,0140,0150,0160,0170,0180,0190
```

baseline scan:

```text
baseline_correct_item_ids:
0100,0110,0120,0130,0140,0150,0160,0170,0180,0190
```

RelayKV ON report summary:

```text
total_items=10
off_matches_expected: 10/10
on_matches_expected: 10/10
same_first_code: 10/10
same_output_ids: 10/10
baseline_correct_items: 10
baseline_correct_on_matches_expected: 10/10
baseline_correct_same_first_code: 10/10
baseline_correct_same_output_ids: 10/10
```

解釈:

- `easy` は smoke test として完全成功。
- 全 item の recommended blocks が `0,1` であり、retrieval 評価としては簡単すぎる。
- ただし、ON/OFF 比較、expected_code 判定、summary 出力の確認には有用。

### `code_probe_table_medium`

対象 item:

```text
0100,0110,0120,0130,0140,0150,0160,0170,0180,0190,0200,0210
```

baseline scan 結果:

```text
baseline_correct_item_ids:
0100,0110,0120,0130,0140,0150,0160,0170,0180,0190,0200
```

`0210` は baseline 不正解だったため、RelayKV 劣化評価対象から除外。

```text
0210:
  expected_code = KJQ-3472
  off_first_code = KJQ-1852
  off_matches_expected = False
```

RelayKV ON 評価対象:

```text
0100,0110,0120,0130,0140,0150,0160,0170,0180,0190,0200
```

recommended blocks:

```text
0100〜0160:
  recommended_blocks = 0,1

0170〜0200:
  recommended_blocks = 0,1,2
```

RelayKV ON report summary:

```text
total_items=11
off_matches_expected: 11/11
on_matches_expected: 11/11
same_first_code: 11/11
same_output_ids: 10/11
baseline_correct_items: 11
baseline_correct_on_matches_expected: 11/11
baseline_correct_same_first_code: 11/11
baseline_correct_same_output_ids: 10/11
```

JSON summary:

```json
{
  "total_items": 11,
  "off_matches_expected_true": 11,
  "on_matches_expected_true": 11,
  "same_first_code_true": 11,
  "same_output_ids_true": 10,
  "baseline_correct_items": 11,
  "baseline_correct_on_matches_expected_true": 11,
  "baseline_correct_same_first_code_true": 11,
  "baseline_correct_same_output_ids_true": 10
}
```

解釈:

- Qwen2.5-3B baseline が正解できる medium lookup 11件に対して、RelayKV ON は `on_matches_expected` 11/11 を維持した。
- `same_first_code` も 11/11。
- `same_output_ids` は 10/11 で、1件は token列全体が完全一致ではなかった。
- それでも lookup の主要評価値である `SECRET_CODE` は全件維持できた。

## v0.4 の結論

v0.4 では、評価設計が大きく改善した。

特に重要なのは、`same_first_code` だけを見ず、以下を分離したこと。

```text
1. baseline が expected_code に一致しているか
2. RelayKV ON が expected_code を維持しているか
3. RelayKV ON が OFF 出力を再現しているか
```

`code_probe_table_medium` では、baseline-correct subset 11件に対して:

```text
on_matches_expected: 11/11
same_first_code: 11/11
same_output_ids: 10/11
```

となり、manual query-aware RelayKV v0.4 としては良い結果が得られた。

## 注意点

`code_probe_table_easy` と `code_probe_table_medium` は、どちらも recommended blocks がかなり先頭寄り。

```text
easy:
  0,1

medium:
  0,1 または 0,1,2
```

そのため、今回の結果は以下の確認としては強い。

- safe launch 条件で RelayKV ON が壊れていない。
- expected_code 評価・baseline-correct subset 評価が機能する。
- baseline が正解できる lookup では RelayKV ON が主要評価値を維持できる。

一方で、以下の評価としてはまだ弱い。

- 中盤〜後半 cold block retrieval
- block 3〜8 以降への query-aware selection
- neighbor-radius の有効性
- long-context retrieval としての実用性

## 次の課題

### 1. `code_probe_table_medium_spread` または `medium_long` の追加

次に必要なのは、baseline-friendly だが target rows が block 0〜2 以外にも分散する case。

狙い:

```text
baseline-correct: 8件以上
recommended_blocks: 0,1,2 だけでなく 3〜8 以降に分散
small より簡単
easy / medium より retrieval 評価として有効
```

### 2. block group ごとの ON 実行効率化

現在は `plan --server-command` により起動コマンドは出るが、同じ blocks の item を手動でまとめている。

将来的には:

```text
plan が blocks ごとに item を group 化して出力
```

できると便利。

### 3. server-side 実 env の検証

`actual_relaykv_blocks` はクライアント側で明示指定した意図の記録であり、サーバー側 env を直接取得しているわけではない。

ただし、safe helper の起動ログでは以下を確認済み。

```text
RELAYKV_V0_APPLY: true
RELAYKV_V0_RETRIEVAL_BLOCKS: <blocks>
```

今後は server log と output JSON の対応をより明確にする余地がある。

## コミット状況

v0.4 の実装変更はコミット済み。

想定コミット内容:

```text
- baseline-scan command
- code_probe_table_easy
- code_probe_table_medium
- expected_code-aware report summary
- safe server helper / plan improvements
```

## 次回開始時の推奨コマンド

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate

git status
git log --oneline -5
```

次回は `code_probe_table_medium_spread` または `code_probe_table_medium_long` の追加から始めるのがよい。
