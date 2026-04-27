# SGLang RelayKV v0.2 devlog

Date: 2026-04-27  
Repo: `~/work/sglang-relaykv`  
Branch: `relaykv-v0`  
Fork: `https://github.com/rinsakamo/sglang`  
Target backend: `TritonAttnBackend`

## 1. 目的

SGLang fork 上で、RelayKV v0 の manual query-aware selection を、手作業だけでなくスクリプトで再現・比較・集計できる状態にする。

今回の焦点は、SGLang 内部の自動 request-aware selection ではなく、Triton backend 上で確認済みの manual KV selection を外側スクリプトで安定して評価すること。

触らない対象は以下。

- RadixCache
- CUDA graph replay
- speculative decode
- disaggregation
- SGLang runtime の大きな構造変更

## 2. 実装したこと

主な変更対象は以下。

- `scripts/relaykv_compare_outputs.py`
- `scripts/start_relaykv_server.sh` 相当の起動支援スクリプトを追加した場合は、そのスクリプト

### 2.1 `recommend-blocks` の env 出力

`recommend-blocks` で推奨 retrieval blocks を出すだけでなく、SGLang server 起動時に使う env export を出せるようにした。

例:

```bash
python scripts/relaykv_compare_outputs.py recommend-blocks \
  --case code_probe_table_small \
  --item-id 0220 \
  --print-env
```

期待される出力例:

```bash
export RELAYKV_V0_APPLY=1
export RELAYKV_V0_RETRIEVAL_BLOCKS=16,17,18
```

### 2.2 `report` サブコマンド

複数 item の OFF/ON 結果をまとめて比較する `report` を追加した。

初期実装では pre-existing `*_compare.json` を読む想定だったが、既存の `compare` は標準出力のみで保存ファイルを作らないため、`*_off.json` と `*_on.json` のペアを直接読む方式に修正した。

例:

```bash
python scripts/relaykv_compare_outputs.py report \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0160,0180,0200,0210,0220,0240 \
  --out-json /tmp/relaykv_compare/report_code_probe_table_small_b3_6items.json \
  --out-md /tmp/relaykv_compare/report_code_probe_table_small_b3_6items.md
```

report で出す主な列:

- `item_id`
- `recommended_blocks`
- `actual_relaykv_blocks`
- `same_output_ids`
- `same_first_code`
- `off_first_code`
- `on_first_code`
- `first_diff_index`
- `off_num_tokens`
- `on_num_tokens`
- `error`

### 2.3 `plan` サブコマンド

複数 item に対して推奨 blocks とサーバー起動用 env をまとめて出す `plan` を追加した。

例:

```bash
python scripts/relaykv_compare_outputs.py plan \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0160,0180,0200,0210,0220,0240
```

出力例:

```bash
# item_id=0160 recommended_blocks=11,12,13
export RELAYKV_V0_APPLY=1
export RELAYKV_V0_RETRIEVAL_BLOCKS=11,12,13

# item_id=0180 recommended_blocks=13,14,15
export RELAYKV_V0_APPLY=1
export RELAYKV_V0_RETRIEVAL_BLOCKS=13,14,15
```

### 2.4 `--relaykv-blocks`

`run` クライアント側プロセスと SGLang server 側プロセスは別プロセスなので、server 側 env は `run` の JSON metadata には自動では残らない。

そのため、`run` に `--relaykv-blocks` を追加し、意図した RelayKV blocks を `_relaykv_compare_meta` に明示保存できるようにした。

例:

```bash
python scripts/relaykv_compare_outputs.py run \
  --label on \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-id 0220 \
  --relaykv-blocks 16,17,18
```

保存される metadata 例:

```json
{
  "case": "code_probe_table_small",
  "max_new_tokens": 32,
  "temperature": 0.0,
  "prompt_chars": 9484,
  "item_id": 220,
  "relaykv_v0_apply": "1",
  "relaykv_blocks_arg": "16,17,18",
  "relaykv_v0_retrieval_blocks": "16,17,18"
}
```

### 2.5 `--tag`

block 幅違いの ON 結果を上書きしないように、`run` と `report` に `--tag` を追加した。

例:

```bash
python scripts/relaykv_compare_outputs.py run \
  --label on \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-id 0160 \
  --relaykv-blocks 10,11,12,13,14 \
  --tag b5
```

出力ファイル:

```text
/tmp/relaykv_compare/code_probe_table_small_0160_on_b5.json
```

report 側も `--tag b5` を指定すると、OFF は通常の `*_off.json`、ON は `*_on_b5.json` を読む。

## 3. 実験条件

- model: `Qwen/Qwen2.5-3B-Instruct`
- backend: `TritonAttnBackend`
- case: `code_probe_table_small`
- output dir: `/tmp/relaykv_compare`
- temperature: `0.0`
- max_new_tokens: `32`
- item_ids: `0160,0180,0200,0210,0220,0240`

評価指標:

- `same_output_ids`: OFF/ON の生成 token 列が完全一致するか
- `same_first_code`: `KJQ-\d{4}` の first code が一致するか
- `first_diff_index`: 生成 token 列が最初に分岐した位置
- `off_first_code` / `on_first_code`: 抽出された最初の code

## 4. b3: recommended 3-block 結果

実行:

```bash
python scripts/relaykv_compare_outputs.py report \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0160,0180,0200,0210,0220,0240 \
  --out-json /tmp/relaykv_compare/report_code_probe_table_small_b3_6items.json \
  --out-md /tmp/relaykv_compare/report_code_probe_table_small_b3_6items.md
```

結果:

| item_id | recommended_blocks | actual_relaykv_blocks | same_output_ids | same_first_code | off_first_code | on_first_code | first_diff_index | off_num_tokens | on_num_tokens |
|---:|---|---|---:|---:|---|---|---:|---:|---:|
| 160 | 11,12,13 |  | False | False | KJQ-3528 | KJQ-4603 | 4 | 32 | 32 |
| 180 | 13,14,15 |  | False | False | KJQ-0180 | KJQ-0182 | 7 | 32 | 32 |
| 200 | 14,15,16 |  | True | True | KJQ-4282 | KJQ-4282 |  | 32 | 32 |
| 210 | 15,16,17 |  | True | True | KJQ-3472 | KJQ-3472 |  | 29 | 29 |
| 220 | 16,17,18 | 16,17,18 | True | True | KJQ-2662 | KJQ-2662 |  | 32 | 32 |
| 240 | 17,18,19 |  | False | False | KJQ-3123 | KJQ-3472 | 5 | 29 | 32 |

まとめ:

- `same_output_ids=True`: 3/6
- `same_first_code=True`: 3/6
- failed first_code: 0160, 0180, 0240

注記:

- `actual_relaykv_blocks` が空欄の行は、`--relaykv-blocks` を metadata に保存する前に取得した過去結果。
- 0220 は後から `--relaykv-blocks 16,17,18` 付きで取り直したため、actual blocks が残っている。

## 5. b5: 失敗組を 5-block に拡張

b3 で失敗した 0160, 0180, 0240 について、周辺 block を含めた 5-block で ON を取り直した。

実行例:

```bash
python scripts/relaykv_compare_outputs.py run \
  --label on \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-id 0160 \
  --relaykv-blocks 10,11,12,13,14 \
  --tag b5
```

report:

```bash
python scripts/relaykv_compare_outputs.py report \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0160,0180,0240 \
  --tag b5 \
  --out-json /tmp/relaykv_compare/report_code_probe_table_small_b5_failures.json \
  --out-md /tmp/relaykv_compare/report_code_probe_table_small_b5_failures.md
```

結果:

| item_id | recommended_blocks | actual_relaykv_blocks | same_output_ids | same_first_code | off_first_code | on_first_code | first_diff_index | off_num_tokens | on_num_tokens |
|---:|---|---|---:|---:|---|---|---:|---:|---:|
| 160 | 11,12,13 | 10,11,12,13,14 | True | True | KJQ-3528 | KJQ-3528 |  | 32 | 32 |
| 180 | 13,14,15 | 12,13,14,15,16 | False | False | KJQ-0180 | KJQ-3821 | 4 | 32 | 32 |
| 240 | 17,18,19 | 16,17,18,19,20 | True | True | KJQ-3123 | KJQ-3123 |  | 29 | 29 |

まとめ:

- b3 失敗組 3件中、b5 で 2件が first_code 回復。
- 0160 と 0240 は token 列も完全一致まで回復。
- 0180 は b5 でも失敗。

## 6. b7: 0180 を 7-block に拡張

b5 でも失敗した 0180 について、さらに 7-block に拡張した。

実行:

```bash
python scripts/relaykv_compare_outputs.py run \
  --label on \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-id 0180 \
  --relaykv-blocks 11,12,13,14,15,16,17 \
  --tag b7
```

report:

```bash
python scripts/relaykv_compare_outputs.py report \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0180 \
  --tag b7 \
  --out-json /tmp/relaykv_compare/report_code_probe_table_small_0180_b7.json \
  --out-md /tmp/relaykv_compare/report_code_probe_table_small_0180_b7.md
```

結果:

| item_id | recommended_blocks | actual_relaykv_blocks | same_output_ids | same_first_code | off_first_code | on_first_code | first_diff_index | off_num_tokens | on_num_tokens |
|---:|---|---|---:|---:|---|---|---:|---:|---:|
| 180 | 13,14,15 | 11,12,13,14,15,16,17 | False | True | KJQ-0180 | KJQ-0180 | 23 | 32 | 32 |

まとめ:

- 0180 は b7 で `same_first_code=True` まで回復。
- `same_output_ids=False` のままなので完全一致ではない。
- ただし lookup タスクの主要評価値である first code は維持できた。

## 7. 全体の解釈

今回の v0.2 実験では、manual query-aware RelayKV は 3-block 固定だと脆いケースがあることが分かった。

一方で、失敗した 0160 / 0180 / 0240 は、5〜7 block に周辺拡張することで全て `same_first_code=True` まで回復した。

したがって、現時点の重要な観察は以下。

1. Triton decode の KV selection は SGLang 上で実際に効いている。
2. query-aware selection の block 幅と周辺余白が品質に強く効く。
3. 3-block selection は一部の lookup 位置では狭すぎる。
4. 5〜7 block の小さな neighborhood expansion で task-critical code を回復できる可能性がある。
5. 完全な token 列一致までは保証されないが、lookup 型評価の first code は維持できるケースが増える。

研究メモとしては、以下の表現が妥当。

> 3-block selection is too brittle for some lookup positions, while a small neighborhood expansion to 5–7 blocks recovers the task-critical code in all tested failure cases.

## 8. 実装上の確認済み事項

これまでに確認済みの SGLang 側 RelayKV v0 の事実:

- Backend は `TritonAttnBackend`。
- RelayKV metadata は `init_forward_metadata` まで到達する。
- token spans は `req_to_token_pool.req_to_token` から physical KV indices に解決できる。
- Triton decode の `kv_indices` / `kv_indptr` を差し替え可能。
- `num_kv_splits` は selected length に合わせられる。
- `RELAYKV_V0_APPLY` で適用 ON/OFF を制御可能。
- `RELAYKV_V0_RETRIEVAL_BLOCKS` で retrieval blocks を手動指定可能。
- Qwen2.5-1.5B は lookup baseline が弱く、Qwen2.5-3B の方が `code_probe_table_small` で実験しやすい。

## 9. 次の課題

### 9.1 block 幅の自動調整

今回の結果から、固定 3-block ではなく、以下のような方針が必要。

- 基本は recommended center blocks
- 周辺 ±1 block を含める b5
- 不安定ケースでは ±2 block の b7
- item / query position / confidence に応じて幅を変える

### 9.2 actual blocks と recommended blocks の分離

`recommended_blocks` はロジック上の標準推奨。

`actual_relaykv_blocks` は実際に ON server で使った blocks。

今後の report では両方を必ず出し、b3/b5/b7 の比較を明確にする。

### 9.3 SGLang 内部自動 selection はまだ後回し

次にすぐ SGLang 内部 request-aware selection に進むよりも、まずは外側で以下を詰める。

- b3/b5/b7 の成功率
- prompt 長や item 位置ごとの傾向
- first_code 維持率
- full output 一致率
- retrieval blocks の安全余白

その後、内部 selection を入れる。

### 9.4 server 起動支援

毎回 server 側 env を手で入れるのはミスが起きやすい。

`./scripts/start_relaykv_server.sh off` や
`./scripts/start_relaykv_server.sh on 16,17,18`
のような helper があると、実験再現性が上がる。

## 10. 次回開始時の推奨コマンド

状態確認:

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate

git status
git branch --show-current
git log --oneline -5
```

b3 report 再生成:

```bash
python scripts/relaykv_compare_outputs.py report \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0160,0180,0200,0210,0220,0240 \
  --out-json /tmp/relaykv_compare/report_code_probe_table_small_b3_6items.json \
  --out-md /tmp/relaykv_compare/report_code_probe_table_small_b3_6items.md
```

b5 report 再生成:

```bash
python scripts/relaykv_compare_outputs.py report \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0160,0180,0240 \
  --tag b5 \
  --out-json /tmp/relaykv_compare/report_code_probe_table_small_b5_failures.json \
  --out-md /tmp/relaykv_compare/report_code_probe_table_small_b5_failures.md
```

b7 report 再生成:

```bash
python scripts/relaykv_compare_outputs.py report \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0180 \
  --tag b7 \
  --out-json /tmp/relaykv_compare/report_code_probe_table_small_0180_b7.json \
  --out-md /tmp/relaykv_compare/report_code_probe_table_small_0180_b7.md
```

## 11. コミット候補

ここまでの変更をまだコミットしていない場合:

```bash
git add scripts/relaykv_compare_outputs.py scripts/start_relaykv_server.sh
git commit -m "Improve RelayKV scripted comparison workflow"
git push mine relaykv-v0
```

`start_relaykv_server.sh` を追加していない場合:

```bash
git add scripts/relaykv_compare_outputs.py
git commit -m "Improve RelayKV scripted comparison workflow"
git push mine relaykv-v0
```
