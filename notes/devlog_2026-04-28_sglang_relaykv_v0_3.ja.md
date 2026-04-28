# SGLang RelayKV v0.3 devlog

Date: 2026-04-28  
Repo: `~/work/sglang-relaykv`  
Branch: `relaykv-v0`  
Fork: `https://github.com/rinsakamo/sglang`

## 目的

SGLang Triton backend 上の manual query-aware RelayKV v0.2 を、より検証しやすい v0.3 へ進めた。

主な目的は以下。

- SGLang サーバー起動条件を安全側に固定する
- `recommend-blocks` / `plan` に neighbor expansion を追加する
- `report` に `expected_code` と expected 一致判定を追加する
- `same_first_code` だけでなく、baseline 正解 subset を分けて評価する
- 実験結果の再現性と解釈可能性を上げる

## 実装・スクリプト変更

主に以下を追加・改善した。

### 1. `scripts/start_relaykv_server.sh`

RelayKV 実験用の SGLang 起動ヘルパーを追加した。

代表的な使い方:

```bash
./scripts/start_relaykv_server.sh off
./scripts/start_relaykv_server.sh on 16,17,18
```

ON mode では以下を設定する。

```bash
RELAYKV_V0_APPLY=true
RELAYKV_V0_RETRIEVAL_BLOCKS=<blocks>
```

起動オプションは、RelayKV v0 の検証を優先して安全側に固定した。

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --attention-backend triton \
  --trust-remote-code \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --disable-overlap-schedule
```

これにより、CUDA graph replay / piecewise CUDA graph / overlap schedule による検証上の揺れを避ける方針にした。

### 2. `recommend-blocks --neighbor-radius`

`recommend-blocks` に neighbor expansion を追加した。

例:

```bash
python scripts/relaykv_compare_outputs.py recommend-blocks \
  --case code_probe_table_small \
  --item-id 0180 \
  --neighbor-radius 2 \
  --print-env
```

出力例:

```text
recommended_blocks: 13,14,15
expanded_blocks: 11,12,13,14,15,16,17
```

`--neighbor-radius N` は、推奨 block 群の `min(blocks)-N` から `max(blocks)+N` までを連続範囲として展開する。

### 3. `plan --neighbor-radius --server-command`

`plan` から、起動ヘルパーとクライアント実行コマンドをまとめて出せるようにした。

例:

```bash
python scripts/relaykv_compare_outputs.py plan \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0160,0180,0240 \
  --neighbor-radius 2 \
  --server-command
```

出力例:

```bash
# item_id=0180 recommended_blocks=13,14,15
# item_id=0180 expanded_blocks=11,12,13,14,15,16,17
./scripts/start_relaykv_server.sh on 11,12,13,14,15,16,17
python scripts/relaykv_compare_outputs.py run --label on --case code_probe_table_small --out-dir /tmp/relaykv_compare --item-id 0180 --relaykv-blocks 11,12,13,14,15,16,17 --tag r2
```

`--item-id` のゼロ埋めも保持するように修正した。

### 4. `run --relaykv-blocks --tag`

ON 実行結果 JSON に、意図した RelayKV blocks を保存できるようにした。

例:

```bash
python scripts/relaykv_compare_outputs.py run \
  --label on \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-id 0220 \
  --relaykv-blocks 16,17,18 \
  --tag r0
```

保存メタデータ例:

```json
{
  "relaykv_v0_apply": "1",
  "relaykv_blocks_arg": "16,17,18",
  "relaykv_v0_retrieval_blocks": "16,17,18"
}
```

`--tag` により、`on_b5.json`, `on_r2.json`, `on_r2_safe.json` のように、異なる block 幅の結果を上書きせず保存できるようにした。

### 5. `report` の評価列追加

`report` に以下を追加した。

- `actual_relaykv_blocks`
- `expected_code`
- `off_matches_expected`
- `on_matches_expected`
- aggregate summary

これにより、単なる OFF/ON 一致ではなく、baseline がそもそも正解しているかを分けて見られるようになった。

## 実験条件

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Backend: `TritonAttnBackend`
- Case: `code_probe_table_small`
- Items: `0160, 0180, 0200, 0210, 0220, 0240`
- Temperature: `0.0`
- Max new tokens: `32`
- RelayKV v0 mode: manual query-aware block selection
- Safe launch options:
  - `--attention-backend triton`
  - `--sampling-backend pytorch`
  - `--disable-cuda-graph`
  - `--disable-piecewise-cuda-graph`
  - `--disable-overlap-schedule`

## b3 report with expected_code

Command:

```bash
python scripts/relaykv_compare_outputs.py report \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids 0160,0180,0200,0210,0220,0240 \
  --out-json /tmp/relaykv_compare/report_code_probe_table_small_b3_6items_summary.json \
  --out-md /tmp/relaykv_compare/report_code_probe_table_small_b3_6items_summary.md
```

Summary:

```text
summary: total_items=6
off_matches_expected: 3/6
on_matches_expected: 3/6
same_first_code: 3/6
same_output_ids: 3/6
baseline_correct_items: 3
baseline_correct_on_matches_expected: 3/3
baseline_correct_same_first_code: 3/3
```

## 重要な評価整理

今回の最大の整理点は、`same_first_code` だけでは RelayKV の品質評価として不十分だと分かったこと。

### 指標の意味

```text
off_matches_expected:
  baseline が expected_code に一致しているか。
  これが false の item は、baseline 自体が lookup に失敗している。

on_matches_expected:
  RelayKV ON が expected_code に一致しているか。
  実タスク正解に近い指標。

same_first_code:
  RelayKV ON が OFF baseline の first_code を再現したか。
  baseline が誤答している場合は「誤答再現」になる。
```

## Item別結果

### baseline 正解 item

| item_id | expected_code | off_first_code | on_first_code | same_output_ids | same_first_code | on_matches_expected |
|---:|---|---|---|---:|---:|---:|
| 0200 | KJQ-4282 | KJQ-4282 | KJQ-4282 | True | True | True |
| 0210 | KJQ-3472 | KJQ-3472 | KJQ-3472 | True | True | True |
| 0220 | KJQ-2662 | KJQ-2662 | KJQ-2662 | True | True | True |

baseline が正解している 3 件では、RelayKV ON も expected code / first_code / output_ids をすべて維持した。

### baseline 不正解 item

| item_id | expected_code | off_first_code | b3 on_first_code | 評価扱い |
|---:|---|---|---|---|
| 0160 | KJQ-7522 | KJQ-3528 | KJQ-4603 | baseline 不正解のため品質劣化評価から分離 |
| 0180 | KJQ-5902 | KJQ-0180 | KJQ-0182 | baseline 不正解のため品質劣化評価から分離 |
| 0240 | KJQ-1042 | KJQ-3123 | KJQ-3472 | baseline 不正解のため品質劣化評価から分離 |

これらは RelayKV が壊した item というより、Qwen2.5-3B baseline が expected_code に到達していない item と扱うべき。

## neighbor expansion の結果

失敗組に対して、neighbor expansion を試した。

### 0180

- recommended: `13,14,15`
- r1 / 5-block: `12,13,14,15,16` は失敗
- r2 / 7-block: `11,12,13,14,15,16,17` は OFF first_code 再現に成功

safe retry:

```text
actual_relaykv_blocks = 11,12,13,14,15,16,17
same_output_ids = False
same_first_code = True
off_first_code = KJQ-0180
on_first_code = KJQ-0180
first_diff_index = 23
```

ただし expected_code は `KJQ-5902` なので、これは expected 正解ではなく OFF 誤答の再現。

### 0240

- recommended: `17,18,19`
- r1 / 5-block: `16,17,18,19,20` で OFF first_code 再現
- r2 / 7-block: `15,16,17,18,19,20,21` でも OFF first_code 再現

ただし expected_code は `KJQ-1042` なので、これも expected 正解ではなく OFF 誤答の再現。

### 0160

- recommended: `11,12,13`
- r1 / 5-block: `10,11,12,13,14` は safe 条件で失敗
- r2 / 7-block: `9,10,11,12,13,14,15` も safe 条件で失敗

`0160` は expected_code が `KJQ-7522` だが、OFF baseline が `KJQ-3528` なので、そもそも baseline 不正解。

## 解釈

v0.3 時点での確実な結論は以下。

```text
Qwen2.5-3B baseline が正解できる code_probe_table_small item に限れば、
manual query-aware RelayKV 3-block selection は 3/3 で output_ids / first_code / expected_code を維持した。
```

一方で、baseline 不正解 item を含めると、`same_first_code` は「正解維持」ではなく「baseline 出力再現」を見ているだけになる。

そのため、以後の評価では以下を分ける必要がある。

```text
1. baseline-correct subset:
   RelayKV が正解出力を維持できるかを見る主評価対象。

2. baseline-failed subset:
   OFF 誤答を再現するか、または ON が expected に近づくかを見る参考対象。
```

## 実装上の確認

安全オプション付き起動ヘルパーで以下を確認済み。

- `RELAYKV_V0_APPLY=true` が設定される
- `RELAYKV_V0_RETRIEVAL_BLOCKS=<blocks>` が設定される
- `server_args` で `attention_backend='triton'`
- `sampling_backend='pytorch'`
- `disable_cuda_graph=True`
- `disable_piecewise_cuda_graph=True`
- `disable_overlap_schedule=True`
- 実リクエスト時に `RelayKV v0 APPLY selected kv indices` が出る
- retrieval block range が意図した blocks に対応している

例: blocks `10,11,12,13,14` では、retrieval span が以下になった。

```text
block 10: [2560, 2816)
block 11: [2816, 3072)
block 12: [3072, 3328)
block 13: [3328, 3584)
block 14: [3584, 3840)
```

## 今後の方針

### 1. 評価セットを増やす

今の6件では、baseline 正解が3件だけ。次は baseline が正解できる item を増やして評価したい。

候補:

```text
item_ids を 0100〜0300 などで広げる
まず OFF baseline を一括取得
expected_code と一致する item だけを primary eval にする
```

### 2. baseline-correct subset 専用レポート

`report` の summary で baseline-correct subset は見えるようになったが、次は baseline-correct item のみを抽出するオプションがあると便利。

候補:

```bash
python scripts/relaykv_compare_outputs.py report \
  --case code_probe_table_small \
  --out-dir /tmp/relaykv_compare \
  --item-ids ... \
  --only-baseline-correct
```

### 3. adaptive block width

単純な neighbor expansion は万能ではない。

```text
0180: r2 で OFF 再現
0240: r1/r2 で OFF 再現
0160: r1/r2 でも OFF 再現できず
```

固定 radius より、以下のような adaptive selection が必要そう。

- target block の近傍密度
- expected row / item の token range
- 周辺ブロックに別コードが多すぎる場合の contamination risk
- anchor / recent の干渉
- block幅を広げるか狭めるかの判定

### 4. SGLang 内部 request-aware selection はまだ後回し

現時点では外側 script で評価軸を固める段階。

次に SGLang 内部 selection に進む前に、以下を先に固めるべき。

- baseline-correct item の数を増やす
- safe launch 条件の統一
- expected/on/off の評価レポート
- block width と品質の関係

## コミット済み内容

このセッションでは以下の方向でコミット済み。

- RelayKV report aggregate summary
- expected_code / expected match 列追加
- actual_relaykv_blocks 追加
- `--tag` / `--relaykv-blocks` による結果管理
- neighbor expansion for block recommendations
- server helper and safe launch options

## 次セッションへの引き継ぎ

次にやるなら、最優先は以下。

```bash
cd ~/work/sglang-relaykv
source .venv/bin/activate

git status
git log --oneline -5
```

その後、baseline-correct item を増やすために、より多い item_id で OFF baseline を取得し、`expected_code` と一致する item を抽出する。

現時点の評価方針:

```text
primary metric:
  baseline-correct subset における on_matches_expected / same_first_code / same_output_ids

secondary metric:
  baseline-failed subset における same_first_code と on_matches_expected
```
