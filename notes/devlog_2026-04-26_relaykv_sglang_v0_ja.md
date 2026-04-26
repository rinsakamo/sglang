# Devlog: RelayKV / SGLang v0 integration session

Date: 2026-04-26  
Repository: `~/work/sglang-relaykv`  
Branch: `relaykv-v0`  
Remote push target: `mine/relaykv-v0`  
Main model used in final validation: `Qwen/Qwen2.5-3B-Instruct`  
Attention backend used: `TritonAttnBackend`

## 1. Goal

Move RelayKV three-tier KV policy into SGLang as fast as possible.

Scope for this session:

- Do not modify RadixCache.
- Do not modify `req_to_token_pool` or `token_to_kv_pool` globally.
- Do not touch speculative decode, disaggregation, or CUDA graph replay.
- Focus only on SGLang normal decode path.
- First confirm metadata propagation.
- Then apply selected KV indices in one backend only: Triton attention backend.

## 2. Starting point

Before this session, SGLang already had a first debug commit:

```text
5afbec02d Add RelayKV debug metadata to SGLang decode path
```

Existing changes:

- `ForwardBatch` had `relaykv_debug: Optional[dict] = None`.
- `ModelRunner.forward_decode()` injected static RelayKV debug metadata before `attn_backend.init_forward_metadata(forward_batch)`.
- Initial debug metadata looked like:

```python
forward_batch.relaykv_debug = {
    "mode": "static_three_tier_v0",
    "block_size": 256,
    "recent_window": 256,
    "anchor_blocks": 1,
    "retrieval_blocks": [12, 13, 14],
}
```

## 3. Environment and server setup

Initial run failed because `sglang` was not installed in the active venv. After activating the correct venv, Rust/Cargo and `protoc` were needed for editable install/build.

Runtime issues encountered and resolved:

- Missing `libnuma.so.1` for `sgl_kernel`.
- Piecewise CUDA graph failed because `nvcc` / `CUDA_HOME` was unavailable.
- Overlap schedule triggered JIT CUDA build paths that required CUDA installation.

Stable launch flags used:

```bash
python3 -m sglang.launch_server   --model-path Qwen/Qwen2.5-3B-Instruct   --host 127.0.0.1   --port 30000   --trust-remote-code   --attention-backend triton   --sampling-backend pytorch   --disable-cuda-graph   --disable-piecewise-cuda-graph   --disable-overlap-schedule
```

For RelayKV ON/OFF:

```bash
RELAYKV_V0_APPLY=true
RELAYKV_V0_RETRIEVAL_BLOCKS=16,17,18
```

The server was usually run as:

```bash
RELAYKV_V0_APPLY=true RELAYKV_V0_RETRIEVAL_BLOCKS=16,17,18 python3 -m sglang.launch_server   --model-path Qwen/Qwen2.5-3B-Instruct   --host 127.0.0.1   --port 30000   --trust-remote-code   --attention-backend triton   --sampling-backend pytorch   --disable-cuda-graph   --disable-piecewise-cuda-graph   --disable-overlap-schedule   2>&1 | tee /tmp/sglang_relaykv_qwen25_3b_queryaware_item220.log
```

## 4. Backend identified

Log confirmed decode backend:

```text
RelayKV v0 forward_decode: attn_backend=TritonAttnBackend
RelayKV v0 init_forward_metadata: backend=TritonAttnBackend
```

Therefore all experimental apply code was limited to:

```text
python/sglang/srt/layers/attention/triton_backend.py
```

## 5. Metadata propagation confirmed

Inside `TritonAttnBackend.init_forward_metadata(forward_batch)`, `forward_batch.relaykv_debug` became visible.

Static spans were resolved into:

- anchor span
- retrieval spans
- recent span

Short prompt example:

```text
seq_len=7
spans=[
  {'tier': 'anchor', 'block_id': 0, 'start': 0, 'end': 7},
  {'tier': 'recent', 'block_id': None, 'start': 0, 'end': 7}
]
```

Long prompt example:

```text
seq_len=6002
spans=[
  anchor [0,256),
  retrieval block 12 [3072,3328),
  retrieval block 13 [3328,3584),
  retrieval block 14 [3584,3840),
  recent [5746,6002)
]
```

## 6. Physical KV index construction

Added helpers in `triton_backend.py`:

- `_relaykv_resolve_static_spans(...)`
- `_relaykv_resolve_kv_index_summaries(...)`
- `_relaykv_build_selected_kv_indices(...)`
- `_relaykv_selected_kv_indices_summary(...)`
- `_relaykv_tensor_summary(...)`
- `_relaykv_pool_summary(...)`

Confirmed access to:

```text
forward_batch.req_pool_indices
forward_batch.seq_lens
forward_batch.req_to_token_pool.req_to_token
```

Example pool summary:

```text
req_to_token shape=(2692, 32772), dtype=torch.int32, device=cuda:0
```

Confirmed span-to-KV mapping:

```text
full_len=6002
selected_len=1280
kv_indptr=[0,1280]
```

## 7. Actual Triton KV apply path

Added guarded apply path in Triton decode metadata construction.

Core idea:

```python
if relaykv_apply_ok:
    kv_indices = relaykv_selected_kv_indices.to(torch.int64)

    relaykv_kv_indptr = torch.empty(
        2,
        dtype=kv_indptr.dtype,
        device=kv_indptr.device,
    )
    relaykv_kv_indptr[0] = 0
    relaykv_kv_indptr[1] = kv_indices.numel()
    kv_indptr = relaykv_kv_indptr
```

Guard conditions:

```text
RELAYKV_V0_APPLY
relaykv_debug is not None
bs == 1
relaykv_selected_kv_indices is not None
selected_len > 0
selected_len < full_len
```

Also confirmed `num_kv_splits` uses selected length when apply is on:

```text
original_seq_lens=[21708]
used_seq_lens=[1792]
num_kv_splits_shape=(1,)
sample=[8]
```

## 8. Env var controls added

`triton_backend.py`:

```python
RELAYKV_V0_APPLY = get_bool_env_var("RELAYKV_V0_APPLY", "false")
```

`model_runner.py`:

- Reads `RELAYKV_V0_RETRIEVAL_BLOCKS`.
- If unset, falls back to static default blocks.
- If set, parses comma-separated block IDs.

Conceptual form:

```python
relaykv_retrieval_blocks_env = os.environ.get("RELAYKV_V0_RETRIEVAL_BLOCKS")
if relaykv_retrieval_blocks_env:
    relaykv_retrieval_blocks = [
        int(x.strip())
        for x in relaykv_retrieval_blocks_env.split(",")
        if x.strip()
    ]
else:
    relaykv_retrieval_blocks = [12, 13, 14, 15, 16]
```

Then:

```python
"retrieval_blocks": relaykv_retrieval_blocks
```

This completed **manual query-aware v0**.

## 9. Comparison script added/extended

Main file:

```text
scripts/relaykv_compare_outputs.py
```

Features added:

### Run

```bash
python3 scripts/relaykv_compare_outputs.py run   --label off   --case code_probe_table_small   --item-id 220   --max-new-tokens 32
```

### Compare

```bash
python3 scripts/relaykv_compare_outputs.py compare   --case code_probe_table_small   --item-id 220
```

The compare output includes:

- `same_text`
- `same_output_ids`
- `off_first_code`
- `on_first_code`
- `same_first_code`
- token counts
- first differing output token index
- diff window

Added code extraction:

```python
re.search(r"KJQ-\d{4}", text)
```

### Recommend blocks

```bash
python3 scripts/relaykv_compare_outputs.py recommend-blocks   --case code_probe_table_small   --item-id 220
```

Example output:

```text
item_id: 220
expected_code: KJQ-2662
token_range: [4400, 4419)
block_range: [17, 17]
recommended_blocks: 16,17,18

export command:
export RELAYKV_V0_RETRIEVAL_BLOCKS=16,17,18
```

## 10. Model change

`Qwen/Qwen2.5-1.5B-Instruct` was too weak for exact lookup probes. Baseline often failed, so it was not suitable for evaluating RelayKV quality.

Switched to:

```text
Qwen/Qwen2.5-3B-Instruct
```

This made `code_probe_table_small` usable as a baseline-valid lookup probe.

## 11. Key experiments

### 11.1 Repeated summary

Prompt length around 6000 tokens.

Result:

```text
OFF/ON same_text=True
OFF/ON same_output_ids=True
```

With apply:

```text
full_len≈6002
selected_len=1280
```

### 11.2 Early anchor probe

Result:

```text
The secret code is BLUE-17.
same_text=True
same_output_ids=True
```

This confirmed anchor retention works in the apply path.

### 11.3 Number probe

`Item 1000` was selected-out, but model still answered the number correctly by pattern completion. Token outputs diverged after answer. This probe was judged too easy and not suitable for strict lookup quality.

### 11.4 Code lookup with 1.5B

Baseline failed often. Not useful for RelayKV evaluation.

### 11.5 Code lookup with 3B table prompt

Case:

```text
code_probe_table_small
260 rows
ITEM_ID=0210 | SECRET_CODE=KJQ-3472
```

Baseline OFF succeeded.

## 12. Static block sweeps with item 210

Model: `Qwen/Qwen2.5-3B-Instruct`  
Case: `code_probe_table_small`  
Target: `item_id=210`  
Expected: `KJQ-3472`

Findings:

```text
[10,11,12,13,14,15,16] selected_len=2304 -> same_output_ids=True
[12,13,14,15,16]       selected_len=1792 -> same_output_ids=True
[13,14,15,16]          selected_len=1536 -> same_output_ids=True
[14,15,16]             selected_len=1280 -> same_output_ids=True
[13,14,16]             selected_len=1280 -> same_output_ids=True
[13,14,15]             selected_len=1280 -> wrong / NG
[12,13,14,15]          selected_len=1536 -> wrong / NG
[14,16]                selected_len=1024 -> same_first_code=True, same_output_ids=False
[16]                   selected_len=768  -> same_first_code=True, same_output_ids=False
```

Interpretation:

- Token count alone is not enough.
- Which block is selected matters.
- For item 210, block 16 was especially important.
- Some very small selected sets preserved the answer code but not exact generation.

## 13. Multi-item static test

Items tested:

```text
160, 180, 200, 210, 220, 240
```

With static `[14,15,16]` / selected_len=1280:

```text
item 200: same_output_ids=True
item 210: same_first_code=True but not full exact in one later run
item 220: wrong under static blocks
other items: baseline or ON unstable
```

With static `[12,13,14,15,16]` / selected_len≈1792:

```text
item 200: OK
item 210: OK
item 220: NG because item 220 is in block 17
```

This showed fixed blocks do not generalize across lookup targets.

## 14. Token block analysis

For `code_probe_table_small`, Qwen2.5-3B token positions were:

```text
item=160 code=KJQ-7522 token=[3200,3219) blocks=[12,12] recommended=[11,12,13]
item=180 code=KJQ-5902 token=[3600,3619) blocks=[14,14] recommended=[13,14,15]
item=200 code=KJQ-4282 token=[4000,4019) blocks=[15,15] recommended=[14,15,16]
item=210 code=KJQ-3472 token=[4200,4219) blocks=[16,16] recommended=[15,16,17]
item=220 code=KJQ-2662 token=[4400,4419) blocks=[17,17] recommended=[16,17,18]
item=240 code=KJQ-1042 token=[4800,4819) blocks=[18,18] recommended=[17,18,19]
```

## 15. Manual query-aware validation

Using `recommend-blocks`, item-specific retrieval blocks were used via env var.

### item 200

```text
RELAYKV_V0_RETRIEVAL_BLOCKS=14,15,16
expected: KJQ-4282

same_output_ids=True
same_first_code=True
```

### item 210

```text
RELAYKV_V0_RETRIEVAL_BLOCKS=15,16,17
expected: KJQ-3472

same_output_ids=True
same_first_code=True
```

### item 220

```text
RELAYKV_V0_RETRIEVAL_BLOCKS=16,17,18
expected: KJQ-2662

same_output_ids=False
same_first_code=True
```

Important conclusion:

```text
Fixed blocks failed item 220.
Manual query-aware blocks restored the correct answer code.
```

## 16. Current implementation state

Files changed in the final commit:

```text
python/sglang/srt/model_executor/model_runner.py
python/sglang/srt/layers/attention/triton_backend.py
scripts/relaykv_compare_outputs.py
```

Expected commit message already made by user:

```text
Add manual query-aware RelayKV block selection controls
```

Working tree should be checked at next session:

```bash
cd ~/work/sglang-relaykv
git status
git log --oneline -5
git remote -v
```

## 17. Safety / rollback state

Default should be safe if:

```python
RELAYKV_V0_APPLY = get_bool_env_var("RELAYKV_V0_APPLY", "false")
```

So RelayKV apply is OFF unless env var is set.

Manual query-aware blocks are only active if server is started with:

```bash
RELAYKV_V0_RETRIEVAL_BLOCKS=...
```

## 18. Next session plan

### Step 1: Verify repo state

```bash
cd ~/work/sglang-relaykv
git status
git log --oneline -5
```

### Step 2: Confirm server still works with default OFF

```bash
python3 -m sglang.launch_server   --model-path Qwen/Qwen2.5-3B-Instruct   --host 127.0.0.1   --port 30000   --trust-remote-code   --attention-backend triton   --sampling-backend pytorch   --disable-cuda-graph   --disable-piecewise-cuda-graph   --disable-overlap-schedule
```

### Step 3: Reproduce manual query-aware item 220

```bash
python3 scripts/relaykv_compare_outputs.py recommend-blocks   --case code_probe_table_small   --item-id 220
```

Then launch with:

```bash
RELAYKV_V0_APPLY=true RELAYKV_V0_RETRIEVAL_BLOCKS=16,17,18 python3 -m sglang.launch_server   --model-path Qwen/Qwen2.5-3B-Instruct   --host 127.0.0.1   --port 30000   --trust-remote-code   --attention-backend triton   --sampling-backend pytorch   --disable-cuda-graph   --disable-piecewise-cuda-graph   --disable-overlap-schedule
```

Run:

```bash
python3 scripts/relaykv_compare_outputs.py run   --label on   --case code_probe_table_small   --item-id 220   --max-new-tokens 32

python3 scripts/relaykv_compare_outputs.py compare   --case code_probe_table_small   --item-id 220
```

Expected:

```text
same_first_code=True
```

### Step 4: Next actual implementation target

Move from manual env var query-aware to internal query-aware:

```text
v0.1 current:
  env var controls retrieval blocks

v0.2 next:
  script-assisted query-aware block recommendation

v0.3 later:
  SGLang request-aware retrieval block selection inside ModelRunner / scheduler metadata path
```

Initial v0.2 direction:

- Keep core SGLang apply path unchanged.
- Extend comparison script to:
  - run `recommend-blocks`
  - print exact server env command
  - optionally write a small shell snippet for each item.
- Do not yet attempt automatic request parsing inside SGLang.

## 19. Next-session handoff prompt

```text
RelayKV / SGLang 実装セッションの続きです。

Repo:
- SGLang fork: https://github.com/rinsakamo/sglang
- local dir: ~/work/sglang-relaykv
- branch: relaykv-v0
- remote mine: https://github.com/rinsakamo/sglang.git

Current state:
- Manual query-aware RelayKV v0 has been committed and pushed.
- Main changed files:
  - python/sglang/srt/model_executor/model_runner.py
  - python/sglang/srt/layers/attention/triton_backend.py
  - scripts/relaykv_compare_outputs.py
- `RELAYKV_V0_APPLY` env var controls whether Triton decode applies selected KV.
- `RELAYKV_V0_RETRIEVAL_BLOCKS` env var controls retrieval blocks.
- `scripts/relaykv_compare_outputs.py` supports:
  - run
  - compare
  - recommend-blocks
  - first-code comparison via `KJQ-\d{4}` extraction.

Important results:
- Backend: TritonAttnBackend.
- RelayKV metadata reaches `init_forward_metadata`.
- Token spans are resolved into physical KV indices via `req_to_token_pool.req_to_token`.
- `kv_indices` and `kv_indptr` can be replaced in Triton decode.
- `num_kv_splits` can use selected length.
- Qwen2.5-1.5B was too weak for lookup baseline.
- Qwen2.5-3B works for `code_probe_table_small`.
- For item 200:
  - recommended blocks `14,15,16`
  - same_output_ids=True
  - same_first_code=True
- For item 210:
  - recommended blocks `15,16,17`
  - same_output_ids=True
  - same_first_code=True
- For item 220:
  - fixed blocks failed
  - recommended blocks `16,17,18`
  - same_output_ids=False
  - same_first_code=True
  - correct code recovered: KJQ-2662

Recommended next step:
- Do not touch RadixCache, CUDA graph replay, speculative decode, or disaggregation.
- Continue in Triton backend only.
- Treat current state as v0.1 manual query-aware.
- Build v0.2 scripted query-aware:
  - improve `relaykv_compare_outputs.py recommend-blocks`
  - generate server env commands automatically
  - optionally run a small multi-item report summarizing same_output_ids and same_first_code
- Only after that consider internal SGLang request-aware selection.
```
