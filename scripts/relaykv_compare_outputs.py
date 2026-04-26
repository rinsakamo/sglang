#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List
from transformers import AutoTokenizer

import requests


DEFAULT_URL = "http://127.0.0.1:30000/generate"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_BLOCK_SIZE = 256


def build_prompt(case: str, item_id: int | None = None) -> str:
    if case == "repeated_summary":
        text = " ".join(["RelayKV test sentence."] * 1200)
        return text + "\n\nSummarize the above in one short sentence."

    if case == "number_probe":
        chunks = []
        for i in range(1200):
            chunks.append(f"Item {i}: RelayKV test sentence number {i}.")
        text = "\n".join(chunks)
        return text + "\n\nWhat is the number in Item 1000? Answer briefly."

    if case == "early_anchor_probe":
        chunks = ["IMPORTANT ANCHOR FACT: The secret code is BLUE-17."]
        for i in range(1200):
            chunks.append(f"Filler line {i}: RelayKV test sentence.")
        text = "\n".join(chunks)
        return text + "\n\nWhat is the secret code? Answer briefly."

    if case == "code_probe_unselected":
        chunks = []
        for i in range(1200):
            # Pseudo-random code that is hard to infer from item number.
            code = f"KJQ-{(i * 7919 + 482) % 10000:04d}"
            chunks.append(f"Item {i}: secret code = {code}.")
        text = "\n".join(chunks)
        return text + "\n\nWhat is the secret code in Item 1000? Answer only the code."

    if case == "code_probe_retrieval":
        chunks = []
        for i in range(1200):
            code = f"KJQ-{(i * 7919 + 482) % 10000:04d}"
            chunks.append(f"Item {i}: secret code = {code}.")
        text = "\n".join(chunks)
        # Around Item 210, it is expected that token positions are likely to fall into the retrieval blocks [3072,3840)
        return text + "\n\nWhat is the secret code in Item 210? Answer only the code."

    if case == "code_probe_retrieval_small":
        chunks = []
        for i in range(260):
            code = f"KJQ-{(i * 7919 + 482) % 10000:04d}"
            chunks.append(f"Item {i}: secret code = {code}.")
        text = "\n".join(chunks)
        return text + "\n\nWhat is the secret code in Item 210? Answer only the code."

    if case == "code_probe_table_small":
        if item_id is None:
            item_id = 210
        text = build_table_text(num_items=260)
        return (
            text
            + f"\n\nTask: Look up the row with ITEM_ID={item_id:04d}."
            + "\nReturn exactly one token-like code in the format KJQ-0000."
            + "\nDo not explain."
            + "\nAnswer:"
        )

    raise ValueError(f"Unknown case: {case}")

def code_for_item(i: int) -> str:
    return f"KJQ-{(i * 7919 + 482) % 10000:04d}"


def build_table_text(num_items: int = 260) -> str:
    chunks = []
    for i in range(num_items):
        code = code_for_item(i)
        chunks.append(f"ITEM_ID={i:04d} | SECRET_CODE={code}")
    return "\n".join(chunks)


def recommend_blocks_for_table_item(
    item_id: int,
    model: str = DEFAULT_MODEL,
    block_size: int = DEFAULT_BLOCK_SIZE,
    radius: int = 1,
) -> dict:
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    text = build_table_text(num_items=260)
    code = code_for_item(item_id)
    needle = f"ITEM_ID={item_id:04d} | SECRET_CODE={code}"

    char_pos = text.index(needle)
    token_start = len(tok.encode(text[:char_pos], add_special_tokens=False))
    token_end = token_start + len(tok.encode(needle, add_special_tokens=False))

    block_start = token_start // block_size
    block_end = (token_end - 1) // block_size

    blocks = set()
    for b in range(block_start - radius, block_end + radius + 1):
        if b >= 0:
            blocks.add(b)

    return {
        "item_id": item_id,
        "expected_code": code,
        "token_start": token_start,
        "token_end": token_end,
        "block_start": block_start,
        "block_end": block_end,
        "recommended_blocks": sorted(blocks),
    }


def format_blocks_csv(blocks: List[int]) -> str:
    return ",".join(str(x) for x in blocks)


def make_relaykv_env_exports(blocks: List[int]) -> List[str]:
    blocks_csv = format_blocks_csv(blocks)
    return [
        "export RELAYKV_V0_APPLY=1",
        f"export RELAYKV_V0_RETRIEVAL_BLOCKS={blocks_csv}",
    ]

def run_request(
    url: str,
    case: str,
    max_new_tokens: int,
    temperature: float,
    output_path: Path,
    item_id: int | None = None,
) -> None:
    prompt = build_prompt(case, item_id=item_id)

    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=300,
    )
    response.raise_for_status()

    data = response.json()
    data["_relaykv_compare_meta"] = {
        "case": case,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "prompt_chars": len(prompt),
        "item_id": item_id,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"saved: {output_path}")
    print("text:")
    print(data.get("text", ""))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_file(path: Path) -> Dict[str, Any]:
    return load_json(path)


def parse_item_ids(text: str) -> List[int]:
    return [int(part) for part in text.split(",") if part]


def first_diff_index(a: List[int], b: List[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None

def extract_first_code(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"KJQ-\d{4}", text)
    return m.group(0) if m else None


def summarize_off_on_pair(
    off: Dict[str, Any],
    on: Dict[str, Any],
    item_id: int | None = None,
    path: str | None = None,
    recommended_blocks: List[int] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    off_text = off.get("text")
    on_text = on.get("text")
    off_ids = off.get("output_ids") or []
    on_ids = on.get("output_ids") or []
    off_code = extract_first_code(off_text)
    on_code = extract_first_code(on_text)
    same_output_ids = off_ids == on_ids

    if item_id is None:
        item_id = (off.get("_relaykv_compare_meta") or {}).get("item_id")
    if item_id is None:
        item_id = (on.get("_relaykv_compare_meta") or {}).get("item_id")

    return {
        "path": path,
        "item_id": item_id,
        "recommended_blocks": recommended_blocks,
        "same_output_ids": same_output_ids,
        "same_first_code": off_code == on_code,
        "off_first_code": off_code,
        "on_first_code": on_code,
        "first_diff_index": first_diff_index(off_ids, on_ids),
        "off_num_tokens": len(off_ids),
        "on_num_tokens": len(on_ids),
        "error": error,
    }


def summarize_compare_result(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    meta = data.get("_relaykv_compare_meta") or {}
    item_id = data.get("item_id")
    if item_id is None:
        item_id = meta.get("item_id")

    recommended_blocks = data.get("recommended_blocks")
    if recommended_blocks is None:
        recommended_blocks = meta.get("recommended_blocks")

    return {
        "path": str(path),
        "item_id": item_id,
        "recommended_blocks": recommended_blocks,
        "same_output_ids": data.get("same_output_ids"),
        "same_first_code": data.get("same_first_code"),
        "off_first_code": data.get("off_first_code"),
        "on_first_code": data.get("on_first_code"),
        "first_diff_index": data.get("first_diff_index"),
        "off_num_tokens": data.get("off_num_tokens"),
        "on_num_tokens": data.get("on_num_tokens"),
        "error": data.get("error"),
    }


def format_report_markdown(items: List[Dict[str, Any]]) -> str:
    headers = [
        "path",
        "item_id",
        "recommended_blocks",
        "same_output_ids",
        "same_first_code",
        "off_first_code",
        "on_first_code",
        "first_diff_index",
        "off_num_tokens",
        "on_num_tokens",
        "error",
    ]

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ",".join(str(x) for x in value)
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for item in items:
        row = [fmt(item.get(header)).replace("|", "\\|") for header in headers]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

def compare_outputs(off_path: Path, on_path: Path) -> None:
    off = load_json(off_path)
    on = load_json(on_path)
    summary = summarize_off_on_pair(off, on)
    off_text = off.get("text")
    on_text = on.get("text")
    off_ids = off.get("output_ids") or []
    on_ids = on.get("output_ids") or []
    same_text = off_text == on_text

    print("=== OFF text ===")
    print(off_text)
    print()
    print("=== ON text ===")
    print(on_text)
    print()
    print("=== compare ===")
    print(f"same_text: {same_text}")
    print(f"same_output_ids: {summary['same_output_ids']}")
    print(f"off_first_code: {summary['off_first_code']}")
    print(f"on_first_code: {summary['on_first_code']}")
    print(f"same_first_code: {summary['same_first_code']}")
    print(f"off_num_tokens: {summary['off_num_tokens']}")
    print(f"on_num_tokens: {summary['on_num_tokens']}")
    print(f"first_diff_index: {summary['first_diff_index']}")

    if summary["first_diff_index"] is not None:
        print()
        print("=== diff window ===")
        start = max(0, summary["first_diff_index"] - 5)
        end = summary["first_diff_index"] + 10
        print("OFF ids:", off_ids[start:end])
        print("ON  ids:", on_ids[start:end])

    print()
    print("=== meta ===")
    print("OFF:", json.dumps(off.get("meta_info", {}), ensure_ascii=False, indent=2)[:1000])
    print("ON :", json.dumps(on.get("meta_info", {}), ensure_ascii=False, indent=2)[:1000])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--label", required=True, choices=["off", "on"])
    run_p.add_argument(
        "--case",
        default="repeated_summary",
        choices=[
            "repeated_summary",
            "number_probe",
            "early_anchor_probe",
            "code_probe_unselected",
            "code_probe_retrieval",
            "code_probe_retrieval_small",
            "code_probe_table_small",
        ],
    )
    run_p.add_argument("--url", default=DEFAULT_URL)
    run_p.add_argument("--max-new-tokens", type=int, default=32)
    run_p.add_argument("--temperature", type=float, default=0.0)
    run_p.add_argument("--out-dir", default="/tmp/relaykv_compare")
    run_p.add_argument("--item-id", type=int, default=None)

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("--case", default="repeated_summary")
    cmp_p.add_argument("--out-dir", default="/tmp/relaykv_compare")
    cmp_p.add_argument("--item-id", type=int, default=None)

    rec_p = sub.add_parser("recommend-blocks")
    rec_p.add_argument("--case", default="code_probe_table_small")
    rec_p.add_argument("--item-id", type=int, required=True)
    rec_p.add_argument("--model", default=DEFAULT_MODEL)
    rec_p.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    rec_p.add_argument("--radius", type=int, default=1)
    rec_p.add_argument("--print-env", action="store_true")
    rec_p.add_argument("--json", action="store_true")

    report_p = sub.add_parser("report")
    report_p.add_argument("--inputs", nargs="+", default=None)
    report_p.add_argument("--case", default=None)
    report_p.add_argument("--out-dir", default=None)
    report_p.add_argument("--item-ids", default=None)
    report_p.add_argument("--out-json", default=None)
    report_p.add_argument("--out-md", default=None)

    args = parser.parse_args()

    if args.cmd == "run":
        suffix = f"_{args.item_id:04d}" if args.item_id is not None else ""
        out_path = Path(args.out_dir) / f"{args.case}{suffix}_{args.label}.json"
        run_request(
            url=args.url,
            case=args.case,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_path=out_path,
            item_id=args.item_id,
        )

    elif args.cmd == "compare":
        out_dir = Path(args.out_dir)
        suffix = f"_{args.item_id:04d}" if args.item_id is not None else ""
        off_path = out_dir / f"{args.case}{suffix}_off.json"
        on_path = out_dir / f"{args.case}{suffix}_on.json"
        compare_outputs(off_path, on_path)

    elif args.cmd == "recommend-blocks":
        if args.case != "code_probe_table_small":
            raise ValueError("recommend-blocks currently supports code_probe_table_small only")

        info = recommend_blocks_for_table_item(
            item_id=args.item_id,
            model=args.model,
            block_size=args.block_size,
            radius=args.radius,
        )

        blocks_csv = format_blocks_csv(info["recommended_blocks"])
        env_exports = make_relaykv_env_exports(info["recommended_blocks"])

        if args.json:
            print(
                json.dumps(
                    {
                        "recommended_blocks": info["recommended_blocks"],
                        "relaykv_v0_apply": 1,
                        "relaykv_v0_retrieval_blocks": blocks_csv,
                        "env_exports": env_exports,
                        **info,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        print(f"case: {args.case}")
        print(f"model: {args.model}")
        print(f"item_id: {info['item_id']}")
        print(f"expected_code: {info['expected_code']}")
        print(f"token_range: [{info['token_start']}, {info['token_end']})")
        print(f"block_range: [{info['block_start']}, {info['block_end']}]")
        print(f"recommended_blocks: {blocks_csv}")
        print()
        print("export command:")
        print(f"export RELAYKV_V0_RETRIEVAL_BLOCKS={blocks_csv}")

        if args.print_env:
            print()
            for line in env_exports:
                print(line)

    elif args.cmd == "report":
        items = []
        if args.inputs is not None:
            for input_path in args.inputs:
                path = Path(input_path)
                data = load_json_file(path)
                items.append(summarize_compare_result(data, path))

        if args.case is not None and args.out_dir is not None and args.item_ids is not None:
            out_dir = Path(args.out_dir)
            for item_id in parse_item_ids(args.item_ids):
                suffix = f"_{item_id:04d}"
                off_path = out_dir / f"{args.case}{suffix}_off.json"
                on_path = out_dir / f"{args.case}{suffix}_on.json"
                try:
                    off = load_json_file(off_path)
                    on = load_json_file(on_path)
                    items.append(
                        summarize_off_on_pair(
                            off,
                            on,
                            item_id=item_id,
                            path=f"{off_path},{on_path}",
                        )
                    )
                except FileNotFoundError as exc:
                    items.append(
                        {
                            "path": f"{off_path},{on_path}",
                            "item_id": item_id,
                            "recommended_blocks": None,
                            "same_output_ids": None,
                            "same_first_code": None,
                            "off_first_code": None,
                            "on_first_code": None,
                            "first_diff_index": None,
                            "off_num_tokens": None,
                            "on_num_tokens": None,
                            "error": str(exc),
                        }
                    )

        if not items:
            raise ValueError("report requires --inputs or (--case, --out-dir, and --item-ids)")

        markdown = format_report_markdown(items)
        print(markdown)

        if args.out_json is not None:
            out_json_path = Path(args.out_json)
            out_json_path.parent.mkdir(parents=True, exist_ok=True)
            out_json_path.write_text(
                json.dumps({"items": items}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if args.out_md is not None:
            out_md_path = Path(args.out_md)
            out_md_path.parent.mkdir(parents=True, exist_ok=True)
            out_md_path.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
