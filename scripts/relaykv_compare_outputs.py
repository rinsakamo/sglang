#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import requests


DEFAULT_URL = "http://127.0.0.1:30000/generate"


def build_prompt(case: str) -> str:
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

    raise ValueError(f"Unknown case: {case}")


def run_request(
    url: str,
    case: str,
    max_new_tokens: int,
    temperature: float,
    output_path: Path,
) -> None:
    prompt = build_prompt(case)

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


def first_diff_index(a: List[int], b: List[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def compare_outputs(off_path: Path, on_path: Path) -> None:
    off = load_json(off_path)
    on = load_json(on_path)

    off_text = off.get("text")
    on_text = on.get("text")

    off_ids = off.get("output_ids") or []
    on_ids = on.get("output_ids") or []

    same_text = off_text == on_text
    same_output_ids = off_ids == on_ids
    diff_idx = first_diff_index(off_ids, on_ids)

    print("=== OFF text ===")
    print(off_text)
    print()
    print("=== ON text ===")
    print(on_text)
    print()
    print("=== compare ===")
    print(f"same_text: {same_text}")
    print(f"same_output_ids: {same_output_ids}")
    print(f"off_num_tokens: {len(off_ids)}")
    print(f"on_num_tokens: {len(on_ids)}")
    print(f"first_diff_index: {diff_idx}")

    if diff_idx is not None:
        print()
        print("=== diff window ===")
        start = max(0, diff_idx - 5)
        end = diff_idx + 10
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
        choices=["repeated_summary", "number_probe", "early_anchor_probe"],
    )
    run_p.add_argument("--url", default=DEFAULT_URL)
    run_p.add_argument("--max-new-tokens", type=int, default=32)
    run_p.add_argument("--temperature", type=float, default=0.0)
    run_p.add_argument("--out-dir", default="/tmp/relaykv_compare")

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("--case", default="repeated_summary")
    cmp_p.add_argument("--out-dir", default="/tmp/relaykv_compare")

    args = parser.parse_args()

    if args.cmd == "run":
        out_path = Path(args.out_dir) / f"{args.case}_{args.label}.json"
        run_request(
            url=args.url,
            case=args.case,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            output_path=out_path,
        )

    elif args.cmd == "compare":
        out_dir = Path(args.out_dir)
        off_path = out_dir / f"{args.case}_off.json"
        on_path = out_dir / f"{args.case}_on.json"
        compare_outputs(off_path, on_path)


if __name__ == "__main__":
    main()