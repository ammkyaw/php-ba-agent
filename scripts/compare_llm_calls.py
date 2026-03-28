#!/usr/bin/env python3

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * p) - 1)
    return ordered[index]


def load_rows(path):
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: no JSONL records found")
    return rows


def summarize(rows):
    durations = [float(row["duration_s"]) for row in rows]
    tokens_in = [float(row.get("tokens_in", 0)) for row in rows]
    tokens_out = [float(row.get("tokens_out", 0)) for row in rows]
    total_tokens = [inp + out for inp, out in zip(tokens_in, tokens_out)]
    output_tps = [out / dur for out, dur in zip(tokens_out, durations) if dur > 0]
    total_tps = [total / dur for total, dur in zip(total_tokens, durations) if dur > 0]

    duration_sum = sum(durations)
    output_sum = sum(tokens_out)
    total_sum = sum(total_tokens)

    return {
        "calls": len(rows),
        "duration_sum": duration_sum,
        "tokens_in_sum": sum(tokens_in),
        "tokens_out_sum": output_sum,
        "tokens_total_sum": total_sum,
        "latency_min": min(durations),
        "latency_median": statistics.median(durations),
        "latency_mean": statistics.mean(durations),
        "latency_p95": percentile(durations, 0.95),
        "latency_max": max(durations),
        "avg_in_per_call": statistics.mean(tokens_in),
        "avg_out_per_call": statistics.mean(tokens_out),
        "median_out_per_call": statistics.median(tokens_out),
        "output_tps_median": statistics.median(output_tps) if output_tps else 0.0,
        "output_tps_mean": statistics.mean(output_tps) if output_tps else 0.0,
        "output_tps_p95": percentile(output_tps, 0.95),
        "output_tps_aggregate": output_sum / duration_sum if duration_sum else 0.0,
        "total_tps_aggregate": total_sum / duration_sum if duration_sum else 0.0,
        "seconds_per_1k_output": (duration_sum / output_sum * 1000) if output_sum else math.inf,
        "seconds_per_1k_total": (duration_sum / total_sum * 1000) if total_sum else math.inf,
    }


def print_summary(label, summary):
    print(label)
    print(f"  calls: {summary['calls']}")
    print(f"  wall time (sum): {summary['duration_sum']:.3f}s")
    print(f"  input tokens: {summary['tokens_in_sum']:.0f}")
    print(f"  output tokens: {summary['tokens_out_sum']:.0f}")
    print(f"  total tokens: {summary['tokens_total_sum']:.0f}")
    print(
        "  latency (min / median / mean / p95 / max): "
        f"{summary['latency_min']:.3f}s / "
        f"{summary['latency_median']:.3f}s / "
        f"{summary['latency_mean']:.3f}s / "
        f"{summary['latency_p95']:.3f}s / "
        f"{summary['latency_max']:.3f}s"
    )
    print(
        "  output tok/s (median / mean / p95 / aggregate): "
        f"{summary['output_tps_median']:.3f} / "
        f"{summary['output_tps_mean']:.3f} / "
        f"{summary['output_tps_p95']:.3f} / "
        f"{summary['output_tps_aggregate']:.3f}"
    )
    print(f"  total tok/s (aggregate): {summary['total_tps_aggregate']:.3f}")
    print(
        "  avg tokens per call (in / out): "
        f"{summary['avg_in_per_call']:.1f} / {summary['avg_out_per_call']:.1f}"
    )
    print(f"  median output tokens per call: {summary['median_out_per_call']:.1f}")
    print(f"  seconds per 1k output tokens: {summary['seconds_per_1k_output']:.3f}")
    print(f"  seconds per 1k total tokens: {summary['seconds_per_1k_total']:.3f}")


def print_comparison(left_label, left_summary, right_label, right_summary):
    print("\ncomparison")
    output_speedup = (
        left_summary["output_tps_aggregate"] / right_summary["output_tps_aggregate"]
        if right_summary["output_tps_aggregate"]
        else math.inf
    )
    total_speedup = (
        left_summary["total_tps_aggregate"] / right_summary["total_tps_aggregate"]
        if right_summary["total_tps_aggregate"]
        else math.inf
    )
    output_efficiency = (
        right_summary["seconds_per_1k_output"] / left_summary["seconds_per_1k_output"]
        if left_summary["seconds_per_1k_output"] not in (0, math.inf)
        else math.inf
    )
    print(
        f"  aggregate output throughput: {left_label} is {output_speedup:.2f}x "
        f"{'faster' if output_speedup >= 1 else 'slower'} than {right_label}"
    )
    print(
        f"  aggregate total throughput: {left_label} is {total_speedup:.2f}x "
        f"{'faster' if total_speedup >= 1 else 'slower'} than {right_label}"
    )
    print(
        f"  normalized output speed: {left_label} needs "
        f"{left_summary['seconds_per_1k_output']:.3f}s per 1k output tokens vs "
        f"{right_summary['seconds_per_1k_output']:.3f}s for {right_label} "
        f"({output_efficiency:.2f}x advantage)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare speed and throughput between two LLM call JSONL logs."
    )
    parser.add_argument("left", type=Path, help="First JSONL log")
    parser.add_argument("right", type=Path, help="Second JSONL log")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    args = parser.parse_args()

    left_rows = load_rows(args.left)
    right_rows = load_rows(args.right)
    left_summary = summarize(left_rows)
    right_summary = summarize(right_rows)

    print_summary(args.left_label, left_summary)
    print()
    print_summary(args.right_label, right_summary)
    print_comparison(args.left_label, left_summary, args.right_label, right_summary)


if __name__ == "__main__":
    main()
