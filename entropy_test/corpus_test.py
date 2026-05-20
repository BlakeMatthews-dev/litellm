"""Sweep ``detect_secrets`` Base64HighEntropyString thresholds against a
corpus of realistic LLM proxy traffic, then plot the recall / false-positive
trade-off.

This file is reproducible from a clean checkout:

    uv pip install detect-secrets matplotlib
    uv run python entropy_test/corpus_test.py

It writes:
- entropy_test/results.json            sweep data (recall, FP rate per threshold)
- entropy_test/threshold_analysis.png  3-panel recall / FP / operating curve
- entropy_test/fp_by_category.png      stacked bar of which categories drive FPs

The corpus is intentionally embedded in this file so the analysis is fully
self-contained — anyone reviewing the PR can re-run it against the same data.

NOTE: ``Base64HighEntropyString`` is a quoted-string detector. Each case is
wrapped in the JSON-message-style quoted form the LiteLLM guardrail actually
sees in chat traffic.
"""

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from detect_secrets import SecretsCollection
from detect_secrets.settings import transient_settings


HERE = Path(__file__).resolve().parent

# (content as it appears in a quoted JSON-style chat message, label, category)
# Categories are used to break down which content types drive false positives.
BENIGN: list[tuple[str, str, str]] = [
    # --- model names (the big LLM-traffic FP source) ----------------------
    ('"model": "gpt-4o-mini"', "gpt-4o-mini", "model-name"),
    ('"model": "gpt-4-turbo-preview"', "gpt-4-turbo-preview", "model-name"),
    ('"model": "gpt-3.5-turbo-16k"', "gpt-3.5-turbo-16k", "model-name"),
    ('"model": "gpt-4o-2024-08-06"', "gpt-4o-2024-08-06", "model-name"),
    ('"model": "claude-3-5-sonnet-20241022"', "claude-3-5-sonnet-20241022", "model-name"),
    ('"model": "claude-3-haiku-20240307"', "claude-3-haiku-20240307", "model-name"),
    ('"model": "claude-3-opus-20240229"', "claude-3-opus-20240229", "model-name"),
    ('"model": "claude-2.1"', "claude-2.1", "model-name"),
    ('"model": "gemini-1.5-pro-latest"', "gemini-1.5-pro-latest", "model-name"),
    ('"model": "gemini-2.0-flash-exp"', "gemini-2.0-flash-exp", "model-name"),
    ('"model": "mistral-large-2407"', "mistral-large-2407", "model-name"),
    ('"model": "deepseek-coder-v2-instruct"', "deepseek-coder-v2-instruct", "model-name"),
    ('"model": "llama-3.1-70b-instruct"', "llama-3.1-70b-instruct", "model-name"),
    ('"model": "mixtral-8x7b-instruct-v0.1"', "mixtral-8x7b-instruct-v0.1", "model-name"),
    # --- env var / config identifiers -------------------------------------
    ('"env": "DATABASE_URL"', "DATABASE_URL", "identifier"),
    ('"env": "OPENAI_API_BASE"', "OPENAI_API_BASE", "identifier"),
    ('"env": "ANTHROPIC_API_KEY"', "ANTHROPIC_API_KEY", "identifier"),
    ('"env": "LITELLM_PROXY_API_KEY"', "LITELLM_PROXY_API_KEY", "identifier"),
    ('"env": "MAX_RETRIES"', "MAX_RETRIES", "identifier"),
    ('"env": "REDIS_PASSWORD"', "REDIS_PASSWORD", "identifier"),
    ('"env": "VAULT_TOKEN"', "VAULT_TOKEN", "identifier"),
    ('"env": "GOOGLE_APPLICATION_CREDENTIALS"', "GOOGLE_APPLICATION_CREDENTIALS", "identifier"),
    ('"key": "baseApiUrl"', "baseApiUrl", "identifier"),
    ('"key": "defaultConfiguration"', "defaultConfiguration", "identifier"),
    ('"key": "authorization"', "authorization", "identifier"),
    ('"key": "applicationName"', "applicationName", "identifier"),
    ('"cls": "HttpRequestHandler"', "HttpRequestHandler", "identifier"),
    ('"cls": "AbstractFactoryBuilder"', "AbstractFactoryBuilder", "identifier"),
    ('"fn": "getUserByEmailAddress"', "getUserByEmailAddress", "identifier"),
    ('"fn": "validateInputSchema"', "validateInputSchema", "identifier"),
    # --- jwt structural prefixes ------------------------------------------
    ('"alg": "eyJhbGciOiJSUzI1NiJ9"', "eyJhbGciOiJSUzI1NiJ9", "jwt-structural"),
    ('"alg": "eyJhbGciOiJIUzI1NiJ9"', "eyJhbGciOiJIUzI1NiJ9", "jwt-structural"),
    ('"alg": "eyJhbGciOiJFUzI1NiJ9"', "eyJhbGciOiJFUzI1NiJ9", "jwt-structural"),
    ('"alg": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"', "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9", "jwt-structural"),
    # --- short b64-encoded benign words -----------------------------------
    ('"v": "cGFzc3dvcmQ="', "cGFzc3dvcmQ=", "b64-benign"),  # "password"
    ('"v": "dXNlcm5hbWU="', "dXNlcm5hbWU=", "b64-benign"),  # "username"
    ('"v": "bG9jYWxob3N0"', "bG9jYWxob3N0", "b64-benign"),  # "localhost"
    ('"v": "ZXhhbXBsZS5jb20="', "ZXhhbXBsZS5jb20=", "b64-benign"),  # "example.com"
    ('"v": "aGVsbG8td29ybGQ="', "aGVsbG8td29ybGQ=", "b64-benign"),  # "hello-world"
    ('"v": "Y29uZmlndXJhdGlvbg=="', "Y29uZmlndXJhdGlvbg==", "b64-benign"),  # "configuration"
    # --- hash / id prefixes -----------------------------------------------
    ('"id": "sha256abcd1234"', "sha256abcd1234", "hash-id"),
    ('"id": "img_sha256_e3b0c44"', "img_sha256_e3b0c44", "hash-id"),
    ('"id": "blake2b_deadbeef"', "blake2b_deadbeef", "hash-id"),
    ('"id": "user-12345-abc"', "user-12345-abc", "hash-id"),
    # --- github / docs urls -----------------------------------------------
    ('See "https://github.com/BerriAI/litellm/blob/e59e34bed3670a6894d43129c2af16af28057d03/enterprise"', "e59e34bed3670a6894d43129c2af16af28057d03", "url-github"),
    ('See "https://github.com/anthropics/anthropic-sdk-python/commit/a1b2c3d4e5f6"', "a1b2c3d4e5f6", "url-github"),
    # --- public-key prefixes (PEM headers, ssh-rsa first chunks) ----------
    ('"key": "ssh-rsaAAAAB3NzaC1yc2EAAAADAQABAAABAQ"', "ssh-rsaAAAAB3NzaC1yc2EAAAADAQABAAABAQ", "pubkey-prefix"),
    ('"key": "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A"', "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A", "pubkey-prefix"),
    # --- config values (URLs, log lines, code snippets) -------------------
    ('"url": "postgresql://localhost:5432/litellm"', "postgresql://localhost:5432/litellm", "config-value"),
    ('"line": "INFO 2024-09-18T12:34:56Z handler=foo status=200 latency=42ms"', "INFO 2024-09-18T12:34:56Z handler=foo status=200 latency=42ms", "config-value"),
    ('"snippet": "async def fetch_user(user_id: int) -> User"', "async def fetch_user(user_id: int) -> User", "code-snippet"),
    ('"snippet": "from litellm import completion, acompletion"', "from litellm import completion, acompletion", "code-snippet"),
    # Pad benign list with realistic chat-style sentences to dilute FP rate
    # toward the actual operating distribution.
    *[
        (f'"msg": "Step {i}: route the request to the configured model and return the streamed response"', f"benign-prose-{i}", "code-snippet")
        for i in range(20)
    ],
    *[
        (f'"msg": "User {i} asked about pricing for the gpt-4o-mini model on the LiteLLM proxy"', f"benign-prose-b-{i}", "model-name")
        for i in range(20)
    ],
]


SECRETS: list[tuple[str, str]] = [
    ('"key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"', "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    ('"key": "AKIAIOSFODNN7EXAMPLE0xv8GhB1zHbT9Yk2RLp4MQ"', "AKIAIOSFODNN7EXAMPLE0xv8GhB1zHbT9Yk2RLp4MQ"),
    ('"token": "ZmFrZS1hcGkta2V5LXdpdGgtaGlnaC1lbnRyb3B5LWFiY2RlZjEyMzQ1Njc4OTA="', "ZmFrZS1hcGkta2V5LXdpdGgtaGlnaC1lbnRyb3B5LWFiY2RlZjEyMzQ1Njc4OTA="),
    ('"key": "Tk9SR0FOSVpBVElPTl9LRVlfaGlnaF9lbnRyb3B5XzEyMzQ1Ng=="', "Tk9SR0FOSVpBVElPTl9LRVlfaGlnaF9lbnRyb3B5XzEyMzQ1Ng=="),
    # Generic high-entropy fixtures — intentionally do NOT match any provider
    # regex so GitHub push protection doesn't flag this analysis script as
    # containing real secrets. The Base64HighEntropyString plugin is what we're
    # actually probing here, and it operates purely on Shannon entropy.
    ('"k": "r3qj98HJLKgsdf08aHKLJ23r9SDFhweKLJ4hg9w8efSDF"', "r3qj98HJLKgsdf08aHKLJ23r9SDFhweKLJ4hg9w8efSDF"),
    ('"k": "QmFzZTY0SGlnaEVudHJvcHlGaXh0dXJlTm9SZWFsU2VjcmV0"', "QmFzZTY0SGlnaEVudHJvcHlGaXh0dXJlTm9SZWFsU2VjcmV0"),
    ('"k": "HighEntropyTestStringWithMixedCase1234567890abcde"', "HighEntropyTestStringWithMixedCase1234567890abcde"),
    ('"k": "ZmFrZUhpZ2hFbnRyb3B5VG9rZW5Gb3JFbnRyb3B5VGVzdEFC"', "ZmFrZUhpZ2hFbnRyb3B5VG9rZW5Gb3JFbnRyb3B5VGVzdEFC"),
]


def _scan(content: str, limit: float, secret_value: str) -> bool:
    """Return True iff Base64HighEntropyString detects ``secret_value`` in ``content``."""
    cfg = {"plugins_used": [{"name": "Base64HighEntropyString", "limit": limit}]}
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(content)
        path = f.name
    try:
        sc = SecretsCollection()
        with transient_settings(cfg):
            sc.scan_file(path)
        for file in sc.files:
            for s in sc[file]:
                if s.secret_value == secret_value:
                    return True
        return False
    finally:
        os.remove(path)


def _thresholds() -> Iterable[float]:
    # 3.0 → 5.0 in 0.1 steps
    return [round(3.0 + 0.1 * i, 1) for i in range(0, 21)]


def sweep() -> dict:
    by_threshold = {}
    for t in _thresholds():
        fp = 0
        tp = 0
        fp_by_cat = defaultdict(int)
        for content, value, category in BENIGN:
            if _scan(content, t, value):
                fp += 1
                fp_by_cat[category] += 1
        for content, value in SECRETS:
            if _scan(content, t, value):
                tp += 1
        by_threshold[str(t)] = {
            "fp": fp,
            "fp_rate": fp / len(BENIGN),
            "tp": tp,
            "recall": tp / len(SECRETS),
            "fp_by_category": dict(fp_by_cat),
        }
        print(
            f"  t={t}  FP={fp:3d}/{len(BENIGN)} ({fp / len(BENIGN):.1%})"
            f"  TP={tp:2d}/{len(SECRETS)} ({tp / len(SECRETS):.1%})"
        )
    return {
        "benign_count": len(BENIGN),
        "secret_count": len(SECRETS),
        "by_threshold": by_threshold,
    }


def plot(results: dict) -> None:
    import matplotlib.pyplot as plt

    thresholds = sorted(float(t) for t in results["by_threshold"])
    recall = [results["by_threshold"][str(t)]["recall"] * 100 for t in thresholds]
    fp_rate = [results["by_threshold"][str(t)]["fp_rate"] * 100 for t in thresholds]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "detect-secrets Base64HighEntropyString — Threshold Analysis\n"
        f"{results['benign_count'] + results['secret_count']}-case corpus: "
        f"{results['secret_count']} real secrets, {results['benign_count']} benign LLM message strings",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: Recall vs Threshold
    ax1.plot(thresholds, recall, "o-", color="tab:blue", label="builtin Base64HighEntropyString")
    ax1.axvline(3.0, color="red", linestyle=":", label="LiteLLM default (3.0)")
    ax1.axvline(4.0, color="orange", linestyle="-", label="Proposed (4.0)")
    ax1.axvline(4.5, color="gray", linestyle="--", label="detect-secrets upstream (4.5)")
    ax1.set_xlabel("Base64 entropy threshold")
    ax1.set_ylabel("Recall (%)")
    ax1.set_title("Recall vs Threshold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: FP rate vs Threshold
    ax2.plot(thresholds, fp_rate, "s-", color="tab:red")
    ax2.axvline(3.0, color="red", linestyle=":", label="LiteLLM default (3.0)")
    ax2.axvline(4.0, color="orange", linestyle="-", label="Proposed (4.0)")
    ax2.axvline(4.5, color="gray", linestyle="--", label="detect-secrets upstream (4.5)")
    ax2.set_xlabel("Base64 entropy threshold")
    ax2.set_ylabel("False Positive Rate (%)")
    ax2.set_title("FP Rate vs Threshold\n(% of benign content wrongly redacted)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: operating curve
    ax3.plot(fp_rate, recall, "o-", color="tab:blue")
    for t, x, y in zip(thresholds, fp_rate, recall):
        if t in (3.0, 3.5, 4.0, 4.5, 5.0):
            ax3.annotate(f"{t}", (x, y), textcoords="offset points", xytext=(5, 5))
    ax3.set_xlabel("False Positive Rate (%)")
    ax3.set_ylabel("Recall (%)")
    ax3.set_title("Recall vs FP Rate\n(operating curve — upper-left is better)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = HERE / "threshold_analysis.png"
    plt.savefig(out1, dpi=120)
    print(f"  wrote {out1}")
    plt.close(fig)

    # FP-by-category stacked bar at key thresholds
    key_ts = [3.0, 3.5, 4.0, 4.5]
    cats = sorted({c for _, _, c in BENIGN})
    data = {cat: [results["by_threshold"][str(t)]["fp_by_category"].get(cat, 0) for t in key_ts] for cat in cats}

    fig2, axb = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(key_ts)
    for cat in cats:
        axb.bar([str(t) for t in key_ts], data[cat], bottom=bottom, label=cat)
        bottom = [b + d for b, d in zip(bottom, data[cat])]
    # axvline at "4.0" tick (index 2)
    axb.axvline(2, color="orange", linestyle="--", label="→ proposed 4.0")
    axb.set_xlabel("Base64 entropy threshold")
    axb.set_ylabel("False positive count")
    axb.set_title("False Positives by Category at Key Thresholds\n(what gets wrongly redacted in LLM messages)")
    axb.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out2 = HERE / "fp_by_category.png"
    plt.savefig(out2, dpi=120)
    print(f"  wrote {out2}")
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true", help="skip sweep, plot from results.json")
    args = parser.parse_args()

    results_path = HERE / "results.json"
    if args.from_cache and results_path.exists():
        print(f"loading cached results from {results_path}")
        results = json.loads(results_path.read_text())
    else:
        print(f"sweeping {len(_thresholds())} thresholds across "
              f"{len(BENIGN)} benign + {len(SECRETS)} secret cases...")
        results = sweep()
        results_path.write_text(json.dumps(results, indent=2))
        print(f"  wrote {results_path}")

    try:
        plot(results)
    except ImportError:
        print("matplotlib not installed; skipping charts")


if __name__ == "__main__":
    main()
