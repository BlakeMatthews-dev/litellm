# What is this?
## Tests the hide_secrets enterprise guardrail. Specifically pins the
## Base64HighEntropyString limit in the default config to 4.0 and exercises
## the paired false-positive regression: each string that was flagged at the
## old limit (3.0) must be clean at the new limit (4.0), while real high-
## entropy secrets must still be caught.

import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import hash_token
from litellm_enterprise.enterprise_callbacks.secret_detection import (
    _ENTERPRISE_SecretDetection,
    _default_detect_secrets_config,
)


@pytest.fixture(autouse=True)
def _reset_detect_secrets_plugin_cache():
    """``detect_secrets`` lru-caches the secret-type → class mapping at
    module level. Earlier tests in this file scan with a minimal plugin
    config; if that mapping is cached first, later tests using the full
    default config can't resolve custom plugin class names (``AdafruitKeyDetector``,
    etc.). Clearing the cache before each test pins isolation."""
    from detect_secrets.core.plugins.util import (
        get_mapping_from_secret_type_to_class,
    )

    get_mapping_from_secret_type_to_class.cache_clear()
    yield
    get_mapping_from_secret_type_to_class.cache_clear()


# Strings drawn from realistic LLM proxy traffic that the OLD 3.0 limit
# wrongly flagged as base64 secrets. Each is presented in the quoted form
# the guardrail actually sees in a JSON-serialized chat message.
_FALSE_POSITIVE_CASES = [
    ('"model": "gpt-4o-mini"', "gpt-4o-mini"),
    ('"model": "claude-3-5-sonnet-20241022"', "claude-3-5-sonnet-20241022"),
    ('"alg": "eyJhbGciOiJSUzI1NiJ9"', "eyJhbGciOiJSUzI1NiJ9"),
    ('"encoded": "cGFzc3dvcmQ="', "cGFzc3dvcmQ="),
    ('"value": "bG9jYWxob3N0"', "bG9jYWxob3N0"),
    ('"env": "DATABASE_URL"', "DATABASE_URL"),
    ('"cls": "HttpRequestHandler"', "HttpRequestHandler"),
    ('"hash": "sha256abcd1234"', "sha256abcd1234"),
]

# Real high-entropy strings that MUST still be caught at the new limit.
_TRUE_POSITIVE_CASES = [
    '"key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"',
    '"key": "AKIAIOSFODNN7EXAMPLE0xv8GhB1zHbT9Yk2RLp4MQ"',
    '"token": "ZmFrZS1hcGkta2V5LXdpdGgtaGlnaC1lbnRyb3B5LWFiY2RlZjEyMzQ1Njc4OTA="',
    '"key": "Tk9SR0FOSVpBVElPTl9LRVlfaGlnaF9lbnRyb3B5XzEyMzQ1Ng=="',
]


def _scan_with_limit(content: str, limit: float):
    """Scan content using only the Base64HighEntropyString plugin at ``limit``.

    Isolating the plugin makes the test independent of other detectors in
    the default config and pins behavior to the entropy threshold itself.
    """
    obj = _ENTERPRISE_SecretDetection(
        detect_secrets_config={
            "plugins_used": [{"name": "Base64HighEntropyString", "limit": limit}]
        }
    )
    return obj.scan_message_for_secrets(content)


@pytest.mark.parametrize("content,expected_value", _FALSE_POSITIVE_CASES)
def test_fp_strings_were_flagged_at_old_limit(content, expected_value):
    """At limit=3.0 (the previous default), each FP string is detected."""
    detected = _scan_with_limit(content, 3.0)
    values = [s["value"] for s in detected]
    assert expected_value in values, (
        f"Expected old limit=3.0 to flag {expected_value!r} (this proves it was "
        f"a real regression). Got {values!r}."
    )


@pytest.mark.parametrize("content,expected_value", _FALSE_POSITIVE_CASES)
def test_fp_strings_are_clean_at_new_limit(content, expected_value):
    """At limit=4.0 (the new default), the same FP strings are NOT detected."""
    detected = _scan_with_limit(content, 4.0)
    values = [s["value"] for s in detected]
    assert expected_value not in values, (
        f"New limit=4.0 still flags {expected_value!r}, which is the regression "
        f"this change is meant to fix. Full detected: {values!r}."
    )


@pytest.mark.parametrize("content", _TRUE_POSITIVE_CASES)
def test_real_secrets_still_caught_at_new_limit(content):
    detected = _scan_with_limit(content, 4.0)
    assert detected, (
        f"Real high-entropy secret was NOT caught at limit=4.0: {content!r}. "
        f"Raising the limit must not silently drop genuine secrets."
    )


def test_default_config_base64_limit_is_4_0():
    """Pin the default config so a future bump back to 3.0 fails CI."""
    b64_plugins = [
        p
        for p in _default_detect_secrets_config["plugins_used"]
        if p.get("name") == "Base64HighEntropyString"
    ]
    assert len(b64_plugins) == 1, (
        f"Expected exactly one Base64HighEntropyString entry in default config, "
        f"found {len(b64_plugins)}."
    )
    assert b64_plugins[0]["limit"] == 4.0, (
        f"Default Base64HighEntropyString limit changed to {b64_plugins[0]['limit']}; "
        f"see PR raising it to 4.0 to reduce false positives on LLM traffic."
    )


def test_user_config_overrides_default():
    """A user-supplied detect_secrets_config fully replaces the default."""
    custom = {"plugins_used": [{"name": "Base64HighEntropyString", "limit": 5.0}]}
    obj = _ENTERPRISE_SecretDetection(detect_secrets_config=custom)
    assert obj.user_defined_detect_secrets_config == custom

    # At limit=5.0, the AWS demo key (entropy ~4.7) should NOT be caught
    # whereas at the default 4.0 it IS caught. This proves the user config
    # actually flows through to scan_message_for_secrets.
    aws_demo = '"key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'
    custom_detected = obj.scan_message_for_secrets(aws_demo)
    assert not any(
        s["value"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        for s in custom_detected
    ), "User-supplied limit=5.0 should have suppressed the AWS demo key"


@pytest.mark.asyncio
async def test_hide_secrets_redacts_high_entropy_token():
    """End-to-end: a real high-entropy token in a chat message is mutated to [REDACTED]."""
    obj = _ENTERPRISE_SecretDetection()
    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-12345"))
    cache = DualCache()
    token = "ZmFrZS1hcGkta2V5LXdpdGgtaGlnaC1lbnRyb3B5LWFiY2RlZjEyMzQ1Njc4OTA="
    data = {"messages": [{"role": "user", "content": f'here is my api key: "{token}"'}]}
    await obj.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion",
    )
    redacted_content = data["messages"][0]["content"]
    assert (
        token not in redacted_content
    ), f"Token was not redacted from message content: {redacted_content!r}"
    assert "[REDACTED]" in redacted_content


@pytest.mark.asyncio
async def test_hide_secrets_does_not_redact_model_name():
    """End-to-end: a message discussing model names is not mutated."""
    obj = _ENTERPRISE_SecretDetection()
    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-12345"))
    cache = DualCache()
    original = 'Please route this to "gpt-4o-mini" with temperature 0.7'
    data = {"messages": [{"role": "user", "content": original}]}
    await obj.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion",
    )
    assert (
        data["messages"][0]["content"] == original
    ), f"Model name message was mutated: {data['messages'][0]['content']!r}"


@pytest.mark.asyncio
async def test_hide_secrets_does_not_redact_jwt_alg_header():
    """End-to-end: a JWT algorithm header (low-entropy structural prefix) is preserved."""
    obj = _ENTERPRISE_SecretDetection()
    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("sk-12345"))
    cache = DualCache()
    original = 'The JWT alg header decodes from "eyJhbGciOiJSUzI1NiJ9" to RS256'
    data = {"messages": [{"role": "user", "content": original}]}
    await obj.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion",
    )
    assert (
        data["messages"][0]["content"] == original
    ), f"JWT alg header message was mutated: {data['messages'][0]['content']!r}"
