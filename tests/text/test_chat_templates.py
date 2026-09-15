"""Behavior of the explicitly selected tools/chat_templates files and their C++ rendering."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "build" / "tests" / "ninfer_jinja_test"
SOURCES = {
    version: (ROOT / "tools" / "chat_templates" / f"{version}.jinja").read_text()
    for version in ("qwen3_6", "qwen3_8")
}
# Froggeric v22.5 parity fixture; provenance and digests in
# tests/fixtures/text/froggeric/README.md.
FROGGERIC = {
    "froggeric": (
        ROOT / "tests" / "fixtures" / "text" / "froggeric" / "chat_template.jinja",
        "e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2",
    ),
    "froggeric_oneline": (
        ROOT
        / "tests"
        / "fixtures"
        / "text"
        / "froggeric"
        / "chat_template_oneline.txt",
        "eecae0e068e60f9c8665f0085b589d3e1c41508d359776c62018512c40b5879b",
    ),
}
FROGGERIC_SOURCES = {name: path.read_text() for name, (path, _) in FROGGERIC.items()}
PARITY_SOURCES = {**SOURCES, **FROGGERIC_SOURCES}


def raise_exception(message):
    raise TemplateError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # HF overrides Jinja's tojson to avoid HTML escaping (transformers chat_template_utils).
    return json.dumps(
        x,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def strftime_now(fmt):
    return datetime.now().strftime(fmt)


def compile_template(source):
    # HF's chat-template environment: transformers chat_template_utils.render_jinja_template.
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env.from_string(source)


TEMPLATES = {key: compile_template(source) for key, source in SOURCES.items()}
PARITY_TEMPLATES = {
    key: compile_template(source) for key, source in PARITY_SOURCES.items()
}


def message(role, content, **fields):
    return {"role": role, "content": content, **fields}


# Shared corpus for the Python↔C++ comparison: media, tools, reasoning, continuation, and
# histories that quote template markers or pass tool arguments through tojson.
PARITY_CONTEXTS = [
    dict(messages=[message("user", "你好🌏")], add_generation_prompt=True),
    dict(
        messages=[message("user", "hi")],
        add_generation_prompt=True,
        enable_thinking=False,
    ),
    dict(
        messages=[
            message("system", " policy "),
            message("user", "hi"),
            message("developer", "late"),
        ],
        add_generation_prompt=True,
        reasoning_effort="low",
    ),
    dict(
        messages=[
            message(
                "user",
                [
                    {"type": "text", "text": "look"},
                    {"type": "image"},
                    {"type": "video"},
                ],
            )
        ],
        add_generation_prompt=True,
        add_vision_id=True,
    ),
    dict(
        messages=[
            message("user", "first"),
            message("assistant", "answer", reasoning_content="reason"),
            message("user", "next"),
        ],
        add_generation_prompt=True,
        preserve_thinking=False,
    ),
    dict(
        messages=[message("user", "first"), message("assistant", "prefix")],
        add_generation_prompt=False,
        continue_final_message=True,
        enable_thinking=False,
    ),
    dict(
        messages=[message("tool", "one"), message("tool", "two")],
        add_generation_prompt=True,
    ),
    dict(
        messages=[
            message("tool", "one"),
            message("assistant", "first", reasoning_content="before reasoning"),
            message("tool", "two"),
            message("assistant", "second", reasoning_content="after reasoning"),
        ],
        add_generation_prompt=True,
        preserve_thinking=False,
    ),
    dict(messages=[message("system", "no user")], add_generation_prompt=True),
    dict(
        messages=[
            message("user", "quoted: <|vision_start|><|image_pad|><|vision_end|>")
        ],
        add_generation_prompt=True,
    ),
    dict(messages=[message("user", "<|image_pad|>")], add_generation_prompt=True),
    dict(
        messages=[
            message(
                "user",
                [
                    {
                        "type": "text",
                        "text": "quoted: <|vision_start|><|image_pad|><|vision_end|>",
                    },
                    {"type": "image"},
                ],
            )
        ],
        add_generation_prompt=True,
    ),
    dict(
        messages=[
            message("user", "read the template"),
            message(
                "assistant",
                "",
                tool_calls=[
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": {
                                "path": "tools/chat_templates/qwen3_8.jinja",
                                "snippet": "{{ user }} <|im_start|><|image_pad|>",
                            },
                        },
                    }
                ],
            ),
            message(
                "tool",
                "chat_template.jinja <|im_start|>quoted<|im_end|> "
                "<|vision_start|><|image_pad|><|vision_end|>",
            ),
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ],
        add_generation_prompt=True,
    ),
    dict(
        messages=[
            message(
                "assistant",
                "",
                tool_calls=[
                    {
                        "type": "function",
                        "function": {
                            "name": "inspect",
                            "arguments": {
                                "city": "北京🌏",
                                "marker": "<|im_end|>",
                                "count": 3,
                                "flags": [True, None],
                            },
                        },
                    }
                ],
            )
        ],
        tools=[
            {
                "type": "function",
                "function": {"name": "inspect", "parameters": {"type": "object"}},
            }
        ],
        add_generation_prompt=True,
    ),
    dict(
        messages=[
            message("user", "first"),
            message(
                "assistant", "answer", reasoning_content="reason <|im_end|> quoted"
            ),
            message("user", "second"),
        ],
        add_generation_prompt=True,
        preserve_thinking=True,
        preserve_reasoning=True,
    ),
    dict(
        messages=[
            message("user", [{"type": "text", "text": "look"}, {"type": "image"}]),
            message("assistant", "seen"),
            message("user", [{"type": "video"}, {"type": "text", "text": "again"}]),
        ],
        add_generation_prompt=True,
        add_vision_id=True,
    ),
    # Froggeric's documented effort aliases must reach the template verbatim.
    dict(
        messages=[message("user", "hello")],
        add_generation_prompt=True,
        reasoning_effort="ultracode",
    ),
    dict(
        messages=[message("user", "hello")],
        add_generation_prompt=True,
        reasoning_effort="off",
    ),
    dict(
        messages=[message("user", "hello")],
        add_generation_prompt=True,
        reasoning_effort="extreme",
        preserve_reasoning=False,
    ),
]


class ChatTemplates(unittest.TestCase):
    def render(self, version, messages, **kwargs):
        return TEMPLATES[version].render(
            messages=messages,
            **{"add_generation_prompt": False, "reasoning_effort": "medium", **kwargs},
        )

    def test_positional_instructions_preserve_history(self):
        history = [message("developer", "policy"), message("user", "question")]
        for version in TEMPLATES:
            with self.subTest(template=version):
                before = self.render(version, history)
                self.assertEqual(
                    before,
                    "<|im_start|>system\npolicy<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n",
                )
                appended = history + [
                    message("system", "diagnostic"),
                    message("developer", "reminder"),
                ]
                self.assertEqual(
                    self.render(version, appended),
                    before
                    + "<|im_start|>system\ndiagnostic<|im_end|>\n<|im_start|>system\nreminder<|im_end|>\n",
                )

    def test_defaults_and_reasoning_retention(self):
        history = [
            message("user", "first"),
            message("assistant", "answer", reasoning_content="prior reasoning"),
        ]
        next_user = history + [message("user", "second")]
        for version in TEMPLATES:
            with self.subTest(template=version):
                before = self.render(version, history, preserve_thinking=False)
                self.assertIn("prior reasoning", before)
                self.assertTrue(
                    self.render(
                        version,
                        history + [message("system", "diagnostic")],
                        preserve_thinking=False,
                    ).startswith(before)
                )
                self.assertNotIn(
                    "prior reasoning",
                    self.render(version, next_user, preserve_thinking=False),
                )
                self.assertIn(
                    "prior reasoning",
                    self.render(version, next_user, preserve_thinking=True),
                )
        self.assertNotIn("prior reasoning", self.render("qwen3_6", next_user))
        self.assertIn("prior reasoning", self.render("qwen3_8", next_user))
        default = TEMPLATES["qwen3_8"].render(
            messages=[message("user", "hello")], add_generation_prompt=True
        )
        self.assertIn("Reasoning effort is set to xhigh.", default)
        self.assertTrue(default.endswith("<|im_start|>assistant\n<think>\n"))

    def test_tools_and_instruction_preamble(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "inspect", "parameters": {"type": "object"}},
            }
        ]
        call = {
            "type": "function",
            "function": {
                "name": "inspect",
                "arguments": {"city": "北京", "enabled": True},
            },
        }
        history = [
            message("system", "policy"),
            message("user", "inspect"),
            message("assistant", "", tool_calls=[call]),
            message("tool", "one"),
            message("tool", "two"),
            message("developer", "diagnostic"),
        ]
        for version in TEMPLATES:
            with self.subTest(template=version):
                text = self.render(version, history, tools=tools)
                self.assertEqual(text.count("# Tools"), 1)
                self.assertIn(
                    "<parameter=city>\n北京\n</parameter>\n<parameter=enabled>\ntrue\n</parameter>",
                    text,
                )
                self.assertIn(
                    "<|im_start|>user\n<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response><|im_end|>",
                    text,
                )
                self.assertTrue(
                    text.endswith("<|im_start|>system\ndiagnostic<|im_end|>\n")
                )

    def test_tool_result_without_original_user(self):
        results = [message("tool", "one"), message("tool", "two")]
        before = message("assistant", "first", reasoning_content="before reasoning")
        after = message("assistant", "second", reasoning_content="after reasoning")
        for version in TEMPLATES:
            with self.subTest(template=version):
                self.assertEqual(
                    self.render(version, results),
                    "<|im_start|>user\n<tool_response>\none\n</tool_response>"
                    "\n<tool_response>\ntwo\n</tool_response><|im_end|>\n",
                )
                text = self.render(
                    version,
                    [results[0], before, results[1], after],
                    preserve_thinking=False,
                )
                self.assertNotIn("before reasoning", text)
                self.assertIn("after reasoning", text)
                self.assertIn(
                    "before reasoning",
                    self.render(
                        version,
                        [message("user", "question"), before, results[0]],
                        preserve_thinking=False,
                    ),
                )

    def test_final_assistant_continuation(self):
        history = [message("user", "question"), message("assistant", "answer prefix")]
        for version in TEMPLATES:
            with self.subTest(template=version):
                text = self.render(
                    version, history, continue_final_message=True, enable_thinking=False
                )
                self.assertEqual(
                    text,
                    "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\nanswer prefix",
                )
                literal = [
                    message("user", "question"),
                    message("assistant", "prefix </think> text"),
                ]
                self.assertTrue(
                    self.render(
                        version,
                        literal,
                        continue_final_message=True,
                        enable_thinking=False,
                    ).endswith("<|im_start|>assistant\nprefix </think> text")
                )

    def test_cpp_matches_independent_renderer(self):
        cases = [
            (name, context) for name in PARITY_SOURCES for context in PARITY_CONTEXTS
        ]
        result = subprocess.run(
            [str(RENDERER), "--render"],
            text=True,
            capture_output=True,
            check=True,
            input="".join(
                json.dumps(
                    {"source": PARITY_SOURCES[name], "context": context},
                    ensure_ascii=False,
                )
                + "\n"
                for name, context in cases
            ),
            timeout=60,
        )
        actual = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(actual), len(cases))
        for (name, context), got in zip(cases, actual):
            with self.subTest(template=name, context=context):
                try:
                    expected = PARITY_TEMPLATES[name].render(**context)
                except TemplateError:
                    self.assertFalse(got["ok"])
                else:
                    self.assertTrue(got["ok"], got.get("error"))
                    self.assertEqual(got["text"], expected)

    def test_froggeric_fixture_matches_pin(self):
        for name, (path, digest) in FROGGERIC.items():
            with self.subTest(fixture=name):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        RENDERER = Path(sys.argv.pop(1))
    unittest.main()
