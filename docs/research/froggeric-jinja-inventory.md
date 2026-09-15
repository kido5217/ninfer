# Froggeric v22.5 Jinja inventory vs NInfer's Jinja engine

Ticket: kido5217/ninfer#7 ("Research: froggeric v22.5 feature inventory vs NInfer's Jinja engine").
Question: which Jinja constructs does froggeric's Qwen-Fixed-Chat-Templates v22.5 use, and which
does NInfer's engine (`third_party/llama-jinja`, forked from llama.cpp `common/jinja`) support,
diverge on, or crash on?

Everything below is either cited to a primary source (file/line, upstream URL) or backed by a
recorded probe command and its observed result. Probe artifacts live under `/tmp/opencode/`
(ephemeral); commands are reproduced inline.

## 1. Pinned sources and hashes

| Artifact | Value |
|---|---|
| froggeric repo | `https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates` |
| froggeric commit (HEAD at clone) | `855bffc49448e299789730ff92c9b8d834d6cc14` (2026-09-04 14:38 +0200) |
| `chat_template.jinja` sha256 | `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2` (28,234 bytes) |
| `chat_template_oneline.txt` sha256 | `eecae0e068e60f9c8665f0085b589d3e1c41508d359776c62018512c40b5879b` (22,391 bytes) |
| template self-identification | `{%- set template_version = "qwen3.8-froggeric-v22.5" %}` (line 1) |

Recorded commands:

```bash
git clone https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates /tmp/opencode/froggeric
git -C /tmp/opencode/froggeric rev-parse HEAD   # 855bffc49448e299789730ff92c9b8d834d6cc14
sha256sum /tmp/opencode/froggeric/chat_template.jinja /tmp/opencode/froggeric/chat_template_oneline.txt
```

**Do the two builds differ beyond whitespace?** No. A lexer that strips whitespace only *outside*
string literals (so whitespace inside rendered strings is preserved) yields identical 19,924-byte
programs for both files. This is stricter than `\s+`-stripping, which would hide string-content
differences:

```
equal ignoring whitespace outside strings: True 19924 19924
```

Froggeric's own suite asserts the same property functionally: `scripts/test_v22.py:123-157`
(cell 94, "chat_template_oneline.txt renders identically to chat_template.jinja").

**NInfer engine provenance.** `third_party/llama-jinja/README.ninfer.md:3-7` pins the source base
to llama.cpp commit `7609846557c50f9d984719a9e1e8c5f3d02f807b`, `common/jinja/`, and states NInfer
maintains the fork, adopting upstream fixes selectively. Upstream still ships `common/jinja` at that
commit (GitHub contents API: 14 entries, including `caps.cpp`/`caps.h`, which NInfer's fork does not
have; NInfer adds local `json.cpp`, `unicode.cpp`, `unicode.h`, `unicode_data.h`). Every shared file
differs from upstream at the pinned base (sha256 comparison of the 11 shared files: all `DIFFERS`),
consistent with the maintained-fork statement.

## 2. How support was probed

**Renderer build (bounded attempt 1: CMake).** The prescribed command failed on this Nix host because
no CUDA toolkit is discoverable:

```bash
cmake -S /home/kido/network/projects/ninfer -B .../build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
# CMake Error at .../CMakeCUDAFindToolkit.cmake:86: Failed to find nvcc.
# (nvcc is not on PATH; the store has cuda13.1-cuda_nvcc-13.1.115 but no merged toolkit root.)
```

**Renderer build (attempt 2: standalone, succeeded).** `tests/cmake/CoreTests.cmake:46-48` shows
`ninfer_jinja_test` is `tests/text/test_jinja.cpp` + `ninfer_jinja` (7 host-only C++ sources) +
`ninfer::json`. Building exactly those sources with g++ 15.2.0 reproduces the target without CUDA:

```bash
gcc -std=c11 -O1 -I third_party/utf8proc -c third_party/utf8proc/utf8proc.c -o utf8proc.o
g++ -std=c++20 -O1 -I src -I third_party/llama-jinja -I third_party -I third_party/utf8proc \
    third_party/llama-jinja/jinja/{json,lexer,parser,runtime,string,value,unicode}.cpp \
    src/text/jinja.cpp src/text/unicode.cpp tests/text/test_jinja.cpp utf8proc.o \
    -o /tmp/opencode/ninfer_jinja_test -pthread
/tmp/opencode/ninfer_jinja_test        # built-in language semantics: exit 0
```

Render protocol (from `tests/text/test_jinja.cpp:140-155`): JSON-lines on stdin,
`{"source": <template>, "context": {...}}` -> `{"ok": true, "text": ...}`.

**Python reference.** Python 3.13.15 with Jinja2 3.1.6 (venv `/tmp/opencode/jenv`), configured as the
repo's independent renderer is (`tests/text/test_chat_templates.py:25-34`): sandboxed environment,
`trim_blocks=True`, `lstrip_blocks=True`, `loopcontrols`, and `tojson` replaced with
`json.dumps(..., ensure_ascii=False)` (the transformers convention). Stock Jinja2's own `tojson`
behaves differently (see G3).

**Cross-check of NInfer's own templates.** `tests/text/test_chat_templates.py` run against the
standalone binary: **6/6 OK** (qwen3_6/qwen3_8 Python-vs-C++ parity across tools, reasoning,
continuation, and tool-result histories).

**Probe corpora and totals**

| Corpus | Command | Result |
|---|---|---|
| 90 minimal construct probes (froggeric + engine constructs) | `/tmp/opencode/jenv/bin/python /tmp/opencode/probe_constructs.py` | 86 exact agreements; 1 text diff; 1 C++-lenient-vs-Python-error; 2 both-error (`raise_exception`) |
| 70 full froggeric scenarios | `/tmp/opencode/jenv/bin/python /tmp/opencode/probe_template.py` | 66 exact; 0 text diffs; 4 both-error (the `raise_exception` guards) |
| 11 qwen3_6/3_8-specific constructs (`|items`, `|safe`, `loop.previtem/nextitem`, `|default`, `not in`, `~`) | inline probe | 11/11 exact |
| 21 engine-variable edge cases (hostile types) | `/tmp/opencode/jenv/bin/python /tmp/opencode/probe_hostile.py` | 20 agree, 1 divergence (G8) |
| froggeric's own `scripts/test_v22.py` (105 cells) routed to the C++ engine | `/tmp/opencode/jenv/bin/python /tmp/opencode/run_froggeric_cpp.py` | **104/105 pass**; the one mismatch is the stock-Jinja2 tojson convention (G3), confirmed by four-way re-compare under the transformers convention (all byte-identical) |

## 3. Construct inventory

Scripted census of `{% %}`/`{{ }}` content with string literals stripped
(`/tmp/opencode/probe_census.py`), cross-checked by reading every line of the three templates.
(The census is conservative: it may list a construct twice in different spellings; the smallest
listing that matches the source is given.)

| Dimension | froggeric v22.5 | `tools/chat_templates/qwen3_6.jinja` | `tools/chat_templates/qwen3_8.jinja` |
|---|---|---|---|
| Tags | `set`, `if/elif/else`, `for`, `macro/endmacro`, `namespace`, `raise_exception` (called, not a tag) | same | same |
| Filters | `join`, `length`, `lower`, `string`, `tojson`, `trim`; (`items` as method `arguments.items()`, lines 378) | `default`, `items` (filter form, line 132), `length`, `safe` (line 134), `string`, `tojson`, `trim` | `default` (line 53), `items` (line 148), `safe` (line 150), `string`, `tojson`, `trim` |
| Tests | `defined`, `undefined`, `none`, `string`, `iterable`, `mapping` (with `not` forms: `is not mapping`, `is not none`) | same + `true`/`false` literals | same + `true`/`false`, plus `not in` |
| Methods | `startswith`, `endswith`, `split`, `lstrip`, `rstrip`, `items` | `startswith`, `endswith`, `split`, `lstrip`, `rstrip` | `startswith`, `endswith` |
| Globals | `namespace(...)`, `raise_exception(...)` | same | same |
| Slicing/indexing | `[::-1]` (line 224), `[:expr]` (lines 147-156, 414, 429, 431), `[-1]` (lines 296, 326, 328), `[expr + expr]` (line 443) | `[::-1]`, `[:expr]`, `[-1]`, `|length` | `[::-1]`, `[:expr]`, `|length` |
| Operators | `~`, `+`, `==`, `in`, `not in`, tuple membership `in ('a','b')`, ternary `x if c else y`, `and/or/not` | same | same + explicit `not in` |
| Engine variables consumed | `preserve_reasoning`, `preserve_thinking`, `reasoning_effort`, `enable_thinking`, `tool_call_format`, `max_tool_arg_chars`, `max_tool_response_chars`, `auto_disable_thinking_with_tools`, `add_vision_id`, inline `<|think_*|>` tags | `enable_thinking`, `preserve_thinking`, `reasoning_effort`, `add_vision_id`, `continue_final_message` | same as qwen3_6 |

Froggeric-specific forms not present in NInfer's own templates: namespace mutation inside nested
scopes (`{% set ns.attr = ... %}` in `for`+`if`, lines 35-80, 135-164, 241-452), the reversed
history scan with index reconstruction (`_msgs[::-1]` + `(_msgs|length - 1) - loop.index0`,
lines 224-233), `content[:1] in ('{','[')` (line 429), `~ (_av|length|string) ~` in truncation
messages (lines 386, 400, 431), and inline tag state carried across messages (lines 36-80).

**All of the above render successfully in NInfer's engine.** The full froggeric template renders
from a plain `messages=[{role:user, content:hi}]` context:

```bash
printf '%s\n' '{"source": "<chat_template.jinja contents>", "context": {"messages":[{"role":"user","content":"hi"}],"add_generation_prompt":true}}' \
  | /tmp/opencode/ninfer_jinja_test --render
# {"ok":true,"text":"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"}
```

and 104 of froggeric's own 105 test cells pass when routed to the C++ renderer (the exception is
G3, a reference-convention difference, not a rendering failure).

## 4. Gap list

### G1 — Request path rejects three `reasoning_effort` aliases that froggeric v22.5 accepts (functional gap)

Froggeric maps `none|off` to thinking-off, `minimal|low` to low, `high|xhigh|max|ultracode|extreme`
to xhigh (template lines 21-30); its README documents `ultracode`/`extreme` (README.md:344).
NInfer validates the string against exactly seven values before the template runs
(`src/serve/request.h:141-150`, used by `src/serve/translate.cpp:143-147` for
`chat_template_kwargs.reasoning_effort`, and by the OpenAI/Anthropic/Responses request parsers).
`off`, `ultracode`, and `extreme` therefore return a 400/invalid-request instead of rendering.

Probe (template-level, both accepted and rendered correctly):
`{"reasoning_effort":"off"|"ultracode"|"extreme"}` in the 70-scenario corpus agreed with Python
exactly (system instruction and thinking mode as documented). Request-path rejection is static
evidence (the validation code above); the full HTTP stack was not run.

### G2 — Engine-populated context does not include the froggeric feature variables (configuration gap, not a failure)

`src/models/qwen3_5/frontend/chat_template.cpp:50-98` builds the template context from
`chat_template_kwargs` plus four typed options (`enable_thinking`, `preserve_thinking`,
`reasoning_effort`, `add_vision_id`), then adds `continue_final_message`, `messages`,
`add_generation_prompt` (lines 144-229), `tools`, and special tokens (line 97). It never sets:

| Variable | Froggeric expectation | Default when undefined | How it can be set on NInfer today |
|---|---|---|---|
| `preserve_reasoning` | alias preferred over `preserve_thinking` (lines 8-14; README.md:118,182-194) | falls back to `preserve_thinking`, then `true` | only via `chat_template_kwargs` (top-level `preserve_reasoning` is not parsed; `resolve_prompt_semantics` handles only `preserve_thinking`, `translate.cpp:126-160`) |
| `tool_call_format` | `'xml'` default, `'json'` Hermes override (line 2; README.md:177,211) | `'xml'` | only via `chat_template_kwargs` (pass-through: `openai_chat_request.cpp:811-818`, `translate.cpp:156`) |
| `max_tool_arg_chars` | truncation limit, `0`=off (line 15; README.md:222) | `0` | only via `chat_template_kwargs` |
| `max_tool_response_chars` | truncation limit, `0`=off (line 16; README.md:223) | `0` | only via `chat_template_kwargs` |
| `auto_disable_thinking_with_tools` | disable thinking when tools present (line 7) | `false` | only via `chat_template_kwargs` |
| `add_vision_id` | `"Picture N:"` prefixes (line 5) | `false` | `translate.cpp:292` forces `false`; only via `chat_template_kwargs` |

Inline `<|think_*|>` tags are template-handled (froggeric strips them and carries sticky state,
lines 36-80, 245-256); NInfer passes user content through unchanged, which is what the template
expects. Consequence of the missing defaults: everything still renders (probes agree with Python),
but the documented truncation/json/auto-disable features are off unless the client sends raw
`chat_template_kwargs`. Note also that switching the served template to froggeric changes default
reasoning retention: froggeric defaults `_preserve_thinking = true` (line 13), while NInfer's
qwen3_6 strips past reasoning unless `preserve_thinking` is true (test
`test_defaults_and_reasoning_retention`, `tests/text/test_chat_templates.py:95`).

### G3 — `tojson` convention: NInfer matches transformers, not stock Jinja2 (reference divergence, intentional)

- Stock Jinja2 3.1.6 `tojson` sorts object keys (`sort_keys=True`) and HTML-escapes
  (`<` -> `\u003c`, `&`, `'`) and escapes non-ASCII.
- NInfer's engine emits insertion-ordered, UTF-8 (`北京🌏`), unescaped JSON; the repo explicitly
  configures the Python reference the same way (`tests/text/test_chat_templates.py:30-32`,
  "Transformers' chat-template JSON convention").

Probe (`probe_hostile.py` tojson block):

```
tojson_dict_html  native='{"a": "\u003cb\u003e", "u": "\ud83c\udf0f"}'
                  plain='{"a": "<b>", "u": "🌏"}'   cpp='{"a": "<b>", "u": "🌏"}'
```

The one froggeric-suite cell that fails in the C++ shim run (cell 94, "system + tools") is exactly
this: the suite's Python side uses stock Jinja2 `tojson` (sorted keys) against C++ insertion order.
Re-running the same four-way comparison with the transformers convention gives byte-identical
output on both engines:

```
with transformers tojson convention: py_j==py_o: True  cpp_j==cpp_o: True  py_j==cpp_j: True  py_o==cpp_o: True
```

This is a correctness-neutral difference for the model, but it breaks byte-level parity with any
harness that renders through stock Jinja2 defaults (including froggeric's own CI).

### G4 — Deep macro recursion crashes the renderer with SIGSEGV (latent crash, not triggered by v22.5)

```bash
printf '%s\n' '{"source":"{% macro f(n) %}{% if n > 0 %}{{ f(n-1) }}x{% endif %}{% endmacro %}{{ f(10000) }}","context":{}}' \
  | /tmp/opencode/ninfer_jinja_test --render; echo "rc=$?"
# rc=-11 (SIGSEGV); depth 1000 succeeds with 'x'*1000
```

Python/Jinja2 raises `RecursionError` at depth 500 already, i.e. it fails closed. Upstream llama.cpp
has an open PR for exactly this: [ggml-org/llama.cpp#19085](https://github.com/ggml-org/llama.cpp/pull/19085)
("fix(jinja): enforce recursion limit to prevent stack overflow", open since 2026-01-25, proposes a
macro depth limit of 100; references a DoS PoC). NInfer's fork does not contain it. Froggeric v22.5
has no recursive macro, so the pinned template does not trigger it; the risk applies to any
user-supplied chat template (the server exposes `--chat-template`, `src/serve/serve_options.cpp:290`).

### G5 — Object keys named after engine built-ins are shadowed by dot access (divergence, latent)

NInfer resolves `o.string`, `o.length`, `o.tojson`, `o.items`, ... as callables/methods before
looking up object keys; using them as plain attributes errors. Python Jinja2 resolves dict methods
for `items/keys/get` (renders a bound method) but resolves `string`/`length`/`tojson` to the key.

```
items_key   py='<built-in method items of dict object at ...>'  cpp='ERR ... Function is not a string value'
string_key  py='v'                                              cpp='ERR ... Function is not a string value'
length_key  py='v'                                              cpp='ERR ... Function is not a string value'
subscript   py='value'                                          cpp='value'   (o['items'] works)
```

This is the same failure class as the minja report
[ggml-org/llama.cpp#25916](https://github.com/ggml-org/llama.cpp/issues/25916) (tool schema with a
property named `items`; closed as not planned). Froggeric is not affected: tools are serialized with
`|tojson` (line 175) and arguments are iterated with the method call `tc.arguments.items()` (line
378), both of which work. Latent for templates that dot-access a schema field named
`items`/`length`/`string`/... .

### G6 — `is iterable` excludes mappings; `is mapping` includes engine namespaces (divergence, guarded in all three templates)

```
{{ d is iterable }}  d={"a":1}:   py=True   cpp=False
{{ ns is mapping }}  ns=namespace(...): py=False  cpp=True
```

Static cause: `test_is_iterable` is `test_type_fn<value_array, value_string, value_undefined>`
(`third_party/llama-jinja/jinja/value.cpp:361`), i.e. no `value_object`; `test_is_mapping` is
`test_type_fn<value_object>` (line 363) and the engine implements `namespace()` as an object
(line 255). All three templates guard with `... is not mapping`, and namespaces are never tested, so
no rendering impact.

### G7 — Undefined attribute access is lenient (divergence, not breaking)

```
{{ missing.attr }}   py=UndefinedError (stock Jinja2), cpp='' (empty)
```

Froggeric guards every optional attribute with `is defined`/`is not none` (verified by 70-scenario
parity under Python's `StrictUndefined`-style suite and `probe_constructs.py`), so this only means
malformed contexts fail silently on NInfer where the reference raises.

### G8 — Non-integer `max_tool_arg_chars` (degenerate input)

`max_tool_arg_chars=2.5`: Python raises `TypeError: slice indices must be integers`; NInfer truncates
at 2 and reports `original length 6 chars`. Integer and string-typed values agree (strings raise on
both sides at the `> 0` comparison). Minor; only reachable by sending a float in
`chat_template_kwargs`.

### General engine gaps seen while probing (not used by froggeric, relevant to custom templates)

- `map('filter')`: `ERR NotImplemented: map: filter-mapping not implemented`; `sum`: `ERR Unknown
  (built-in) filter 'sum' for type Array`. Upstream tracks these in open PR
  [ggml-org/llama.cpp#24033](https://github.com/ggml-org/llama.cpp/pull/24033) ("jinja: implement
  map('filter')", which also notes `sum` is unsupported and hidden by the upstream test helper).
- `{{ s | join('') }}` (string receiver) is registered as `string_join_not_implemented`
  (`value.cpp:755`); array `join` is implemented (line 827) and is what froggeric uses.
- Dot integer literals (`l.0`) already work (upstream merged
  [ggml-org/llama.cpp#28817](https://github.com/ggml-org/llama.cpp/pull/28817) on 2026-09-12, before
  the pinned fork base 2026-09-15): `{{ l.0 }}|{{ l.1 }}` -> `a|b`.
- `raise_exception` works and maps to a render error (both engines error; NInfer reports
  `Error: Jinja Exception: <message>`), and the failed render does not poison later renders.

## 5. Engine-populated context, concretely

For a request through the OpenAI route (`src/serve/openai_chat_request.cpp`,
`src/serve/translate.cpp`), the context handed to the template contains:

1. `messages` — always string content when text-only, `[{"type":"text"|"image"|"video", ...}]` when
   media is present; `reasoning_content`, `tool_calls[].function.{name,arguments}` (arguments parsed
   from JSON), `tool_call_id` when present (`chat_template.cpp:166-221`).
2. `tools` — parsed JSON objects (`chat_template.cpp:223-229`).
3. `add_generation_prompt`, `continue_final_message` (`chat_template.cpp:154-156`).
4. Special tokens (`bos_token`, `eos_token`, ... `additional_special_tokens`, `chat_template.cpp:97`).
5. Typed options when resolved: `enable_thinking`, `preserve_thinking`, `reasoning_effort`,
   `add_vision_id`.

Froggeric additionally looks for the variables in G2 plus inline `<|think_*|>` tags; only the tags
are engine-independent (template-internal). `preserve_reasoning` in particular is never populated by
NInfer even though froggeric prefers it; since `template_parameters` erases null-valued
`preserve_thinking` (`chat_template.cpp:65-68`) and froggeric's fallback is `true`, an unset
`preserve_thinking` keeps reasoning, whereas NInfer's qwen3_6 default drops it.

## 6. Sources that could not be verified / limitations

- The full NInfer Engine request path was not executed end-to-end (no server run in this session).
  G1/G2 claims about what the request layer passes are static citations to
  `src/serve/{request.h,translate.cpp,openai_chat_request.cpp,serve_options.cpp}` and
  `src/models/qwen3_5/frontend/chat_template.cpp`; the rendering consequences were probed at the
  renderer level.
- The CMake target build (`ninfer_jinja_test`) could not be produced because CUDA toolkit discovery
  fails on this Nix host (`Failed to find nvcc`). The standalone build compiles the identical host
  sources listed in `tests/cmake/CoreTests.cmake:46-48`; behavioral evidence comes from that binary.
- Python reference is Jinja2 3.1.6 (not the transformers-pinned version); the `tojson` replacement
  used is the one the repo's own independent renderer defines
  (`tests/text/test_chat_templates.py:30-32`).
- Froggeric's suite was run with its own stock-Jinja2 environment for the Python side of cell 94,
  which is why that cell mismatches (G3); this is a harness convention difference, proven by the
  four-way re-compare.
- Upstream llama.cpp PR/issue states cited as of 2026-09-15 (GitHub API);
  #19085 and #24033 are open/unmerged, #25916 was closed as not planned, #28817 is merged.
