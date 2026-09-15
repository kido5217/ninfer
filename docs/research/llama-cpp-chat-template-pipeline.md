# llama.cpp's chat-template pipeline: template structure vs message content

Ticket: [kido5217/ninfer#5](https://github.com/kido5217/ninfer/issues/5) (`wayfinder:research`)

- **Pinned llama.cpp revision read**: `38a5b42d9a3e82e0a586bcd1caed121f36c87a73` (master, 2026-09-15 20:57 +0200, "HIP: Enable AllReduce for ROCm (#27825)"). All `file:line` citations below are at this revision.
- **Fork base named in the ticket**: `7609846557c50f9d984719a9e1e8c5f3d02f807b` (2026-09-15 13:50 +0200, "rpc : hash-cache only weights (#28789)").
- **Method**: shallow clone (`git clone --depth 400`), source read directly; upstream intent read from GitHub issues/PRs via the GitHub API (`gh`). No secondary write-ups were used.

The short answer: at the pinned revision llama.cpp renders a chat template into **one string** and tokenizes that full string with `parse_special = true`. The Jinja engine does tag which rendered bytes came from user input (`jinja::string_part::is_input`), but the tag is flattened away before tokenization and no downstream consumer reads it. Upstream knows this (tracked in issue #28249) and has landed only the tokenizer primitive needed to fix it. Multimodal markers are random per server run to avoid collisions with user text; media count/type validation happens in `mtmd`. Tool-call parsing is generated from the template text itself.

## 1. How a rendered prompt becomes tokens

The default Jinja route renders in `common/jinja`, then hands a plain `std::string` to the server, which tokenizes it in one call:

- `common_chat_templates_apply()` dispatches to `common_chat_templates_apply_jinja()` when `use_jinja` is set (`common/chat.cpp:1424-1428`). The Jinja route is default (`common/chat.h:255`).
- The autoparser generator renders the prompt with `common_chat_template_direct_apply()` (`common/chat-auto-parser-generator.cpp:36`), which reaches `common_chat_template_direct_apply_impl()` (`common/chat.cpp:905-966`):
  - `jinja::global_from_json(ctx, inp, inputs.mark_input)` (`common/chat.cpp:949`) builds the context and marks input strings;
  - `auto parts = jinja::runtime::gather_string_parts(results);` (`common/chat.cpp:954`);
  - `std::string result = parts->as_string().str();` (`common/chat.cpp:956`) — the only consumer of `parts` is this flattening.
- `common_chat_params.prompt` is a single `std::string`; there is no parts field (`common/chat.h:269-283`).
- The server copies it into the task JSON: `llama_params["prompt"] = chat_params.prompt;` (`tools/server/server-common.cpp:1363`).
- `handle_completions_impl()` tokenizes (`tools/server/server-context.cpp:4291-4300`):
  - multimodal path: `process_mtmd_prompt(...)` (`tools/server/server-context.cpp:4296`), which builds `mtmd_input_text{ ..., /* add_special */ true, /* parse_special */ true }` (`tools/server/server-common.cpp:952-957`);
  - every other path: `tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ...)` (`tools/server/server-context.cpp:4299`) — `add_special = true`, `parse_special = true`.
- The standalone `/tokenize` endpoint also defaults to `parse_special = true` (`tools/server/server-context.cpp:5089-5093`).

So: **the whole template output is tokenized as one string.** Nothing at tokenization distinguishes literal template text from interpolated message content. `parse_special = true` means any string match for a cached special token inside the rendered text becomes that token; `parse_special = false` skips `CONTROL`/`UNKNOWN` tokens but still pre-tokenizes `USER_DEFINED` ones (`src/llama-vocab.cpp:3232-3237`, cache built at `src/llama-vocab.cpp:3012-3015`). That last detail matters: "parse_special=false" is not a complete escape hatch.

## 2. Special-token literals in message content today

The renderer does tag origin, but the tag dies before the tokenizer:

- `jinja::string_part` carries `bool is_input = false; // may skip parsing special tokens if true` (`common/jinja/string.h:16-21`); transformations preserve the flag (`common/jinja/string.h:11-15`, `common/jinja/string.cpp:39-43`, `98-104`).
- `global_from_json(..., mark_input)` marks JSON strings as input (`common/jinja/value.cpp:1358-1370`, `1454-1463`); `runtime::gather_string_parts()` joins adjacent parts with the same flag (`common/jinja/runtime.h:761-774`).
- `autoparser::generation_params::mark_input = true` by default (`common/chat-auto-parser.h:73`), and it is passed through at `common/chat.cpp:949`.
- After the flatten at `common/chat.cpp:954-956` nothing downstream can tell the difference: there are **zero** references to `is_input`/`mark_input` under `tools/` (server, mtmd). The server's four `tokenize_input_prompts` call sites all pass `parse_special = true` (`tools/server/server-context.cpp:4299`, `4883`, `5427`; `tools/server/server-common.cpp:1828`).

Upstream intent, in order:

- **PR #18462** ("implement new jinja template engine", merged 2026-01-16, merge commit `c15395f73c42805d9609a55749a1d4b5b2251379`) introduced the engine and input marking, explicitly "implemented in this PR, but left unused. In a follow-up PR, it will be added to server and enabled via a flag". The PR description documents the `is_input` semantics and the two caveats: dynamically built tokens (`'<|' + role + '|>'`) are treated as input, and template-added spaces become separate tokens. https://github.com/ggml-org/llama.cpp/pull/18462
- `common/jinja/README.md:31-88` states the contract and the caveat: "Downstream applications like `llama-server` can then make informed decisions about special token parsing based on the `is_input` flag."
- **Issue #24382** ("Eval bug: Special Token Injection", 2026-06-09) reported exactly the failure mode: user content `A<|turn>B` tokenized as control token 105 inside the Gemma-4 turn structure. ngxson: the infrastructure post-dates `chat.cpp`, and `chat.cpp`'s regex replacements make sanitation "extremely tricky". Closed 2026-09-02 as replaced by #28249. https://github.com/ggml-org/llama.cpp/issues/24382
- **Issue #26532** ("jinja input marking is never consumed by llama-server", 2026-08-03, closed 2026-08-07). Documents the same state: `chat-auto-parser.h` defaults `mark_input = true`, `chat.cpp` calls `global_from_json(..., mark_input)`, `tools/server/` has zero references, all four `tokenize_input_prompts` calls pass `parse_special = true`; a ChatML payload in user content tokenizes identically to a real system turn. Maintainer response (CISC): still on ngxson's TODO, "not trivial (if at all 100% feasible) so not high priority". https://github.com/ggml-org/llama.cpp/issues/26532
- **Issue #26273** ("Add a CLI flag to escape special tokens in user input and tool responses", 2026-07-29, closed stale 2026-09-12) contains a detailed but unmerged proof-of-concept: refactor `chat.cpp` to emit `common_chat_rendered_part{text, is_input}` (`prompt_parts`), add `tokenize_prompt_parts_with_input_escaping()`, and select `parse_special=false` for `is_input` parts in the server. The author states "I don't expect this patch to land." https://github.com/ggml-org/llama.cpp/issues/26273
- **Issue #28249** ("wiring up jinja input marking", open, created 2026-09-02, assigned) is the live tracker. Checklist: `[x] add mtmd_tokenize_from_parts for fine-control per-segment tokenization`, `[ ] Refactor chat.cpp to output parts instead of a single string`, `[ ] Wire it up to server`. https://github.com/ggml-org/llama.cpp/issues/28249
- The only landed piece is **PR #28250** ("mtmd: add mtmd_tokenize_from_parts()", merged 2026-09-02, merge commit `7339054744f109c4cd89b75689dbb8a2c154d60e`) which explicitly "Target support jinja's input marking: #28249". `mtmd_tokenize_from_parts()` exists at `tools/mtmd/mtmd.cpp:1750-1772` and its text parts carry per-part `parse_special` (`tools/mtmd/mtmd.h:69-79`, `tools/mtmd/mtmd.cpp:1180-1198`, `1750-1766`). The server does not use it for chat. https://github.com/ggml-org/llama.cpp/pull/28250
- Related but unresolved: **#26309** ("post /tokenize `parse_special:false` is not being respected", 2026-07-30, closed stale 2026-09-13). At the pinned revision the endpoint does pass the caller's flag through (`tools/server/server-context.cpp:5089-5093`); I did not reproduce whether the reported Gemma-4 behavior persists, so treat the issue as unverified.

## 3. Multimodal markers: emission and validation

- Request parsing rewrites OpenAI content parts (`image_url`, `input_audio`, `input_video`) into `{"type":"media_marker","text": get_media_marker()}` before templating (`tools/server/server-common.cpp:1236-1283`). The template then renders the marker string wherever the media content part was.
- The marker is **random per server run**: `"<__media_" + random_string() + "__>"`, overridable via `LLAMA_MEDIA_MARKER` (`tools/server/server-common.cpp:134-143`), and is published in `/props` (`tools/server/server-context.cpp:4623`). This was **PR #21962** (merged 2026-04-15), the fix for **#21955** ("hardcoded media marker is not transparent for user input"), where user text containing `<__media__>` caused a 400 `Failed to tokenize prompt`. https://github.com/ggml-org/llama.cpp/issues/21955 https://github.com/ggml-org/llama.cpp/pull/21962
- `mtmd_tokenize()` splits the rendered text on the marker string (`tools/mtmd/mtmd.cpp:1154`, `1688-1707`), consumes one bitmap per marker **in the order the server passed the files** (`tools/mtmd/mtmd.cpp:1156-1165`), and hard-fails on mismatch:
  - more markers than bitmaps: `"number of media markers in text (%zu) exceeds number of bitmaps (%zu)"` (`tools/mtmd/mtmd.cpp:1158-1160`);
  - count mismatch: same check after expansion (`tools/mtmd/mtmd.cpp:1167-1175`).
- Media kind is taken from the bitmap (`is_audio`, `tools/mtmd/mtmd.cpp:1342`), with model capability checks ("model does not support vision input", `tools/mtmd/mtmd.cpp:1347-1350`; "model does not support audio input", `tools/mtmd/mtmd.cpp:1560-1563`); only one media type per call (`tools/mtmd/mtmd.cpp:1341`).
- A rendered pad with no media is therefore a request error, surfaced as HTTP 400 `"Failed to tokenize prompt"` from `handle_completions_impl`'s catch (`tools/server/server-context.cpp:4343-4345`); issue #21955 shows the exact log (`number of bitmaps (0) does not match number of markers (2)`).
- Validation is **count/type/order by media kind only** — there is no check that a marker corresponds to a specific media item beyond positional order, and user-typed text is scanned for the marker too. Collisions are made impractical by the random marker, not impossible: the marker is statically visible in `/props`, so this mitigates accidents, not a motivated injector.

## 4. Tool calls: conventions and their coupling to the template

At the pinned revision, output parsing is generated **from the template source**, not from a hand-maintained per-format list:

- Legacy format enum is now only `CONTENT_ONLY` plus PEG formats (`PEG_SIMPLE`, `PEG_NATIVE`, `PEG_GEMMA4`, `PEG_MINIMAX_M3`) (`common/chat.h:228-238`); the Jinja route emits `COMMON_CHAT_FORMAT_PEG_NATIVE` (`common/chat-auto-parser-generator.cpp:38`).
- `autoparser::autoparser::analyze_template(tmpl)` performs differential analysis of the template; `build_parser()` builds the PEG parser, using template-derived literals as parser tokens and as grammar triggers/stops (`common/chat-auto-parser-generator.cpp:23-97`; `common/chat-auto-parser.h:380-410`). `preserved_tokens` and `additional_stops` come from the analysis (`common/chat-auto-parser-generator.cpp:39-41`).
- Tool wire formats are classified, not fixed: `tool_format { NONE, JSON_NATIVE, TAG_WITH_JSON, TAG_WITH_TAGGED }` (`common/chat-auto-parser.h:148-153`) with analysis fields for the concrete markers (`section_start`, `per_call_start`, `name_prefix`, `args_separator`, ...) (`common/chat-auto-parser.h:170-200`).
- Handwritten specialized handlers still exist for known model families: Ministral/Magistral 3, GPT-OSS, Muse Glimmer, Functionary v3.2, Kimi K2 Thinking, Kimi K3, Cohere2 MoE, LFM2, LFM2.5, GigaChatV3, MiniMax-M3, DeepSeek V3.2/V4, MiniCPM5, Qwen3-Coder (`common/chat.cpp:1090-1210`); the legacy C++ template path is `common/chat.cpp:1359-1422`.
- Coupling: the parser's literals are whatever the template emits. Any template edit can silently move parser boundaries, and the parser operates on generated text only — the template/content split is not available to it either.

## 5. Upstream delta after the fork base

`7609846..38a5b42` is exactly 4 commits, and **none touch chat templates or special-token handling**:

```
38a5b42 2026-09-15 HIP: Enable AllReduce for ROCm (#27825)
9f31776 2026-09-15 opencl: choose the MoE expert matmul by batch size for speculative decoding/MTP (#27637)
d1d3c33 2026-09-15 ci: build MUSA for only 1 arch (#28944)
6011c34 2026-09-15 docs: Rule of thumb for AI review time [no ci] (#28945)
```

`git log 7609846..HEAD -- common/chat.cpp common/chat.h common/jinja tools/server tools/mtmd` is empty; the diff touches only `.github/workflows`, `CONTRIBUTING.md`, `ggml/src/ggml-cuda/allreduce.{cu,cuh}`, `ggml/src/ggml-cuda/vendors/hip.h`, `ggml/src/ggml-opencl/*`. There is no upstream chat-template/special-token fix to pick up between the fork base and current master, and the input-marking wiring (#28249) remains open.

## 6. Adopt/reject for NInfer

Context (NInfer current state, for grounding the decision):

- NInfer's Jinja already tracks provenance per rendered region: `TemplateInputRegion{pointer, tag}` → `TemplateOutputRegion{tag, begin, end, source_offset}` (`src/text/jinja.h:17-39`), surfaced in `RenderedChat` together with media placeholder byte spans (`src/models/qwen3_5/frontend/chat_template.h:113-126`).
- The rendered text is encoded in one pass with added-token parsing enabled by default (`EncodeOptions.parse_added_tokens = true`, `src/models/qwen3_5/frontend/tokenizer.h:16-18`; literal added-token scan at `src/models/qwen3_5/frontend/tokenizer.cpp:877-924`; single call at `src/models/qwen3_5/frontend/processor.cpp:760-761`).
- Qwen media pads are located by scanning for the fully wrapped pad (`src/models/qwen3_5/frontend/prompt_layout.cpp:116-141`) and validated against input media count, order and modality, plus exact placeholder bytes (`src/models/qwen3_5/frontend/processor.cpp:541-567`).
- The shipped Qwen templates render content raw, with no escaping (`tools/chat_templates/qwen3_6.jinja:9-45`, `97-99`).

**Adopt**

1. **Origin-aware tokenization at the parts level.** Make the content-vs-template distinction reach the tokenizer: encode rendered byte spans that came from message content (user/tool/reasoning text) with added-token matching disabled, while template-authored literals keep it. llama.cpp validated the model (`is_input` + per-part `parse_special` via `mtmd_tokenize_from_parts`) but has not wired it; NInfer's region tags already provide the spans, so it can do this deterministically instead of repeating llama.cpp's five-month gap.
2. **Document the escape semantics precisely.** llama.cpp's `parse_special=false` still matches `USER_DEFINED` tokens (`src/llama-vocab.cpp:3232-3237`); a NInfer content escape must disable *all* added-token matching for content spans (`parse_added_tokens=false` is exactly that primitive) and state that template literals are the only source of structural tokens. Also account for the README caveats: tokens constructed by concatenation inside the template, and template-added spaces, legitimately shift tokenization.
3. **Keep NInfer's stricter media validation.** Count + per-item order + modality + exact placeholder bytes (`processor.cpp:541-567`) is stronger than llama.cpp's count check plus positional bitmap consumption (`mtmd.cpp:1154-1175`); no change needed beyond extending the same rigor to placeholders that appear in content-origin spans (they must never be treated as media).
4. **Use provenance, not marker randomization, for anti-collision.** Because NInfer knows which bytes the template emitted for media, content-injected `<|vision_start|><|image_pad|><|vision_end|>` can be classified as content and left as text; llama.cpp's random per-run marker (`server-common.cpp:134-143`) is a string-pipeline workaround whose marker is still discoverable via `/props`.

**Reject**

1. **Single flattened prompt string.** `chat.cpp:954-956` + `common_chat_params.prompt` (`chat.h:269-283`) is the root cause of the whole issue; NInfer should carry typed rendered parts through to encoding, not adopt a string-only interface.
2. **Differential/template-generated output parsing.** llama.cpp generates its PEG tool parser from arbitrary template source (`chat-auto-parser-generator.cpp:23-97`). NInfer supports fixed Qwen3.5-family frontends with a known template and parser; a template-analyzing parser generator is cost without a product requirement.
3. **Global `parse_special=true`** for prompt tokenization (`server-context.cpp:4299`, `server-common.cpp:952-957`), and any content-escaping scheme that relies on it being flipped globally (it would also disable structural tokens).
4. **Random per-server media marker as the safety mechanism** (`server-common.cpp:134-143`): correct-by-probability and publicly exposed rather than correct-by-construction.

**Not verified / open**

- Whether #26309's `/tokenize` `parse_special=false` bug is actually fixed at the pinned revision; the code passes the flag through, but the issue predates recent tokenizer changes and I did not run a model. No llama.cpp build or inference was performed; all findings are from source and upstream issue/PR text.
- #28249's two remaining items (chat.cpp parts refactor, server wiring) are unlanded; if they land, the exact `is_input`-driven tokenization policy may become worth re-reading before NInfer finalizes its own policy.

## Sources

Pinned llama.cpp commit `38a5b42d9a3e82e0a586bcd1caed121f36c87a73` (fork base `7609846557c50f9d984719a9e1e8c5f3d02f807b`):

- `common/chat.cpp:905-966`, `1215-1355`, `1359-1422`, `850-872`
- `common/chat.h:228-283`
- `common/chat-auto-parser.h:54-78`, `148-200`, `380-410`
- `common/chat-auto-parser-generator.cpp:23-97`
- `common/jinja/string.h`, `common/jinja/string.cpp:39-104`, `common/jinja/value.cpp:1358-1370`, `1454-1463`, `common/jinja/runtime.h:761-774`, `common/jinja/README.md:31-88`
- `src/llama-vocab.cpp:3012-3015`, `3226-3238`
- `tools/server/server-common.cpp:134-143`, `867-901`, `930-1029`, `1151-1163`, `1236-1283`, `1360-1363`, `1828-1833`
- `tools/server/server-context.cpp:2185-2248`, `4257-4345`, `5084-5093`, `5404-5427`
- `tools/mtmd/mtmd.h:69-79`, `300-320`, `tools/mtmd/mtmd.cpp:1130-1198`, `1338-1350`, `1560-1563`, `1688-1707`, `1736-1772`

Upstream discussions (primary):

- PR #18462 (jinja engine + input marking, merged): https://github.com/ggml-org/llama.cpp/pull/18462
- Issue #24382 (special token injection): https://github.com/ggml-org/llama.cpp/issues/24382
- Issue #26532 (input marking never consumed): https://github.com/ggml-org/llama.cpp/issues/26532
- Issue #26273 (escape-special flag FR + POC): https://github.com/ggml-org/llama.cpp/issues/26273
- Issue #28249 (wire input marking; open): https://github.com/ggml-org/llama.cpp/issues/28249
- PR #28250 (`mtmd_tokenize_from_parts`, merged): https://github.com/ggml-org/llama.cpp/pull/28250
- Issue #21955 (media marker collision): https://github.com/ggml-org/llama.cpp/issues/21955
- PR #21962 (random media marker, merged): https://github.com/ggml-org/llama.cpp/pull/21962
- Issue #26309 (`/tokenize` parse_special, closed stale): https://github.com/ggml-org/llama.cpp/issues/26309

NInfer references (current master at time of writing):

- `src/text/jinja.h:17-39`, `src/models/qwen3_5/frontend/chat_template.h:113-126`, `src/models/qwen3_5/frontend/processor.cpp:541-567`, `733-761`, `src/models/qwen3_5/frontend/prompt_layout.cpp:116-141`, `src/models/qwen3_5/frontend/tokenizer.h:16-18`, `src/models/qwen3_5/frontend/tokenizer.cpp:877-924`, `tools/chat_templates/qwen3_6.jinja:9-45`, `97-99`
