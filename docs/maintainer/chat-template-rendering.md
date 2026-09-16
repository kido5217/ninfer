# Chat-template rendering contract

Scope: how rendered chat-template bytes are classified, which consumers depend on the
classification, and what must stay true when the renderer, tokenizer, or prompt layout change. It
covers the maintained templates in `tools/chat_templates/` and external templates selected by
`--chat-template`, including froggeric's Qwen-Fixed-Chat-Templates.

## Content origin vs template origin

A render contains two kinds of bytes:

- **structural** — template-authored bytes: role markers, control tokens, fixed template text;
- **literal** — ordinary request data substituted into the template.

Every rendered string part carries a `literal` flag. `TemplateOutput::literal_spans`
(`src/text/jinja.h`) records the byte ranges of literal parts, and `RenderedChat::literal_spans`
(`src/models/qwen3_5/frontend/chat_template.h`) carries them through the frontend.
`src/text/byte_span.h` owns `ByteSpan` and the overlap predicate shared by the consumers.

Context values are literal by default. Only engine-supplied token variables
(`TemplateRenderOptions::control_variables`, built from the tokenizer's `special_tokens` map) opt
out and render as structure. The classification survives Jinja transforms: `tojson`, `format`,
`join`, `indent`, `strftime_now`, array/object reprs, and case mapping keep or remap the literal
flag through the vendored language core (`third_party/llama-jinja/jinja/`).

Consequences: a special-token spelling quoted in message content, tool arguments, or any ordinary
template kwarg is text. It is never recognized as an added token and never counts as a media
placeholder. That behavior fixes Neroued/ninfer#258; the implementation was adopted from upstream
commit `8eaed538`.

## Consumers

| Consumer | Contract |
|---|---|
| Added-token matching (`Tokenizer::encode_with_boundaries`) | An added token is recognized only when its bytes do not overlap a literal span. Content bytes are encoded as ordinary text; literal spans never split ordinary normalization/BPE runs. |
| Prompt layout (`inspect_prompt_layout`) | `<|im_start|>` / `<|im_end|>` role markers, `<think>` boundaries, the canonical reasoning-close serialization, and media pads count only when structural. A media placeholder is one fully structural `<|vision_start|>` *pad* `<|vision_end|>` wrapper, bound to request media inputs in order. |
| Media expansion (`expand_placeholders`) | Placeholder byte ranges are replaced by model token runs; literal spans and every other rendered boundary are mapped through the same expansion. |
| Cache-prefix decisions (`same_prefix`) | Two renders may share a cache prefix only when the prefix bytes and their literal/structural partition agree. Continuation truncation clips regions and literal spans with the text. |

## Error surface (fork-local)

Media contract violations throw `std::invalid_argument`. This fork's messages name the owning
message by role and index and report expected/actual counts, for example
`chat template media placeholder 0 expects video input but the request supplies image (message user[0])`.
Upstream keeps the generic messages; this is the one intentional engine divergence in the
chat-template stack, and `tests/models/qwen3_5/test_frontend.cpp` locks it.

## Evidence

- `tests/text/test_jinja.cpp` — literal classification across transforms, control variables, and
  case mapping; recursion bounds.
- `tests/text/test_chat_templates.py` — C++ renderer against the Python Jinja2 oracle over both
  maintained templates and the pinned froggeric v22.5 fixture (provenance and digests in
  `tests/fixtures/text/froggeric/README.md`).
- `tests/models/qwen3_5/test_frontend.cpp` — quoted-marker tokenization, media binding, and the
  error surface.
