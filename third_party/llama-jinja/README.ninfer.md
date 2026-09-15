# NInfer Jinja source base

This maintained implementation originates from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`7609846557c50f9d984719a9e1e8c5f3d02f807b`](https://github.com/ggml-org/llama.cpp/commit/7609846557c50f9d984719a9e1e8c5f3d02f807b),
`common/jinja/`, under the MIT license; see [LICENSE](LICENSE).

NInfer maintains this fork for chat-template rendering and adopts upstream fixes selectively.

## Adopted upstream fixes

- [ggml-org/llama.cpp#19085](https://github.com/ggml-org/llama.cpp/pull/19085) — runtime
  recursion limit: `context::recursion_depth` / `max_recursion_depth` and the
  `recursion_guard` in `statement::execute`. A user-supplied template that recurses without a
  base case (for example a recursive macro) now raises a template error instead of crashing
  the process. The upstream PR was still open (unmerged) when NInfer adopted it; re-check it
  when syncing this fork.

## Fork-local changes

- Parser recursion bound (kido5217/ninfer#15): `parser::depth_guard` / `k_max_depth` in
  `jinja/parser.cpp` rejects deeply nested template source with
  `Parser Error: Max recursion depth exceeded`. Upstream has no equivalent, so keep this guard
  when syncing.

Unicode case data is generated from Unicode 17.0.0 `SpecialCasing.txt` and
`DerivedCoreProperties.txt`, under [UNICODE-LICENSE](UNICODE-LICENSE):

```bash
python3 tools/generate_jinja_unicode.py /path/to/unicode-17.0.0
```
