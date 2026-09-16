# Upstream master on the froggeric v22.5 parity corpus

Ticket: [kido5217/ninfer#18](https://github.com/kido5217/ninfer/issues/18) ("Research: upstream
master against the froggeric parity corpus"), parent #4.

**Question.** Does upstream master `1d8587bc` (which contains the maintainer's literal-span fix
`8eaed538` "fix(frontend): preserve literal content in chat templates") match the Python Jinja2
oracle on the full froggeric v22.5 parity corpus, using this fork's harness from
`test/froggeric-parity` @ `8eaae08d` unmodified?

**Verdict: yes — no divergence on any of the 64 cells.** 60 cells render byte-identically to the
Python oracle; the 4 cells the oracle rejects via `raise_exception` are rejected by the C++ renderer
with the same message. 0 text mismatches, 0 interface/protocol mismatches. Upstream's own inline
corpus (18 cells) passes as well; it is a strict coverage subset of the fork corpus, so it would not
have caught a divergence confined to froggeric sources or to fork contexts 9-15. No such divergence
exists at `1d8587bc`.

## 1. Pinned artifacts

| Artifact | Value |
|---|---|
| upstream commit tested | `1d8587bcfe850fba310d8833552f3c0c07e3a4bd` ("perf(nvfp4): read W4A4 activation scales one tile per TMA request") |
| fix under test | `8eaed538` "fix(frontend): preserve literal content in chat templates" — ancestor of `1d8587bc` (`git merge-base --is-ancestor` verified) |
| fork harness commit | `8eaae08de1e3ce8c16e47dc5bf1aed403e9aee95` (`test/froggeric-parity`) |
| fork harness blob id | `9e88bb6065ec8496f3aba1c609a710a6c8789d7d` |
| fork harness sha256 | `bd81053b663018e78d97e9466c58c12b357af8c2e0c0c30b6cd9c2207738ae9f` (used on disk, unmodified) |
| fixture `chat_template.jinja` sha256 | `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2` (pinned; asserted by the harness) |
| fixture `chat_template_oneline.txt` sha256 | `eecae0e068e60f9c8665f0085b589d3e1c41508d359776c62018512c40b5879b` (pinned) |
| scratch worktree (left in place) | `/tmp/opencode/wt-upstream` (detached at `1d8587bc`) |
| renderer binary | `/tmp/opencode/wt-upstream/build/tests/ninfer_jinja_test` (1,387,448 bytes) |
| Python oracle | Python 3.13.15 + Jinja2 3.1.6 (`nix-shell -p python313Packages.jinja2`), pytest 9.0.3 |
| C++ toolchain | nvcc 13.1.115, host compiler GNU 15.2.0 |

The fork harness was installed by content, not by checkout; it is byte-identical to the blob at
`8eaae08d` (sha256 above). The fixture files are the exact pinned bytes. Upstream's tracked sources
in the scratch worktree are unmodified: the only overlays are the harness file and the fixture
directory, and `git status --porcelain -- src third_party tools cmake CMakeLists.txt` is empty
there. The binary was built before the overlay, from upstream's own `tests/text/test_jinja.cpp`.

## 2. Build (scratch upstream worktree)

The prescribed configure failed first because no CUDA toolkit is on `PATH` on this NixOS host
(`nvcc` missing, and the earlier scratch toolkit `/tmp/opencode/cuda13.1` referenced by the main
build has since been deleted from `/tmp`). Workaround: realize `cudaPackages_13_1.cudatoolkit` from
nixpkgs, then point CMake at it. The Nix `nvcc` does not search its own toolkit prefix for
`cuda_runtime.h`, so `NVCC_PREPEND_FLAGS` must carry the include directory or even the compiler-ID
test fails; and `pkg-config`, `ffmpeg` and `curl` are supplied through a `nix-shell` because
`cmake/Dependencies.cmake` requires them at configure time.

```bash
git -C /home/kido/network/projects/ninfer worktree add /tmp/opencode/wt-upstream upstream/master
CUDA=$(nix-build --no-out-link -E \
  'with import <nixpkgs> { config = { allowUnfree = true; }; }; cudaPackages_13_1.cudatoolkit')
# -> /nix/store/0zw5qirgxs5w0kavpxd16p3g4z3jiymz-cuda-merged-13.1

nix-shell -p pkg-config ffmpeg curl --run "export NVCC_PREPEND_FLAGS=-I$CUDA/include; \
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
        -DCUDAToolkit_ROOT=$CUDA -DCMAKE_CUDA_COMPILER=$CUDA/bin/nvcc && \
  cmake --build build -j --target ninfer_jinja_test"
```

Configure log checks: `The CUDA compiler identification is NVIDIA 13.1.115 with host compiler GNU
15.2.0`, `Found CUDAToolkit ... (found version "13.1.115")`, `Found Python3 3.13.15`, `Found
libcurl 8.21.0`. Build logs: `/tmp/opencode/upstream-configure.log`,
`/tmp/opencode/upstream-build.log` (ephemeral).

Harness and fixtures were extracted from the fork without checking out the branch:

```bash
git show 8eaae08d:tests/text/test_chat_templates.py \
  > /tmp/opencode/wt-upstream/tests/text/test_chat_templates.py
git archive 8eaae08d tests/fixtures/text/froggeric | tar -x -C /tmp/opencode/wt-upstream
sha256sum /tmp/opencode/wt-upstream/tests/text/test_chat_templates.py   # bd81053b...
```

## 3. Test execution and results

The harness resolves its default renderer as `<repo>/build/tests/ninfer_jinja_test`; inside the
upstream worktree that default is exactly the upstream-built binary. Three runs (all with the fork
harness unmodified):

```bash
cd /tmp/opencode/wt-upstream
# 1. full module
nix-shell -p python313Packages.jinja2 python313Packages.pytest --run \
  "python -m pytest tests/text/test_chat_templates.py -v"
#    -> 7 passed, 76 subtests passed in 0.45s
#    (test_cpp_matches_independent_renderer PASSED; test_froggeric_fixture_matches_pin PASSED)

# 2. ticket's exact argv[1] form
nix-shell -p python313Packages.jinja2 --run \
  "python tests/text/test_chat_templates.py build/tests/ninfer_jinja_test -v -k cpp_matches"
#    -> Ran 1 test ... OK

# 3. per-cell raw comparison
nix-shell -p python313Packages.jinja2 --run "python /tmp/opencode/parity_detail.py"
```

Per-cell result (4 sources × 16 contexts = 64 cells, from run 3; full output
`/tmp/opencode/parity_detail.txt`):

| Outcome | Count |
|---|---|
| C++ `ok` and text byte-identical to the Python oracle | 60 |
| C++ rejects and Python raises `TemplateError` (consistent rejection) | 4 |
| **Divergences** (text diff, or C++ ok where Python errors, or vice versa) | **0** |
| Interface/protocol errors (bad JSON, missing line, wrong count) | 0 |

Per source: 16 cells each for `qwen3_6`, `qwen3_8`, `froggeric`, `froggeric_oneline`.

**Counterexamples: none.** The only cells that are not plain string equalities are
`qwen3_6` and `qwen3_8` at context indices 8 (`messages=[system "no user"]`) and 13
(assistant-only tool call, no user message), where both sides reject the input:

```
context 8  (system "no user")           -> Python: TemplateError: No user query found in messages.
context 13 (assistant tool call only)   -> Python: TemplateError: No user query found in messages.
```

both matched by the C++ renderer with the same underlying message, e.g. for `qwen3_6`:

```
reference-case:
------------
While executing CallExpression at line 90, column 24 in source:
...lti_step_tool %}    {{- raise_exception('No user query found in messages.') }}...
                                           ^
Error: Jinja Exception: No user query found in messages.
```

The `froggeric` and `froggeric_oneline` sources render all 16 contexts successfully on both sides
(32/32 exact).

## 4. Upstream's own inline corpus

Upstream master's own `tests/text/test_chat_templates.py` (`blob 0fa8c3c6...`, 285 lines) has a
`test_cpp_matches_independent_renderer` over **2 sources × 9 contexts = 18 cells** (`tools/chat_templates/qwen3_6.jinja`,
`qwen3_8.jinja`; `SOURCES` only). Its contexts are exactly the fork's `PARITY_CONTEXTS[0..8]`
(upstream lines 193-253 match fork lines 85-144 verbatim, fields included). Running that file
against the same upstream binary:

```bash
cp /tmp/opencode/upstream-own-test_chat_templates.py tests/text/test_chat_templates.py
nix-shell -p python313Packages.jinja2 python313Packages.pytest --run \
  "python -m pytest tests/text/test_chat_templates.py -k cpp_matches -v"
# -> 1 passed, 5 deselected, 18 subtests passed in 0.11s
```

**Would it have caught the divergences found here?** There are none to catch at `1d8587bc`, so the
question is vacuous for this commit. What can be said about coverage: upstream's inline corpus is a
strict subset of the fork corpus — 2 of 4 sources (the two froggeric fixture files, including the
28 KB production template, are never rendered) and 9 of 16 contexts. The 7 fork-only contexts
(9-15) are precisely the literal/JSON-heavy cases: quoted `<|vision_start|><|image_pad|><|vision_end|>`
in text, a bare `<|image_pad|>` message, a tool argument containing `{{ user }} <|im_start|><|image_pad|>`,
tool arguments with `北京🌏`/`<|im_end|>`/booleans/null, reasoning containing `<|im_end|>`, and mixed
image/video histories. A literal-span regression confined to those constructs or to the froggeric
templates would not be detected by upstream's inline corpus alone (its separate C++ unit suite in
`tests/text/test_jinja.cpp`, which changed substantially between `8eaae08d` and master, is outside
the scope of this question). At `1d8587bc` all 64 cells agree, so adoption evidence for the
re-route decision is positive at this corpus size.

## 5. Limitations and caveats

- Tested exactly one upstream commit (`1d8587bc`); the harness pins only the fixture bytes, so a
  future upstream change to `tools/chat_templates/*.jinja` is in scope of the same harness but was
  not exercised across history.
- Oracle run used Jinja2 3.1.6 / Python 3.13.15 from nixpkgs, matching the harness's documented
  environment; no other Jinja2 version was cross-checked (the harness pins the environment's
  semantics itself via `ImmutableSandboxedEnvironment` + transformers `tojson` shim).
- The 4 both-rejected cells count as agreement per the harness contract; their rejection is
  consistent at the message level, but only `ok=false` is asserted by the harness.
- The renderer target is CPU-only; the CUDA toolkit is needed solely because `project()` declares
  the CUDA language. No GPU was used.
- The scratch worktree `/tmp/opencode/wt-upstream` (with its build) is intentionally left for
  reuse; it is not part of the repository. The findings were written and pushed from a separate
  scratch worktree that was removed afterwards.
- Ephemeral evidence under `/tmp/opencode/`: `parity_detail.py`, `parity_detail.txt`,
  `upstream-configure.log`, `upstream-build.log`, `upstream-own-test_chat_templates.py`.
