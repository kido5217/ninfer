# Froggeric Qwen-Fixed-Chat-Templates fixture

Byte-exact copies of the external reference template, vendored so the C++ renderer can be
qualified against an independent Python Jinja2 oracle without network access.

| File | sha256 | Bytes |
|---|---|---|
| `chat_template.jinja` | `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2` | 28,234 |
| `chat_template_oneline.txt` | `eecae0e068e60f9c8665f0085b589d3e1c41508d359776c62018512c40b5879b` | 22,391 |

- Upstream: <https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates> — `qwen3.8-froggeric-v22.5`
- Pinned revision: `855bffc49448e299789730ff92c9b8d834d6cc14`
- License: Apache-2.0, inherited from Qwen (upstream README, "License" section); copyright
  froggeric and the template's contributors.
- Retrieval:

  ```bash
  git clone https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates /tmp/froggeric
  git -C /tmp/froggeric checkout 855bffc49448e299789730ff92c9b8d834d6cc14
  cp /tmp/froggeric/chat_template.jinja /tmp/froggeric/chat_template_oneline.txt \
     tests/fixtures/text/froggeric/
  ```

`tests/text/test_chat_templates.py` asserts these digests and renders both files against Python
Jinja2: a mismatch means the fixture drifted and the parity claim is void.
