# third_party

The llama.cpp source is not committed. It is downloaded by tag and patched, so the build is reproducible.

## llama.cpp b10434

b10434 is the tag studied by the prior work (Patodiya, arXiv 2609.04748). llama.cpp is MIT-licensed; see its
`LICENSE` file after download.

```bash
cd third_party
curl -L -o b10434.tar.gz https://github.com/ggml-org/llama.cpp/archive/refs/tags/b10434.tar.gz
tar xzf b10434.tar.gz                                   # creates llama.cpp-b10434/
python ../probes/apply_probe_patch.py llama.cpp-b10434  # adds the GGML_PROBE_FLAGS switches
```

- **`probes/apply_probe_patch.py` is the source of truth.** It asserts that every edit matches exactly
  once, and it refuses to patch a tree twice.
- **`probes/probe_interventions_b10434.patch`** is the same change as a unified diff, for reading.
- **Default behaviour is unchanged:** with `GGML_PROBE_FLAGS` unset or 0, the patched build behaves
  exactly as upstream b10434.
