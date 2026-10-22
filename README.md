# cc-energy

Fast energy estimates for Claude Code JSONL transcripts, using a per-call model.
All inputs are token counts; energy is computed in joules and reported in Wh (J ÷ 3600).

```
E = W·(N·e_in + prefill(C)) + O·(N·e_out + decode(C)) + R·kv
```

- W = uncached input + cache writes, O = output, R = cache reads, all for one call.
- N = the model's active parameters.
- C = context in tokens seen by those tokens: R + W/2 for the W term and R + W + O/2 for the
  O term. This correctly charges cache misses, where context reappears as W.
  `--simple-context` uses C = R.
- prefill(C) is attention arithmetic (FLOPs per context token); decode(C) is KV-cache reads
  (bytes per context token).

**One fixed set of efficiency figures:** e_in = 7.8e-12 J and e_out = 3.9e-11 J per active
parameter per token; 2 pJ per attention FLOP; 230 pJ per KV-cache byte.

**Low / central / high are model architectures** (low and central assume mixture-of-experts; energy follows active, not total, parameters):

- low: 37B active parameters, 61 layers (DeepSeek V3 scale)
- central: 122B active, 96 layers (hypothetical midpoint; within the 100–150B independent estimate for Opus 4.6)
- high: 405B active, 126 layers (Llama 3.1 405B scale)

**Attention type** (`--attention`):

- `gqa` (default): grouped-query attention, 8 KV heads, 16-bit cache
- `mla`: compressed latent KV (DeepSeek V3 style), about 3.6× cheaper decoding
- `sparse`: latent attention over the top 2,048 tokens plus a light indexer
  (DeepSeek V3.2 style); long-context costs flatten

`--kv J` adds J joules per cache-read token for KV storage (default 0; try up to 0.018).

See [docs/methodology.md](docs/methodology.md) for the evidence behind the model and
coefficients.

## Install

Prebuilt binaries (Linux x86_64/ARM64/musl, macOS, Windows) are attached to each
[GitHub release](https://github.com/matweldon/cc-energy/releases).

```
# Linux / macOS
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/matweldon/cc-energy/releases/latest/download/cc-energy-installer.sh | sh

# Windows (PowerShell)
powershell -c "irm https://github.com/matweldon/cc-energy/releases/latest/download/cc-energy-installer.ps1 | iex"

# With uv (PyPI)
uv tool install cc-energy

# From source (needs Rust)
cargo install --git https://github.com/matweldon/cc-energy
```

For hooks that run every turn, install once and call `cc-energy` directly. Running through
`uvx` or `npx` each time adds process-startup overhead.

## Build

```
cargo build --release      # binary: target/release/cc-energy (~0.6 MB)
cargo test --release
```

## Use

```
cc-energy                              # cumulative over ~/.claude/projects
cc-energy ~/.claude/projects/<proj>/<session>.jsonl   # one session + its subagents
cc-energy running --line               # only what's new since last run, plus running total
cc-energy running --reset              # start the running total again
cc-energy --json                       # machine-readable
cc-energy --attention sparse           # alternative attention design
```

Paths can be directories (searched recursively) or session files. A session file also pulls in
`<session>/subagents/*.jsonl`; records marked `isSidechain` are counted as subagent work too.
`--no-subagents` excludes both.

### Running mode and the cursor

The cursor (default `~/.claude/cc-energy-cursor.json`, override with `--cursor`) stores, per file:

- the byte offset reached (fast path: seek straight to new data),
- the last request ID processed (recovery: if the file was rewritten or truncated, it rescans
  for that ID and resumes after it; if not found, it rescans from the start),
- a ring of the last 256 request IDs and the usage counted for each, so streaming duplicates
  that straddle runs are counted once, and topped up if a later line reports more tokens.

Half-written final lines are left for the next run. Cursor writes are atomic (write + rename).
A cursor records the attention type it was built with and refuses to mix totals; use
`--reset` or a separate `--cursor` per attention type.

Timings on a 637 MB synthetic transcript: full scan ~0.4 s; running mode with nothing new
~5 ms; with one new call ~11 ms.

### Wiring into Claude Code

Both the status line and hooks receive JSON on stdin containing `transcript_path`.
`--hook` reads it and processes that session plus its subagents.

Status line showing this session's total (rescans the session file; a few ms for normal sessions):

```json
{ "statusLine": { "type": "command", "command": "cc-energy cumulative --hook --line" } }
```

Global running tally updated after each turn (Stop hook), using a dedicated cursor:

```json
{
  "hooks": {
    "Stop": [{ "hooks": [{ "type": "command",
      "command": "cc-energy running --hook --line --cursor ~/.claude/cc-energy-cursor.json" }] }]
  }
}
```

## Releasing

Releases are triggered by **pushing a version tag**. Don't create the release in the GitHub
UI first: the workflow creates it, and an existing release for the same tag would clash.

1. Set `version` in `Cargo.toml` and commit (the PyPI version is read from it too).
2. Tag that commit with the same version and push the tag:

   ```
   git tag v0.1.0
   git push origin v0.1.0
   ```

This runs two workflows:

- **Release** (`.github/workflows/release.yml`, cargo-dist): builds binaries for Linux
  (x86_64, ARM64, static musl), macOS (Intel, Apple Silicon) and Windows. It also builds the
  shell and PowerShell installers and checksums, and creates the GitHub release with them
  attached. cargo-dist checks that the tag matches the crate version.
- **PyPI** (`.github/workflows/pypi.yml`, maturin): builds wheels and publishes them with
  trusted publishing. This needs a one-time "pending publisher" on pypi.org: project
  `cc-energy`, owner `matweldon`, repository `cc-energy`, workflow `pypi.yml`, environment
  `pypi`. Without it, the publish step fails, but the GitHub release is unaffected.

A suffix such as `v0.2.0-beta.1` marks the release as a pre-release. The release workflow
also runs a dry-run plan on pull requests, and the PyPI workflow can be run manually to
build wheels without publishing. To change targets or installers, edit
`dist-workspace.toml` and run `dist generate`.

## Caveats

These are marginal, bottom-up estimates at facility level (PUE included) with wide uncertainty.
Claude's architecture is not public; the central model size and the attention type are the
largest assumptions. Multiply by ~1–2 for real-world serving inefficiency. A top-down
attributional share of fleet energy is roughly 1.5–2× the central figure for heavy use.
Anthropic publishes no per-token energy data. See docs/methodology.md.
