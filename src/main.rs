//! cc-energy: fast energy estimates for Claude Code JSONL transcripts.
//!
//! Model (per API call; inputs in tokens, energy in joules, reported in Wh = J/3600):
//!   E = W*(N*e_in + prefill_attn(Cw)) + O*(N*e_out + decode_attn(Co)) + R*kv
//! where W = uncached input + cache writes, O = output, R = cache reads, N = active parameters,
//! Cw = R + W/2 and Co = R + W + O/2 are the context (tokens) seen by new and output tokens.
//! Prefill attention costs FLOPs per context token; decode attention costs KV-cache bytes read
//! per context token. Both depend on the attention type (--attention gqa|mla|sparse).
//! Efficiency figures are fixed; low/central/high are three model architectures
//! (37B / 122B / 405B active parameters).
//!
//! Modes:
//!   cumulative (default)  scan everything, print totals
//!   running               read only what's new since the last run, using a cursor file
//!                         (per-file byte offset + last request ID for recovery)

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

// ---------- model ----------

// Fixed central efficiency figures, facility level (including PUE).
/// J per attention FLOP: bf16 maths at ~40% utilisation, plus host and building overheads.
const J_PER_ATTN_FLOP: f64 = 2.0e-12;
/// J per KV-cache byte read during decode: accelerator memory streaming plus overheads.
const J_PER_KV_BYTE: f64 = 2.3e-10;
/// Weights cost per active parameter per input token (InferenceMAX-calibrated: 0.288 J at 37B).
const J_IN_PER_PARAM: f64 = 0.288 / 37.0e9;
/// Weights cost per active parameter per output token (5x input).
const J_OUT_PER_PARAM: f64 = 5.0 * J_IN_PER_PARAM;
const J_PER_WH: f64 = 3600.0;

// Attention geometry shared by all three architectures.
const Q_HEADS: f64 = 128.0;
const HEAD_DIM: f64 = 128.0;
const KV_BYTES: f64 = 2.0; // bf16 KV cache
// Grouped-query attention (Llama 3.1 style)
const GQA_KV_HEADS: f64 = 8.0;
// Multi-head latent attention (DeepSeek V3 style)
const MLA_QK_DIM: f64 = 192.0; // 128 no-position + 64 rotary
const MLA_V_DIM: f64 = 128.0;
const MLA_LATENT: f64 = 576.0; // 512 latent + 64 rotary, stored per token per layer
// DeepSeek sparse attention (V3.2 style): MLA over the top-k selected tokens plus an indexer.
const DSA_TOPK: f64 = 2048.0;
const DSA_INDEX_HEADS: f64 = 64.0;
const DSA_INDEX_DIM: f64 = 128.0;
const DSA_INDEX_FLOP_FACTOR: f64 = 0.5; // indexer runs in fp8
const DSA_INDEX_KEY_BYTES: f64 = 128.0; // fp8 index key per token per layer

#[derive(Clone, Copy)]
struct Arch {
    active_params: f64,
    layers: f64,
}

/// low / central / high: DeepSeek-V3-scale MoE, a hypothetical midpoint, Llama-3.1-405B-scale dense.
const ARCHS: [Arch; 3] = [
    Arch { active_params: 37.0e9, layers: 61.0 },
    Arch { active_params: 122.0e9, layers: 96.0 },
    Arch { active_params: 405.0e9, layers: 126.0 },
];

#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Attention {
    Gqa,
    Mla,
    Sparse,
}

impl Attention {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "gqa" => Some(Self::Gqa),
            "mla" => Some(Self::Mla),
            "sparse" | "dsa" => Some(Self::Sparse),
            _ => None,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Gqa => "gqa",
            Self::Mla => "mla",
            Self::Sparse => "sparse",
        }
    }
    /// J per new token for prefill attention over `c` context tokens.
    fn prefill_j(&self, layers: f64, c: f64) -> f64 {
        let mla_flops = 2.0 * Q_HEADS * (MLA_QK_DIM + MLA_V_DIM) * layers;
        let flops = match self {
            Self::Gqa => 4.0 * Q_HEADS * HEAD_DIM * layers * c,
            Self::Mla => mla_flops * c,
            Self::Sparse => {
                mla_flops * c.min(DSA_TOPK)
                    + 2.0 * DSA_INDEX_HEADS * DSA_INDEX_DIM * DSA_INDEX_FLOP_FACTOR * layers * c
            }
        };
        flops * J_PER_ATTN_FLOP
    }
    /// J per output token for decode attention (KV-cache reads) over `c` context tokens.
    fn decode_j(&self, layers: f64, c: f64) -> f64 {
        let bytes = match self {
            Self::Gqa => 2.0 * GQA_KV_HEADS * HEAD_DIM * KV_BYTES * layers * c,
            Self::Mla => MLA_LATENT * KV_BYTES * layers * c,
            Self::Sparse => {
                MLA_LATENT * KV_BYTES * layers * c.min(DSA_TOPK) + DSA_INDEX_KEY_BYTES * layers * c
            }
        };
        bytes * J_PER_KV_BYTE
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Serialize, Deserialize)]
struct Tok {
    input: u64,
    cache_write: u64,
    cache_read: u64,
    output: u64,
}

impl Tok {
    fn w(&self) -> f64 {
        (self.input + self.cache_write) as f64
    }
    fn is_zero(&self) -> bool {
        self.input == 0 && self.cache_write == 0 && self.cache_read == 0 && self.output == 0
    }
    fn max(&self, o: &Tok) -> Tok {
        Tok {
            input: self.input.max(o.input),
            cache_write: self.cache_write.max(o.cache_write),
            cache_read: self.cache_read.max(o.cache_read),
            output: self.output.max(o.output),
        }
    }
}

/// Energy in Wh for one call: [low, central, high].
fn energy(t: &Tok, opts: &Opts) -> [f64; 3] {
    let w = t.w();
    let o = t.output as f64;
    let r = t.cache_read as f64;
    let (cw, co) = if opts.simple_ctx { (r, r) } else { (r + w / 2.0, r + w + o / 2.0) };
    let mut e = [0.0; 3];
    for (i, a) in ARCHS.iter().enumerate() {
        let joules = w * (a.active_params * J_IN_PER_PARAM + opts.attention.prefill_j(a.layers, cw))
            + o * (a.active_params * J_OUT_PER_PARAM + opts.attention.decode_j(a.layers, co))
            + r * opts.kv;
        e[i] = joules / J_PER_WH;
    }
    e
}

// ---------- aggregation ----------

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct Agg {
    calls: u64,
    tokens: Tok,
    wh: [f64; 3],
}

impl Agg {
    fn add(&mut self, calls: u64, t: &Tok, e: &[f64; 3]) {
        self.calls += calls;
        self.tokens.input += t.input;
        self.tokens.cache_write += t.cache_write;
        self.tokens.cache_read += t.cache_read;
        self.tokens.output += t.output;
        for i in 0..3 {
            self.wh[i] += e[i];
        }
    }
    fn merge(&mut self, o: &Agg) {
        self.add(o.calls, &o.tokens, &o.wh);
    }
    fn avg_ctx_k(&self) -> f64 {
        if self.calls == 0 { 0.0 } else { self.tokens.cache_read as f64 / self.calls as f64 / 1000.0 }
    }
}

#[derive(Default, Debug, Serialize, Deserialize)]
struct Report {
    total: Agg,
    main: Agg,
    subagents: Agg,
    by_model: BTreeMap<String, Agg>,
    files: u64,
    subagent_files: u64,
    lines_with_usage: u64,
}

impl Report {
    fn record(&mut self, sub: bool, model: &str, calls: u64, t: &Tok, e: &[f64; 3]) {
        self.total.add(calls, t, e);
        if sub { self.subagents.add(calls, t, e) } else { self.main.add(calls, t, e) }
        self.by_model.entry(model.to_string()).or_default().add(calls, t, e);
    }
    fn merge(&mut self, o: &Report) {
        self.total.merge(&o.total);
        self.main.merge(&o.main);
        self.subagents.merge(&o.subagents);
        for (k, v) in &o.by_model {
            self.by_model.entry(k.clone()).or_default().merge(v);
        }
        self.files += o.files;
        self.subagent_files += o.subagent_files;
        self.lines_with_usage += o.lines_with_usage;
    }
}

// ---------- JSONL parsing ----------

#[derive(Deserialize)]
struct Line {
    #[serde(rename = "requestId")]
    request_id: Option<String>,
    #[serde(rename = "isSidechain")]
    is_sidechain: Option<bool>,
    message: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

struct Rec {
    id: Option<String>,
    sidechain: bool,
    model: String,
    tok: Tok,
}

fn parse_line(line: &str) -> Option<Rec> {
    // Cheap pre-filter: most lines (tool results, attachments) carry no usage block.
    if !line.contains("\"usage\"") {
        return None;
    }
    let l: Line = serde_json::from_str(line).ok()?;
    let msg = l.message?;
    let msg = msg.as_object()?;
    let u: Usage = serde_json::from_value(msg.get("usage")?.clone()).ok()?;
    let tok = Tok {
        input: u.input_tokens.unwrap_or(0),
        cache_write: u.cache_creation_input_tokens.unwrap_or(0),
        cache_read: u.cache_read_input_tokens.unwrap_or(0),
        output: u.output_tokens.unwrap_or(0),
    };
    let id = l
        .request_id
        .or_else(|| msg.get("id").and_then(|v| v.as_str()).map(String::from));
    let model = msg.get("model").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
    Some(Rec { id, sidechain: l.is_sidechain.unwrap_or(false), model, tok })
}

// ---------- dedup state ----------

/// Bounded memory of recently counted request IDs and the usage counted for each,
/// so repeated streaming lines are counted once (taking the max if a later line grew).
#[derive(Default, Serialize, Deserialize, Clone)]
struct Seen {
    ids: VecDeque<(String, Tok)>,
    #[serde(skip)]
    cap: usize,
}

impl Seen {
    fn with_cap(cap: usize) -> Self {
        Seen { ids: VecDeque::new(), cap }
    }
    fn get(&self, id: &str) -> Option<usize> {
        self.ids.iter().rposition(|(k, _)| k == id)
    }
}

struct Opts {
    simple_ctx: bool,
    kv: f64,
    attention: Attention,
}

/// Process records from a reader. `seen_global` is used in cumulative mode (unbounded);
/// `seen_ring` in running mode (bounded, persisted per file).
fn process<R: BufRead>(
    mut rd: R,
    sub_file: bool,
    opts: &Opts,
    rep: &mut Report,
    seen_global: Option<&mut HashMap<String, Tok>>,
    mut seen_ring: Option<&mut Seen>,
    last_id: &mut Option<String>,
) -> io::Result<u64> {
    let mut seen_global = seen_global;
    let mut consumed: u64 = 0;
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = rd.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        if !buf.ends_with('\n') {
            // Partial line still being written: leave it for next time.
            break;
        }
        consumed += n as u64;
        let Some(rec) = parse_line(&buf) else { continue };
        rep.lines_with_usage += 1;
        if rec.tok.is_zero() {
            continue; // synthetic / placeholder records
        }
        let sub = sub_file || rec.sidechain;
        let e_new = energy(&rec.tok, opts);
        let Some(id) = rec.id.clone() else {
            rep.record(sub, &rec.model, 1, &rec.tok, &e_new);
            continue;
        };
        *last_id = Some(id.clone());

        // Look up previous count for this id.
        let prev: Option<Tok> = if let Some(g) = seen_global.as_deref_mut() {
            g.get(&id).copied()
        } else if let Some(r) = seen_ring.as_deref_mut() {
            r.get(&id).map(|i| r.ids[i].1)
        } else {
            None
        };

        match prev {
            None => {
                rep.record(sub, &rec.model, 1, &rec.tok, &e_new);
                if let Some(g) = seen_global.as_deref_mut() {
                    g.insert(id, rec.tok);
                } else if let Some(r) = seen_ring.as_deref_mut() {
                    r.ids.push_back((id, rec.tok));
                    while r.ids.len() > r.cap {
                        r.ids.pop_front();
                    }
                }
            }
            Some(old) => {
                let merged = old.max(&rec.tok);
                if merged != old {
                    let e_old = energy(&old, opts);
                    let e_m = energy(&merged, opts);
                    let dt = Tok {
                        input: merged.input - old.input,
                        cache_write: merged.cache_write - old.cache_write,
                        cache_read: merged.cache_read - old.cache_read,
                        output: merged.output - old.output,
                    };
                    let de = [e_m[0] - e_old[0], e_m[1] - e_old[1], e_m[2] - e_old[2]];
                    rep.record(sub, &rec.model, 0, &dt, &de);
                    if let Some(g) = seen_global.as_deref_mut() {
                        g.insert(id, merged);
                    } else if let Some(r) = seen_ring.as_deref_mut() {
                        if let Some(i) = r.get(&id) {
                            r.ids[i].1 = merged;
                        }
                    }
                }
            }
        }
    }
    Ok(consumed)
}

// ---------- file discovery ----------

fn is_subagent_path(p: &Path) -> bool {
    p.components().any(|c| c.as_os_str() == "subagents")
        || p.file_name().and_then(|n| n.to_str()).map_or(false, |n| n.starts_with("agent-"))
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for ent in rd.flatten() {
        let p = ent.path();
        let Ok(ft) = ent.file_type() else { continue };
        if ft.is_dir() {
            walk(&p, out);
        } else if ft.is_file() && p.extension().map_or(false, |e| e == "jsonl") {
            out.push(p);
        }
    }
}

/// Expand inputs: a directory is walked recursively; a session .jsonl also pulls in
/// its sibling `<session>/subagents/*.jsonl` unless --no-subagents.
fn collect(inputs: &[PathBuf], subagents: bool) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_dir() {
            walk(p, &mut out);
        } else if p.is_file() {
            out.push(p.clone());
            if subagents {
                let side = p.with_extension("");
                if side.is_dir() {
                    walk(&side, &mut out);
                }
            }
        }
    }
    if !subagents {
        out.retain(|p| !is_subagent_path(p));
    }
    out.sort();
    out.dedup();
    out
}

fn default_root() -> PathBuf {
    if let Ok(d) = std::env::var("CLAUDE_CONFIG_DIR") {
        return PathBuf::from(d).join("projects");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".claude").join("projects")
}

// ---------- running-mode cursor ----------

#[derive(Default, Serialize, Deserialize)]
struct FileCursor {
    offset: u64,
    last_request_id: Option<String>,
    seen: Seen,
}

#[derive(Default, Serialize, Deserialize)]
struct Cursor {
    version: u32,
    #[serde(default)]
    attention: Option<Attention>,
    files: BTreeMap<String, FileCursor>,
    totals: Report,
}

const RING: usize = 256;

fn load_cursor(p: &Path) -> Cursor {
    let mut c: Cursor = std::fs::read(p)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    c.version = 1;
    for fc in c.files.values_mut() {
        fc.seen.cap = RING;
    }
    c
}

fn save_cursor(p: &Path, c: &Cursor) -> io::Result<()> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = p.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec(c)?)?;
    std::fs::rename(tmp, p) // atomic replace, safe if several hooks fire at once
}

/// Find the byte offset just after the last line containing `id`.
fn recover_offset(path: &Path, id: &str) -> io::Result<Option<u64>> {
    let mut rd = BufReader::new(File::open(path)?);
    let mut pos = 0u64;
    let mut found = None;
    let mut buf = String::new();
    let needle = format!("\"{}\"", id);
    loop {
        buf.clear();
        let n = rd.read_line(&mut buf)?;
        if n == 0 || !buf.ends_with('\n') {
            break;
        }
        pos += n as u64;
        if buf.contains(&needle) {
            found = Some(pos);
        }
    }
    Ok(found)
}

/// Check the line ending at `offset` still contains the stored request id
/// (i.e. the file wasn't rewritten under us). Reads only a small window.
fn offset_still_valid(path: &Path, offset: u64, id: &Option<String>, len: u64) -> bool {
    if offset > len {
        return false;
    }
    let Some(id) = id else { return true };
    if offset == 0 {
        return true;
    }
    let Ok(mut f) = File::open(path) else { return false };
    // Look back up to 2 MB for the previous line; transcript lines can be large.
    let win = offset.min(2 * 1024 * 1024);
    if f.seek(SeekFrom::Start(offset - win)).is_err() {
        return false;
    }
    let mut v = vec![0u8; win as usize];
    if f.read_exact(&mut v).is_err() {
        return false;
    }
    // The cursor sits after the last processed record, which may be followed by
    // non-usage lines; accept if the id appears anywhere in the window.
    let s = String::from_utf8_lossy(&v);
    s.contains(id.as_str())
}

// ---------- output ----------

fn fmt_tok(n: u64) -> String {
    let f = n as f64;
    if f >= 1e9 { format!("{:.2}B", f / 1e9) }
    else if f >= 1e6 { format!("{:.2}M", f / 1e6) }
    else if f >= 1e3 { format!("{:.1}k", f / 1e3) }
    else { format!("{}", n) }
}

fn fmt_wh(wh: f64) -> String {
    if wh.abs() >= 1000.0 { format!("{:.2} kWh", wh / 1000.0) } else { format!("{:.1} Wh", wh) }
}

fn fmt_range(a: &Agg) -> String {
    format!("{} central ({}–{})", fmt_wh(a.wh[1]), fmt_wh(a.wh[0]), fmt_wh(a.wh[2]))
}

fn print_text(title: &str, r: &Report, att: Attention, out: &mut impl Write) -> io::Result<()> {
    let t = &r.total;
    writeln!(out, "{title}")?;
    writeln!(out, "Model: {} attention · low/central/high = 37B/122B/405B active parameters", att.name())?;
    writeln!(out, "Files: {} ({} subagent) · calls: {} · usage lines: {}", r.files, r.subagent_files, t.calls, r.lines_with_usage)?;
    writeln!(
        out,
        "Tokens: cache reads {} · cache writes {} · uncached input {} · output {}",
        fmt_tok(t.tokens.cache_read), fmt_tok(t.tokens.cache_write), fmt_tok(t.tokens.input), fmt_tok(t.tokens.output)
    )?;
    writeln!(out, "Average context per call: {:.0}k", t.avg_ctx_k())?;
    writeln!(out, "Energy: {}", fmt_range(t))?;
    if r.subagents.calls > 0 || r.subagent_files > 0 {
        writeln!(out, "  main:      {} · {} calls", fmt_range(&r.main), r.main.calls)?;
        writeln!(out, "  subagents: {} · {} calls", fmt_range(&r.subagents), r.subagents.calls)?;
    }
    if r.by_model.len() > 1 {
        writeln!(out, "By model:")?;
        for (m, a) in &r.by_model {
            writeln!(out, "  {m}: {} · {} calls · avg ctx {:.0}k", fmt_range(a), a.calls, a.avg_ctx_k())?;
        }
    }
    Ok(())
}

fn print_line(delta: &Report, total: &Report, out: &mut impl Write) -> io::Result<()> {
    writeln!(
        out,
        "⚡ +{} ({} calls) · total {} ({}–{})",
        fmt_wh(delta.total.wh[1]),
        delta.total.calls,
        fmt_wh(total.total.wh[1]),
        fmt_wh(total.total.wh[0]),
        fmt_wh(total.total.wh[2])
    )
}

// ---------- CLI ----------

const HELP: &str = "cc-energy — energy estimates for Claude Code JSONL transcripts

USAGE:
  cc-energy [cumulative|running] [OPTIONS] [PATH...]

PATH may be a directory (searched recursively) or a session .jsonl
(its <session>/subagents/*.jsonl are included). Default: ~/.claude/projects

MODES:
  cumulative   Scan everything and print totals (default)
  running      Process only new lines since last run; keep a running total
               in the cursor file. Cheap enough to call every turn.

OPTIONS:
  --cursor FILE      Cursor file for running mode
                     (default: ~/.claude/cc-energy-cursor.json)
  --reset            Clear the cursor before running
  --hook             Read Claude Code hook/statusline JSON on stdin and use
                     its transcript_path (plus that session's subagents)
  --json             Machine-readable output
  --line             One-line output (good for a status line)
  --attention TYPE   gqa (default), mla or sparse — see MODEL below
  --kv J             KV-storage adder, joules per cache-read token (default 0; try 0–0.018)
  --simple-context   Use C = cache reads only (no W/2, O/2 correction)
  --no-subagents     Exclude subagent transcripts
  -h, --help         This help

MODEL (inputs in tokens; energy in joules, reported as Wh = J/3600), per API call:
  E = W*(N*e_in + prefill(C)) + O*(N*e_out + decode(C)) + R*kv
  W = uncached input + cache writes, O = output, R = cache reads,
  C = context in tokens (R + W/2 for W, R + W + O/2 for O; R with --simple-context).
  Fixed efficiency: e_in = 7.8e-12 J and e_out = 3.9e-11 J per active parameter;
  attention maths 2 pJ/FLOP; KV-cache reads 230 pJ/byte (facility level).
  low / central / high = model architectures (128 query heads of size 128):
    low      37B active, 61 layers   (DeepSeek-V3 scale)
    central  122B active, 96 layers  (hypothetical midpoint)
    high     405B active, 126 layers (Llama-3.1-405B scale)
  Attention types:
    gqa      8 KV heads, bf16 KV cache (~4 KB per token per layer)
    mla      latent KV of 576 values (~1.2 KB per token per layer)
    sparse   mla over the top 2,048 tokens plus a light fp8 indexer";

fn main() {
    if let Err(e) = run() {
        eprintln!("cc-energy: {e}");
        std::process::exit(1);
    }
}

fn run() -> io::Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let mut mode = "cumulative".to_string();
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut cursor_path: Option<PathBuf> = None;
    let (mut reset, mut hook, mut json, mut line, mut simple, mut subagents) =
        (false, false, false, false, false, true);
    let mut kv = 0.0f64;
    let mut attention = Attention::Gqa;

    while let Some(a) = args.next() {
        match a.as_str() {
            "cumulative" | "running" => mode = a,
            "--cursor" => cursor_path = args.next().map(PathBuf::from),
            "--reset" => reset = true,
            "--hook" => hook = true,
            "--json" => json = true,
            "--line" => line = true,
            "--simple-context" => simple = true,
            "--no-subagents" => subagents = false,
            "--attention" => {
                attention = args.next().as_deref().and_then(Attention::parse).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "--attention must be gqa, mla or sparse")
                })?
            }
            "--kv" => {
                kv = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--kv needs a number"))?
            }
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(());
            }
            s if s.starts_with('-') => {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unknown option {s} (see --help)")))
            }
            _ => paths.push(PathBuf::from(a)),
        }
    }

    if hook {
        let mut s = String::new();
        io::stdin().read_to_string(&mut s)?;
        let v: serde_json::Value = serde_json::from_str(&s)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("hook stdin: {e}")))?;
        if let Some(tp) = v.get("transcript_path").and_then(|x| x.as_str()) {
            paths.push(PathBuf::from(tp));
        }
    }
    if paths.is_empty() {
        paths.push(default_root());
    }

    let files = collect(&paths, subagents);
    let opts = Opts { simple_ctx: simple, kv, attention };
    let stdout = io::stdout();
    let mut out = stdout.lock();

    if mode == "cumulative" {
        let mut rep = Report::default();
        let mut seen: HashMap<String, Tok> = HashMap::new();
        for f in &files {
            let sub = is_subagent_path(f);
            rep.files += 1;
            if sub { rep.subagent_files += 1; }
            let Ok(fh) = File::open(f) else { continue };
            let mut last = None;
            process(BufReader::with_capacity(1 << 20, fh), sub, &opts, &mut rep, Some(&mut seen), None, &mut last)?;
        }
        if json {
            serde_json::to_writer_pretty(&mut out, &rep)?;
            writeln!(out)?;
        } else if line {
            writeln!(out, "⚡ {} ({}–{}) · {} calls", fmt_wh(rep.total.wh[1]), fmt_wh(rep.total.wh[0]), fmt_wh(rep.total.wh[2]), rep.total.calls)?;
        } else {
            print_text("Claude Code energy estimate (cumulative)", &rep, attention, &mut out)?;
        }
        return Ok(());
    }

    // running mode
    let cpath = cursor_path.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(home).join(".claude").join("cc-energy-cursor.json")
    });
    let mut cur = if reset { Cursor { version: 1, ..Default::default() } } else { load_cursor(&cpath) };
    match cur.attention {
        Some(a) if a != attention && !cur.files.is_empty() => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "cursor {} holds totals for --attention {}; use --reset or a different --cursor",
                    cpath.display(),
                    a.name()
                ),
            ))
        }
        _ => cur.attention = Some(attention),
    }
    let mut delta = Report::default();

    for f in &files {
        let key = f.to_string_lossy().to_string();
        let sub = is_subagent_path(f);
        let len = match std::fs::metadata(f) { Ok(m) => m.len(), Err(_) => continue };
        let new_file = !cur.files.contains_key(&key);
        let fc = cur.files.entry(key).or_insert_with(|| FileCursor { seen: Seen::with_cap(RING), ..Default::default() });
        if fc.seen.cap == 0 { fc.seen.cap = RING; }
        if new_file {
            delta.files += 1;
            if sub { delta.subagent_files += 1; }
        }
        if fc.offset == len {
            continue; // nothing new: the common, near-free case
        }
        // Validate the fast-path offset; if the file was rewritten, recover via request id.
        if !offset_still_valid(f, fc.offset, &fc.last_request_id, len) {
            fc.offset = match &fc.last_request_id {
                Some(id) => recover_offset(f, id)?.unwrap_or(0),
                None => 0,
            };
        }
        let mut fh = File::open(f)?;
        fh.seek(SeekFrom::Start(fc.offset))?;
        let mut last = fc.last_request_id.clone();
        let consumed = process(BufReader::with_capacity(1 << 20, fh), sub, &opts, &mut delta, None, Some(&mut fc.seen), &mut last)?;
        fc.offset += consumed;
        fc.last_request_id = last;
    }

    cur.totals.merge(&delta);
    save_cursor(&cpath, &cur)?;

    if json {
        #[derive(Serialize)]
        struct J<'a> { delta: &'a Report, total: &'a Report }
        serde_json::to_writer_pretty(&mut out, &J { delta: &delta, total: &cur.totals })?;
        writeln!(out)?;
    } else if line {
        print_line(&delta, &cur.totals, &mut out)?;
    } else {
        print_text("Since last run", &delta, attention, &mut out)?;
        writeln!(out)?;
        print_text("Running total", &cur.totals, attention, &mut out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(input: u64, cw: u64, cr: u64, out: u64) -> Tok {
        Tok { input, cache_write: cw, cache_read: cr, output: out }
    }

    fn opts(attention: Attention, simple_ctx: bool) -> Opts {
        Opts { simple_ctx, kv: 0.0, attention }
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-30)
    }

    #[test]
    fn output_token_at_zero_context_is_weights_only() {
        // One output token, no context (simple context => C = R = 0).
        let e = energy(&t(0, 0, 0, 1), &opts(Attention::Gqa, true));
        assert!(close(e[0] * J_PER_WH, 1.44, 1e-12));
        assert!(close(e[1] * J_PER_WH, 1.44 * 122.0 / 37.0, 1e-12));
        assert!(close(e[2] * J_PER_WH, 1.44 * 405.0 / 37.0, 1e-12));
    }

    #[test]
    fn matches_python_prototype() {
        // Reference values from an independent Python implementation of the model
        // (W = 5000, O = 400, R = 200,000, so Cw = 202,500 and Co = 205,200; Wh for low/central/high).
        let cases = [
            (Attention::Gqa, [4.118948864, 7.44745519, 13.481001154]),
            (Attention::Mla, [3.739386368, 6.850110934, 12.696986818]),
            (Attention::Sparse, [0.914139093, 2.403820141, 6.861230151]),
        ];
        for (att, expected) in cases {
            let e = energy(&t(0, 5000, 200_000, 400), &opts(att, false));
            for i in 0..3 {
                assert!(close(e[i], expected[i], 1e-8), "{att:?}[{i}]: {} vs {}", e[i], expected[i]);
            }
        }
    }

    #[test]
    fn per_pair_costs_for_gqa_low() {
        // 4 * 128 * 128 * 61 FLOPs * 2 pJ; 2 * 8 * 128 * 2 bytes * 61 * 230 pJ.
        assert!(close(Attention::Gqa.prefill_j(61.0, 1.0), 7.995392e-6, 1e-9));
        assert!(close(Attention::Gqa.decode_j(61.0, 1.0), 5.7466880e-5, 1e-9));
    }

    #[test]
    fn sparse_attention_flattens_beyond_topk() {
        let a = Attention::Sparse;
        // Beyond 2,048 tokens only the indexer grows: slope = indexer cost per pair.
        let slope_prefill = a.prefill_j(61.0, 200_000.0) - a.prefill_j(61.0, 100_000.0);
        let indexer = 2.0 * 64.0 * 128.0 * 0.5 * 61.0 * 100_000.0 * J_PER_ATTN_FLOP;
        assert!(close(slope_prefill, indexer, 1e-9));
        assert!(a.prefill_j(61.0, 200_000.0) < Attention::Mla.prefill_j(61.0, 200_000.0) / 5.0);
    }

    #[test]
    fn estimates_are_monotonic_across_scenarios() {
        for att in [Attention::Gqa, Attention::Mla, Attention::Sparse] {
            for &(w, r, o) in &[(1000, 0, 100), (5000, 50_000, 400), (2000, 300_000, 900)] {
                let e = energy(&t(0, w, r, o), &opts(att, false));
                assert!(e[0] < e[1] && e[1] < e[2], "{att:?} {e:?}");
            }
        }
    }

    #[test]
    fn kv_adder_is_per_cache_read_token() {
        let mut o = opts(Attention::Gqa, true);
        let base = energy(&t(0, 0, 1_000_000, 0), &o);
        o.kv = 0.018;
        let kv = energy(&t(0, 0, 1_000_000, 0), &o);
        assert!(close(kv[1] - base[1], 5.0, 1e-12)); // 1M tokens * 0.018 J = 5 Wh
    }

    fn line(id: &str, cw: u64, cr: u64, out: u64) -> String {
        format!(
            "{{\"type\":\"assistant\",\"requestId\":\"{id}\",\"message\":{{\"model\":\"m\",\"usage\":{{\"input_tokens\":1,\"cache_creation_input_tokens\":{cw},\"cache_read_input_tokens\":{cr},\"output_tokens\":{out}}}}}}}\n"
        )
    }

    #[test]
    fn dedups_streamed_lines_and_takes_max() {
        let data = line("r1", 100, 1000, 10) + &line("r1", 100, 1000, 50) + &line("r2", 10, 2000, 5);
        let o = opts(Attention::Gqa, false);
        let mut rep = Report::default();
        let mut seen = HashMap::new();
        let mut last = None;
        process(data.as_bytes(), false, &o, &mut rep, Some(&mut seen), None, &mut last).unwrap();
        assert_eq!(rep.total.calls, 2);
        assert_eq!(rep.total.tokens.output, 55);
        assert_eq!(rep.total.tokens.cache_read, 3000);
        assert_eq!(last.as_deref(), Some("r2"));
    }

    #[test]
    fn partial_final_line_is_not_consumed() {
        let full = line("r1", 100, 1000, 10);
        let data = full.clone() + "{\"type\":\"assistant\",\"requestId\":\"r2\"";
        let o = opts(Attention::Gqa, false);
        let mut rep = Report::default();
        let mut ring = Seen::with_cap(8);
        let mut last = None;
        let consumed = process(data.as_bytes(), false, &o, &mut rep, None, Some(&mut ring), &mut last).unwrap();
        assert_eq!(consumed, full.len() as u64);
        assert_eq!(rep.total.calls, 1);
    }

    #[test]
    fn sidechain_records_count_as_subagents() {
        let data = "{\"type\":\"assistant\",\"requestId\":\"s1\",\"isSidechain\":true,\"message\":{\"usage\":{\"input_tokens\":1,\"cache_creation_input_tokens\":10,\"cache_read_input_tokens\":100,\"output_tokens\":5}}}\n";
        let o = opts(Attention::Gqa, false);
        let mut rep = Report::default();
        let mut seen = HashMap::new();
        let mut last = None;
        process(data.as_bytes(), false, &o, &mut rep, Some(&mut seen), None, &mut last).unwrap();
        assert_eq!(rep.subagents.calls, 1);
        assert_eq!(rep.main.calls, 0);
    }
}
