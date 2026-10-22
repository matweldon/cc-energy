# Methodology and evidence

This note explains where cc-energy's model and coefficients come from, how they were
checked against independent evidence, and what remains uncertain. It reflects public
information as of October 2026.

**Bottom line.** For agentic coding at long context (100–250k tokens per call), the central
estimate is roughly 5–15 Wh per model call and 1–3 kWh per day for a heavy user. The low and
high cases span about 5× either way. Those figures come from a bottom-up model. Three other
lines of evidence bracket them: published measurements, fleet-level (top-down) allocation,
and an economic ceiling from subscription pricing. All of them are consistent with the
central case, and none points to something well above the high case. No frontier lab
publishes per-token energy figures, so every number here is an estimate.

---

## 1. The model

Each API call is charged:

```
E = W·w_in·(1 + C_w/K_in) + O·w_out·(1 + C_o/K_out) + R·kv
```

- **W:** fresh input tokens processed in the call (uncached input plus cache writes).
- **O:** output tokens.
- **R:** cache-read tokens (context already held in the KV cache).
- **C_w = R + W/2** and **C_o = R + W + O/2:** the average context each new or output token
  attends to. `--simple-context` uses C = R.
- **w_in, w_out:** joules per token at zero context (weights/matmul cost, plus host,
  networking and facility overheads).
- **K_in, K_out:** "crossover" context lengths. A token that sees K tokens of context costs
  twice its base rate; one that sees 10K costs 11×.
- **kv:** optional joules per cache-read token for storing and reloading the KV cache.
  Default 0.

Energy is computed in joules and reported in Wh (J ÷ 3600).

Coefficients:

- **Low:** w_in 0.108 J, K_in 60,000; w_out 0.72 J, K_out 133,000
- **Central:** w_in 0.288 J, K_in 26,700; w_out 1.44 J, K_out 57,100
- **High:** w_in 0.612 J, K_in 28,300; w_out 3.96 J, K_out 78,600
- **kv:** 0 by default, 0.018 J at the top of the plausible range

### Why this form

With KV caching, previously processed context is not recomputed. A call's cost has three
parts:

1. **Weights.** Every new token passes through the model weights. Output tokens cost more
   per token than input because decoding is memory-bandwidth bound and generates one token
   at a time, while prefill processes tokens in parallel.
2. **Prefill attention.** Each new input token attends to all C tokens of context, so the
   cost is proportional to W·C. On a cache miss, the whole context reappears as W, and the
   W/2 term charges for rebuilding it, which is the genuinely quadratic case.
3. **Decode attention.** Generating each output token re-reads the KV cache for all C
   context tokens, so the cost is proportional to O·C.

Two consequences follow:

- **Cache reads have no energy cost of their own in this model** (apart from the optional
  kv term). They only make new and output tokens more expensive. A flat "cache reads cost
  10% of input" factor, priced from API rates, overstates cheap turns (small tool result,
  short reply) and understates heavy ones.
- **The quadratic shows up at session level.** Context grows with every turn, so a
  session's total cost grows roughly with turns². Summed cache-read tokens in the logs
  already capture this, because they equal the sum of context over all calls.

Within a single call the cost is linear in C, unless the context is being built from
scratch.

Sparse or hybrid attention changes the picture. DeepSeek V3.2's attention scales close to
linearly with sequence length, and Kimi Linear and Qwen3-Next experiment in the same
direction ([Muxup](https://muxup.com/2026q1/per-query-energy-consumption-of-llms);
[DeepSeek V3.2 paper](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)).
That would raise K substantially, which is what the low scenario allows for. The attention
mechanisms of current Claude models are not public.

---

## 2. Where the coefficients come from (micro evidence)

### Weights terms: w_in and w_out

- **InferenceMAX benchmarks,** analysed by Muxup
  ([post](https://muxup.com/2026q1/per-query-energy-consumption-of-llms);
  [benchmark](https://inferencemax.semianalysis.com/)). For DeepSeek-R1-0528 (671B total,
  37B active parameters) at fp4 on B200, one 8k-in/1k-out query used about 0.93 Wh and one
  1k-in/8k-out query about 2.73 Wh. The benchmark's actual sequence lengths average 0.9× the
  nominal ones. Solving the two equations then gives about **80 Wh per million input
  tokens** and **360 Wh per million output tokens** (0.29 J and 1.3 J per token), an
  output-to-input ratio of about 4.4. These are best-case benchmark configurations; Muxup
  suggests a real provider might achieve about half that efficiency. The fp8 results are
  roughly 2–4× higher, and GB200 NVL72 rack-scale setups are lower.
- **Microsoft Research (Joule, 2026)**
  ([summary](https://www.microsoft.com/en-us/research/?p=1170806)). Optimised frontier-scale
  inference is put at a median of 0.31 Wh per query (interquartile range 0.16–0.60) for
  about 300 output tokens. A query with 5,000 output tokens rises to 3.91 Wh. Long reasoning
  and agentic queries cost more than ten times as much because of extra tokens and lower
  serving concurrency.
- **Google, Gemini Apps** ([paper](https://arxiv.org/pdf/2508.15734);
  [MIT Technology Review](https://www.technologyreview.com/2025/08/21/1122288/google.gemini.ai.energy/)).
  The median text prompt uses 0.24 Wh: 0.14 accelerator, 0.06 host CPU/DRAM, 0.02 idle
  machines, 0.02 data-centre overhead
  ([breakdown](https://howaiworks.ai/glossary/ai-energy-consumption)). Accelerators are
  only about 58% of the total, so non-accelerator energy belongs in the coefficients.
  Google reports a 33× fall in energy per median prompt over 12 months.
- **OpenAI and Epoch AI.** Sam Altman put an average ChatGPT query at 0.34 Wh
  ([blog](https://blog.samaltman.com/the-gentle-singularity)). Epoch AI estimated about
  0.3 Wh for GPT-4o and about 2.5 Wh for a 10k-token-input query
  ([analysis](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use)).

The central w_in (0.29 J) matches the benchmark-derived input rate. The central w_out
(1.44 J) sits a little above the benchmark's output rate, allowing for production
inefficiency and frontier models that are probably larger than DeepSeek-R1. The 5:1 ratio
also matches Anthropic's input-to-output price ratio
([pricing](https://platform.claude.com/docs/en/about-claude/pricing)), although prices are
not costs.

### Context terms: K_in and K_out

These are first-principles estimates for a frontier-scale model with full attention.

- **Prefill attention:** about 1e-5 J per (new token × context token) pair. This is the
  attention FLOPs per pair for a large model at realistic utilisation, including overheads.
  Central K_in = 0.288 ÷ 1.08e-5 ≈ 26,700 tokens.
- **Decode attention:** about 2.5e-5 J per (output token × context token) pair. This
  assumes about 100 KB of KV cache per token read at roughly 250 pJ per byte, all-in.
  Central K_out = 1.44 ÷ 2.52e-5 ≈ 57,100 tokens.
- **Low case:** compressed KV and sparse attention push K up. **High case:** larger models
  with bigger KV caches push the per-pair cost up.

These are the least certain coefficients, and at long context they dominate the result. At
200k context, central input tokens cost about 8.5× their base rate and output tokens about
4.5×.

### Overheads and batching

- **Non-GPU energy is large.** A full-system profiling study on Blackwell GPUs found that
  GPU-only telemetry missed 41–45% of system energy
  ([Luo et al. 2026](https://arxiv.org/pdf/2609.29707)). The coefficients here are meant to
  be facility-level, including PUE (Google reports a fleet PUE of 1.09:
  [Paperjam](https://paperjam.lu/article/google-a-consomme-37-delectricite-en-plus-en-2025)).
- **Batching dominates efficiency.** In the same study, a sequential, single-user agent
  used 63× more system energy per output token than saturated serving. That was a single
  local server with no batching. Production serving batches many users, but agent traffic
  can lower concurrency (as the Microsoft paper also notes). This is one reason real-world
  figures may exceed the central case by 1–2×.
- **Per-turn energy grows with context.** The same study found per-turn energy rising over
  an agent trajectory while power stayed flat. That is consistent with longer prefill as
  context accumulates, and so with the C terms.
- **Extended thinking changes volume, not efficiency.** Per-token energy was within 1%
  between thinking modes; total energy rose with output volume.

---

## 3. Checks against real usage

Two people have published token logs from their own Claude Code use, with energy estimates
from simpler per-token methods.

- **Simon Couch, January 2026**
  ([post](https://simonpcouch.com/blog/2026-01-20-cc-impact/)). His median session had 24
  calls and 592k tokens, about 23k of context per call. He estimated 41 Wh using flat
  factors derived from Epoch AI with a 5:1 ratio (390 / 1,950 / 490 / 39 Wh per million
  input / output / cache-write / cache-read tokens). This model's central case gives about
  6 Wh for that session. Couch's factors were deliberately pessimistic, scaled from a
  100k-token Epoch query, and apply long-context costs to short-context calls. At long
  context the two approaches converge.
- **Zeke Hausfather, August 2026**
  ([post](https://skepticalscience.com/print.php?n=6600)). Over 8 weeks, 1,138 prompts led
  to over 14,000 calls and 3.2 billion tokens. 96% were cache reads and about 0.4% output,
  with roughly 220k context per call. His central estimate, using Watershed's activity-tier
  factors (Bistline et al. 2026), was 3.0 kWh/day (range 1.2–5.9). This model's central
  case gives about 2 kWh/day (range 0.4–6). His range is consistent across three methods,
  and the overlap with this model is good.

Both posts make the same point: published per-prompt figures (0.24–0.34 Wh) describe chat,
not agents. An agentic prompt costs hundreds of times more, because one prompt triggers
around ten calls, each re-reading a long context.

---

## 4. Top-down (macro) checks

Macro allocation estimates each user's share of everything that exists to serve them,
including idle and ramping capacity, free tiers, internal use and redundancy. It answers a
different question from the marginal bottom-up model and is expected to come out higher.

### Anthropic: fleet energy per dollar of revenue

- **Compute:** reported at about 1.4 GW at the end of 2025 and an expected 5 GW by the end
  of 2026. These figures come from investor reporting, not Anthropic
  ([FourWeekMBA](https://fourweekmba.com/ai-anthropic-gigawatt-threshold-compute-unit-of-account/);
  [Crypto Briefing](https://cryptobriefing.com/anthropic-10-gw-compute-capacity-2027/)).
  A further 3.5 GW of TPU capacity is contracted from 2027
  ([Anthropic](https://www.anthropic.com/news/google-broadcom-partnership-compute)).
- **Revenue:** the run rate was about $9bn at the end of 2025, $30bn+ in April 2026 and
  $47bn in May 2026 (official figures), with a reported $65bn by the end of July
  ([Unite.AI](https://www.unite.ai/?p=453467);
  [Bloomberg via BGOV](https://news.bgov.com/financial-accounting/anthropic-revenue-run-rate-surpasses-65-billion-ahead-of-ipo)).
- **Assumptions:** about 3.2 GW online, 70% average draw, 55% of it serving customer
  inference, PUE 1.1, and an $85bn run rate. That gives about 140 Wh per dollar of
  list-price revenue, with a range of roughly 55–300. Couch's January cross-check implied
  about 75 Wh/$.
- **Applied to the usage logs:** Couch's median session (about $0.50 at list price) comes
  to about 70 Wh (30–150). Hausfather's heavy day (about $46) comes to about 6 kWh
  (2.5–14). Macro sits about 3× above the central micro estimate for heavy long-context
  use, and further above it for light sessions, as expected.

### Google: a loose upper bound

Google's data centres used 42.4 TWh in 2025
([Paperjam](https://paperjam.lu/article/google-a-consomme-37-delectricite-en-plus-en-2025);
[S&P Global](https://www.spglobal.com/energy/en/technology-ai-research-insights/blogs/googles-search-for-power-sustainability-is-yielding-results-but-ai-slows-progress)),
and Google processed over 3.2 quadrillion tokens a month in May 2026
([Google](https://blog.google/innovation-and-ai/sundar-pichai-io-2026/)). If all of that
electricity went to tokens, it would be about 1,100 Wh per million tokens blended. AI is
only part of Google's load, the token mix is mostly Flash-class input, and the two figures
are from different years. A realistic blended figure is a few hundred Wh per million
tokens, consistent with the micro ranges.

---

## 5. An economic ceiling from subscription pricing

SemiAnalysis measured how many tokens each plan allows under an agentic workload
([article](https://newsletter.semianalysis.com/p/anthropic-subscriptions-offer-5x)). Claude
Pro ($20 a month) allows about 2.9 billion Opus 5.5 tokens, or 144M per dollar: 0.4%
input, 96.6% cached input, 2.6% cache writes, 0.3% output. Max 5× and Max 20× give 14.0B
and 28.6B.

**The assumption:** Anthropic does not sell tokens for less than the electricity they use.

- **The ceiling.** A fully used $20 plan can then use at most $20 ÷ electricity price. At
  an assumed 8.5¢/kWh (range 5–15¢) that is about **235 kWh a month** (133–400), or
  roughly 80 Wh per million tokens blended (0.3 J per token).
- **The model, same month.** W = 87M, O = 8.7M, R = 2,801M. At 100k / 150k / 200k / 250k
  context per call:
  - central: 43 / 59 / 75 / 91 kWh
  - high: 89 / 121 / 153 / 185 kWh
  - low: 10–19 kWh
  - kv at 0.018 J adds up to 14 kWh
- **What that implies for the parameters.** The central case sits 3–5× below the ceiling.
  The high case reaches 50–80% of it at typical context, and only exceeds it with dear
  power (about 12¢) and context above about 210k. The ceiling does not move the low or
  central estimates; it shows the high case is near the economic upper limit and rules out
  anything well above it.

### Is the assumption sound?

The common claim that labs "subsidise tokens" mixes up three things:

1. **Below electricity cost.** Very unlikely. Electricity for a maxed-out $20 plan is about
   $5 central, or $10 high; for typical use it is perhaps $1–2.
2. **Below running cost of compute,** including chip depreciation. Plausible for users who
   max out their plan, unproven on average.
3. **Below total cost including training.** That is a company not yet recovering R&D, not
   token-level subsidy.

The evidence:

- **SemiAnalysis's own numbers.** Their −369% margin for a maxed-out Opus user assumes cost
  is 8% of API list price (a modelled 92% API gross margin) and 100% use of the allowance.
  At 20% average use, the same model gives **+6%**.
- **Reported company margins are leaks and conflict.** The Information reported a projected
  2025 gross margin cut to about 40% (revenue minus inference costs), with inference 23%
  over budget ([WebProNews](https://www.webpronews.com/anthropics-margin-squeeze-inference-costs-bite-as-revenue-surges/)).
  An unsourced −94% also circulates
  ([BuildMVPFast](https://www.buildmvpfast.com/blog/anthropic-ipo-s1-sec-filing-2026)).
  A confidential draft S-1 was submitted in June 2026; the public S-1 will give the first
  audited figures. For a sceptical reading, see
  [Where's Your Ed At](https://www.wheresyoured.at/anthropics-profitability-swindle/).
- **Behaviour points to opportunity cost.** Anthropic added weekly caps for round-the-clock
  power users in 2025
  ([TechCrunch](https://techcrunch.com/2025/07/28/anthropic-unveils-new-rate-limits-to-curb-claude-code-power-users/)).
  In June 2026 it moved programmatic use onto a separate credit pool billed at API rates
  beyond the fee
  ([The Register](https://www.theregister.com/ai-ml/2026/05/14/anthropic-tosses-agents-into-the-api-billing-pool/5240748)).
  These changes fit a compute-constrained company pushing heavy use towards
  higher-paying API demand, more than they prove losses.

**Caveat.** Plans are priced on average use, so a single maxed-out subscriber could cost
more than the fee without breaking the assumption on average. Treat the ceiling as a soft
bound on typical use, not a hard per-token limit. Even so, electricity is a small fraction
of $20 at both maximum and typical use.

---

## 6. Interpreting the output

- **What the figures are.** cc-energy reports **marginal, bottom-up, facility-level**
  estimates.
- **Real-world serving.** Multiply by about 1–2 for production serving below benchmark
  efficiency (lower batch sizes for agent traffic, imperfect utilisation).
- **Attributional share of fleet energy.** Multiply the central figure by about 3, and more
  for light, short-context use.
- **Context dominates at long context.** Above roughly 27–57k tokens, the context terms
  exceed the base terms. Compaction and subagents with small fresh contexts reduce energy
  roughly in proportion to the context they avoid.
- **Choice of model.** Smaller models are much cheaper per token (Hausfather cites about
  5–7× between Haiku-class and Opus-class models). The coefficients here assume
  frontier-scale models.
- **Carbon.** Emissions depend on the electricity source. Hausfather's figures used the US
  grid average (341 gCO₂e/kWh); market-based accounting would be lower.

## 7. Main uncertainties

1. **No disclosure from Anthropic.** There are no published per-token, per-prompt or fleet
   inference energy figures for Claude
   ([Watershed via Hausfather](https://skepticalscience.com/print.php?n=6600) calls this
   the field's largest data gap).
2. **Attention architecture and KV cache size of Claude models.** These set K_in and K_out,
   which dominate at long context.
3. **Serving efficiency for agent traffic.** Batch sizes, utilisation and the hardware mix
   across TPU, Trainium and GPU.
4. **KV storage tier.** Whether cached context stays in accelerator memory between turns or
   is offloaded and reloaded (the kv term).
5. **Fleet figures.** Compute online, the inference share and utilisation are
   press-reported, not disclosed.

## Sources

Measurements and models
- Muxup, Per-query energy consumption of LLMs (InferenceMAX analysis): https://muxup.com/2026q1/per-query-energy-consumption-of-llms
- SemiAnalysis InferenceMAX benchmark: https://inferencemax.semianalysis.com/
- Microsoft Research, Energy use of AI inference (Joule, 2026): https://www.microsoft.com/en-us/research/?p=1170806
- Microsoft Research, preprint: https://arxiv.org/pdf/2509.20241
- Google, Measuring the environmental impact of AI inference: https://arxiv.org/pdf/2508.15734
- MIT Technology Review on Google's figures: https://www.technologyreview.com/2025/08/21/1122288/google.gemini.ai.energy/
- Gemini energy breakdown and batching explainer: https://howaiworks.ai/glossary/ai-energy-consumption
- Luo et al., Where does the energy go? Profiling LLM agent inference on Blackwell GPUs: https://arxiv.org/pdf/2609.29707
- Epoch AI, How much energy does ChatGPT use?: https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use
- Sam Altman, The Gentle Singularity: https://blog.samaltman.com/the-gentle-singularity
- DeepSeek V3.2 paper (sparse attention): https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf

Agentic usage logs
- Simon Couch, Electricity use of AI coding agents: https://simonpcouch.com/blog/2026-01-20-cc-impact/
- Zeke Hausfather, The real energy use of agentic AI: https://skepticalscience.com/print.php?n=6600

Fleet and company data
- Anthropic, Google and Broadcom compute partnership: https://www.anthropic.com/news/google-broadcom-partnership-compute
- Anthropic compute reporting (FourWeekMBA): https://fourweekmba.com/ai-anthropic-gigawatt-threshold-compute-unit-of-account/
- Anthropic compute reporting (Crypto Briefing): https://cryptobriefing.com/anthropic-10-gw-compute-capacity-2027/
- Anthropic run rate (Unite.AI): https://www.unite.ai/?p=453467
- Anthropic run rate (Bloomberg via BGOV): https://news.bgov.com/financial-accounting/anthropic-revenue-run-rate-surpasses-65-billion-ahead-of-ipo
- Anthropic pricing: https://platform.claude.com/docs/en/about-claude/pricing
- Google data-centre electricity 2025 (Paperjam): https://paperjam.lu/article/google-a-consomme-37-delectricite-en-plus-en-2025
- Google data-centre electricity (S&P Global): https://www.spglobal.com/energy/en/technology-ai-research-insights/blogs/googles-search-for-power-sustainability-is-yielding-results-but-ai-slows-progress
- Google tokens processed, I/O 2026: https://blog.google/innovation-and-ai/sundar-pichai-io-2026/

Subscriptions and margins
- SemiAnalysis, Anthropic subscriptions offer 5x+ more value: https://newsletter.semianalysis.com/p/anthropic-subscriptions-offer-5x
- WebProNews (citing The Information) on gross margin: https://www.webpronews.com/anthropics-margin-squeeze-inference-costs-bite-as-revenue-surges/
- BuildMVPFast on S-1 status: https://www.buildmvpfast.com/blog/anthropic-ipo-s1-sec-filing-2026
- Where's Your Ed At, sceptical view: https://www.wheresyoured.at/anthropics-profitability-swindle/
- The Register, agents moved to API billing: https://www.theregister.com/ai-ml/2026/05/14/anthropic-tosses-agents-into-the-api-billing-pool/5240748
- TechCrunch, 2025 rate limits: https://techcrunch.com/2025/07/28/anthropic-unveils-new-rate-limits-to-curb-claude-code-power-users/
