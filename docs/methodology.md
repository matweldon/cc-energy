# Methodology and evidence

This note explains where cc-energy's model and parameters come from, how they were checked
against independent evidence, and what remains uncertain. It reflects public information as
of October 2026.

**Bottom line.** For agentic coding at long context (150–250k tokens per call), the central
estimate is roughly 15–25 Wh per model call and 2–4 kWh per day for a heavy user. That uses
the default conventional attention. The low and high scenarios are three model
architectures, and they span roughly 2× either side. Choosing sparse attention cuts
long-context estimates by about 3–4×.

Three other lines of evidence bracket these figures: published measurements, fleet-level
(top-down) allocation, and an economic ceiling from subscription pricing. The central case
is consistent with all three. The high case, a 405B-dense model with full attention, reaches
the economic ceiling at long context. No frontier lab publishes per-token energy figures, so
every number here is an estimate.

---

## 1. The model

Each API call is charged:

```
E = W·(N·e_in  + prefill(C_w))      weights + prefill attention for each new token
  + O·(N·e_out + decode(C_o))       weights + decode attention for each output token
  + R·kv                            optional KV-cache storage
```

- **W:** fresh input tokens processed in the call (uncached input plus cache writes).
- **O:** output tokens.
- **R:** cache-read tokens (context already held in the KV cache).
- **N:** the model's active parameters.
- **C_w = R + W/2** and **C_o = R + W + O/2:** the average context each new or output token
  attends to. `--simple-context` uses C = R.
- **e_in, e_out:** joules per active parameter per token for the weights.
- **prefill(C):** attention arithmetic. FLOPs per context token × joules per FLOP × C.
- **decode(C):** attention memory reads. KV-cache bytes per context token × joules per byte
  × C.

Energy is computed in joules and reported in Wh (J ÷ 3600).

### Why this form

With KV caching, previously processed context is not recomputed. A call's cost has three
parts.

1. **Weights.** Every new token passes through the active weights. Output tokens cost more
   per token than input, because decoding generates one token at a time and is limited by
   memory bandwidth, while prefill processes tokens in parallel.
2. **Prefill attention.** Each new input token attends to all C tokens of context. This is
   arithmetic-bound, proportional to W·C. On a cache miss, the whole context reappears as W,
   and the W/2 term charges for rebuilding it. That is the genuinely quadratic case.
3. **Decode attention.** Each output token re-reads the KV cache for all C context tokens.
   This is memory-bound, proportional to O·C. Unlike the weights, which are read once per
   step and shared across every sequence in a batch, each sequence reads its own KV cache.
   Batching therefore does not reduce this term, which is why long-context decoding is
   expensive.

Two consequences follow:

- **Cache reads have no energy cost of their own in this model** (apart from the optional
  kv term). They only make new and output tokens more expensive. A flat "cache reads cost
  10% of input" factor, priced from API rates, overstates cheap turns and understates heavy
  ones.
- **The quadratic shows up at session level.** Context grows with every turn, so a
  session's total cost grows roughly with turns². Summed cache-read tokens in the logs
  already capture this.

### Fixed efficiency figures (one central set)

All are facility level: they include host CPU and memory, networking, cooling and building
overhead.

- **Weights, input:** e_in = 7.8 × 10⁻¹² J per active parameter per token. Calibrated from
  InferenceMAX benchmarks (section 2): about 0.29 J per input token for a 37B-active model.
- **Weights, output:** e_out = 5 × e_in = 3.9 × 10⁻¹¹ J per active parameter per token. The
  benchmark gives about 4.4×, and Anthropic's prices are 5:1.
- **Attention arithmetic:** 2 pJ per FLOP. A B200 does about 2.25 PFLOP/s of dense 16-bit
  maths at about 1 kW, roughly 0.45 pJ/FLOP at the chip. Multiply by about 1.7 for the rest
  of the server and 1.1 for the building (Google reports a fleet PUE of 1.09), then divide by
  about 40% realistic utilisation. That gives about 2 pJ/FLOP.
- **KV-cache reads:** 230 pJ per byte. A B200 streams about 8 TB/s of memory at about 1 kW,
  roughly 125 pJ/byte at the chip. The same 1.7 × 1.1 overheads give about 230 pJ/byte.
- **KV-cache precision:** 16-bit (2 bytes per value). An 8-bit cache would halve the decode
  attention term.

The weights figure (about 3.9 pJ per weight FLOP) is higher than the attention figure (2 pJ)
because it is measured end to end: it includes expert routing, inter-GPU communication and
idle time in disaggregated serving. It also absorbs a little attention cost from the
benchmark's 8k context, about 14% at 37B. Running the benchmark at fp4 makes it, if
anything, optimistic.

### Low, central and high: model architectures

Claude's architecture is not public. The three scenarios are three model scales, each with
128 query heads of size 128:

- **Low:** 37B active parameters, 61 layers. DeepSeek V3/V3.2 scale, the largest open model
  family with public serving benchmarks
  ([config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json)).
- **Central:** 122B active parameters, 96 layers. A **hypothetical** midpoint, the geometric
  mean of low and high.
- **High:** 405B active parameters (dense), 126 layers. Llama 3.1 405B scale
  ([specs](https://apxml.com/models/llama-3-1-405b)).

**The central model's size is the single largest assumption.** It sets the weights term,
which dominates at short context. At long context the attention terms dominate, and their
scale comes from the layer count.

#### Total versus active parameters

Energy follows **active** parameters, the weights each token actually passes through, not
total parameters. The low and central scenarios assume mixture-of-experts (MoE) models, in
which each token is routed through a few of many experts. DeepSeek V3, for example, has 671B
parameters in total but 37B active. Only the high scenario is dense, standing in for "much
more compute per token than current estimates".

What is known about Claude's size:

- **Anthropic has disclosed nothing:** no total or active parameter counts, expert counts
  or routing ratios.
- **The widely repeated ~1T (Sonnet) and ~5T (Opus) figures** are total-parameter numbers.
  They come from Elon Musk's April 2026 remarks comparing Grok's size to Claude, extended
  in a July 2026 post on X. They are unverified and may describe older models
  ([36Kr](https://eu.36kr.com/en/p/3760679047267075);
  [NextBigFuture](https://www.nextbigfuture.com/2026/04/anthropic-and-xai-model-parameter-counts.html);
  [ExplainX review](https://www.explainx.ai/blog/claude-sonnet-opus-fable-parameter-counts-debate-july-2026);
  [AIThinkerLab review](https://aithinkerlab.com/claude-opus-5-trillion-parameters/)).
- **An independent estimate from serving speed**
  ([Unexcited Neurons](https://unexcitedneurons.substack.com/p/estimating-the-size-of-claude-opus))
  starts from the fact that decode speed is limited by the active weights read per token.
  It calibrates against open models of known size on the same platform (Google Vertex) and
  puts **Opus 4.6 at about 100–150B active parameters and roughly 1–3T total**, ±50%. It
  puts Sonnet at about two-thirds of Opus. The same analysis suggests Opus 4/4.1 were much
  larger (about 5–6T total), which is consistent with their 3× higher prices and may be the
  origin of the 5T figure.

The central scenario (122B active) sits inside the throughput-based range. The low scenario
(37B) is probably too small for Opus-class models but plausible for smaller tiers. These
estimates are for Opus 4.5/4.6; nothing public pins down later versions.

Total size still matters somewhat. A model with trillions of parameters needs more
accelerators just to hold its weights, and routing tokens between experts adds
communication, both of which can reduce efficiency per active parameter. The weights
calibration partly covers this because it comes from an MoE model (DeepSeek, 671B total),
but a 2–3T model could be somewhat less efficient per active parameter than assumed.

### Attention types

These determine the attention terms. The default is GQA.

- **`gqa` (default), grouped-query attention, as in Llama 3.1.** 8 KV heads. Per context
  token per layer: 65,536 prefill FLOPs (4 × 128 × 128) and 4,096 bytes of KV
  (2 × 8 × 128 × 2). This is the most widely documented design in open models, and it is
  conservative: the most expensive of the three.
- **`mla`, multi-head latent attention, as in DeepSeek V3.** Stores a 576-value compressed
  latent per token per layer (1,152 bytes) and uses about 81,920 prefill FLOPs per pair per
  layer (2 × 128 × (192 + 128)). Decoding is about 3.6× cheaper than GQA; prefill is slightly
  dearer.
- **`sparse`, DeepSeek sparse attention, as in V3.2.** MLA applied only to the top 2,048
  tokens chosen by a lightweight 8-bit indexer (64 heads of size 128). Beyond about 2k
  context, only the indexer grows with C, so long-context costs flatten
  ([DeepSeek V3.2 paper](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)).

For agentic workloads, prefill attention (W × C) dominates. GQA and MLA therefore give
similar totals (within about 15–30%), while sparse attention is about 3–4× lower at long
context.

Derived per-token values, GQA:

- **Low:** input 0.29 J and output 1.44 J at zero context. Each 1,000 context tokens adds
  0.008 J per input token and 0.057 J per output token.
- **Central:** input 0.95 J, output 4.75 J. Plus 0.013 J and 0.090 J per 1,000 context
  tokens.
- **High:** input 3.15 J, output 15.8 J. Plus 0.017 J and 0.119 J per 1,000 context tokens.

---

## 2. Evidence for the efficiency figures

- **InferenceMAX benchmarks,** analysed by Muxup
  ([post](https://muxup.com/2026q1/per-query-energy-consumption-of-llms);
  [benchmark](https://inferencemax.semianalysis.com/)). For DeepSeek-R1-0528 (37B active)
  at fp4 on B200, one 8k-in/1k-out query used about 0.93 Wh and one 1k-in/8k-out query about
  2.73 Wh. The benchmark's actual sequence lengths average 0.9× nominal. Solving the two
  equations gives about **80 Wh per million input tokens** and **360 Wh per million output
  tokens**, i.e. 0.29 J and 1.3 J per token, a ratio of about 4.4. These are best-case
  configurations; Muxup suggests real providers might achieve about half that efficiency.
- **Microsoft Research (Joule, 2026)**
  ([summary](https://www.microsoft.com/en-us/research/?p=1170806);
  [preprint](https://arxiv.org/pdf/2509.20241)). Optimised frontier inference is put at a
  median of 0.31 Wh per query (interquartile range 0.16–0.60) for about 300 output tokens,
  and 3.91 Wh for 5,000 output tokens. Agentic and reasoning queries cost more than ten
  times as much, partly because of lower serving concurrency.
- **Google, Gemini Apps** ([paper](https://arxiv.org/pdf/2508.15734);
  [MIT Technology Review](https://www.technologyreview.com/2025/08/21/1122288/google.gemini.ai.energy/)).
  The median text prompt uses 0.24 Wh: 0.14 accelerator, 0.06 host CPU/DRAM, 0.02 idle,
  0.02 overhead ([breakdown](https://howaiworks.ai/glossary/ai-energy-consumption)).
  Accelerators are only about 58% of the total, which supports the ~1.7× server overhead
  factor.
- **Full-system profiling on Blackwell** ([Luo et al. 2026](https://arxiv.org/pdf/2609.29707)).
  GPU-only telemetry missed 41–45% of system energy, again supporting the ~1.7× factor. A
  sequential single-user agent used 63× more energy per output token than saturated
  serving. Per-turn energy rose over agent trajectories while power stayed flat, consistent
  with growing prefill. Extended thinking changed token volume, not per-token efficiency.
- **OpenAI and Epoch AI.** Sam Altman put an average ChatGPT query at 0.34 Wh
  ([blog](https://blog.samaltman.com/the-gentle-singularity)). Epoch AI estimated about
  0.3 Wh for GPT-4o and about 2.5 Wh for a 10k-token-input query
  ([analysis](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use)).
- **Google PUE** of 1.09 for 2025
  ([Paperjam](https://paperjam.lu/article/google-a-consomme-37-delectricite-en-plus-en-2025)).

---

## 3. Checks against real usage

Two people have published token logs from their own Claude Code use.

- **Simon Couch, January 2026**
  ([post](https://simonpcouch.com/blog/2026-01-20-cc-impact/)). His median session had 24
  calls and 592k tokens, about 23k context per call. He estimated 41 Wh using flat factors
  derived from Epoch AI (390 / 1,950 / 490 / 39 Wh per million input / output /
  cache-write / cache-read tokens). This model gives **6 / 15 / 42 Wh** (low / central /
  high, GQA). At this short context the weights term dominates, so the result tracks model
  size.
- **Zeke Hausfather, August 2026**
  ([post](https://skepticalscience.com/print.php?n=6600)). Over 8 weeks, 1,138 prompts led
  to over 14,000 calls and 3.2 billion tokens. 96% were cache reads and about 0.4% output,
  with roughly 220k context per call. His central estimate across three methods, including
  Watershed's activity-tier factors (Bistline et al. 2026), was **3.0 kWh/day (1.2–5.9)**.
  This model gives **2.1 / 3.7 / 6.5 kWh/day** with GQA, 1.7 / 3.2 / 5.8 with MLA, and
  0.4 / 1.1 / 3.1 with sparse attention.

Both posts make the same point: published per-prompt figures (0.24–0.34 Wh) describe chat,
not agents. One agentic prompt triggers around ten calls, each re-reading a long context.

---

## 4. Top-down (macro) checks

Macro allocation estimates each user's share of everything that exists to serve them,
including idle and ramping capacity, free tiers, internal use and redundancy. It is expected
to come out higher than the marginal bottom-up model.

### Anthropic: fleet energy per dollar of revenue

- **Compute:** reported at about 1.4 GW at the end of 2025 and an expected 5 GW by the end
  of 2026. These figures come from investor reporting, not Anthropic
  ([FourWeekMBA](https://fourweekmba.com/ai-anthropic-gigawatt-threshold-compute-unit-of-account/);
  [Crypto Briefing](https://cryptobriefing.com/anthropic-10-gw-compute-capacity-2027/)).
  A further 3.5 GW of TPU capacity is contracted from 2027
  ([Anthropic](https://www.anthropic.com/news/google-broadcom-partnership-compute)).
- **Revenue:** the run rate was about $9bn at the end of 2025 and $47bn in May 2026
  (official), with a reported $65bn by the end of July
  ([Unite.AI](https://www.unite.ai/?p=453467);
  [Bloomberg via BGOV](https://news.bgov.com/financial-accounting/anthropic-revenue-run-rate-surpasses-65-billion-ahead-of-ipo)).
- **Assumptions:** about 3.2 GW online, 70% average draw, 55% of it serving customer
  inference, PUE 1.1, and an $85bn run rate. That gives about **140 Wh per dollar** of
  list-price revenue, with a range of roughly 55–300.
- **Applied to the usage logs:** Couch's median session (about $0.50 at list price) comes
  to about 70 Wh (30–150), against 15 Wh central bottom-up. Hausfather's heavy day (about
  $46) comes to about 6 kWh (2.5–14), against 3.7 kWh. Macro sits about 1.6× above central
  for heavy long-context use, and more for light sessions, as expected.

### Google: a loose upper bound

Google's data centres used 42.4 TWh in 2025
([Paperjam](https://paperjam.lu/article/google-a-consomme-37-delectricite-en-plus-en-2025);
[S&P Global](https://www.spglobal.com/energy/en/technology-ai-research-insights/blogs/googles-search-for-power-sustainability-is-yielding-results-but-ai-slows-progress)),
and Google processed over 3.2 quadrillion tokens a month in May 2026
([Google](https://blog.google/innovation-and-ai/sundar-pichai-io-2026/)). If all of that
electricity went to tokens, it would be about 1,100 Wh per million tokens. AI is only part
of Google's load, mostly on Flash-class models, so a realistic blended figure is a few
hundred Wh per million tokens.

---

## 5. An economic ceiling from subscription pricing

SemiAnalysis measured plan allowances under an agentic workload
([article](https://newsletter.semianalysis.com/p/anthropic-subscriptions-offer-5x)). Claude
Pro ($20 a month) allows about 2.9 billion Opus 5.5 tokens, or 144M per dollar: 0.4% input,
96.6% cached input, 2.6% cache writes, 0.3% output. Max 5× and Max 20× allow 14.0B and
28.6B.

**The assumption:** Anthropic does not sell tokens for less than the electricity they use.

- **The ceiling.** A fully used $20 plan can use at most $20 ÷ electricity price. At an
  assumed 8.5¢/kWh (range 5–15¢), that is about **235 kWh a month** (133–400), or roughly
  80 Wh per million tokens blended.
- **The model, same month** (W = 87M, O = 8.7M, R = 2,801M). Low / central / high in kWh:
  - GQA at 150k / 200k context: 60 / 113 / 217 and 77 / 139 / 252.
  - MLA at 150k / 200k: 53 / 101 / 201 and 67 / 123 / 230.
  - Sparse at 150k / 200k: 15 / 42 / 124 and 17 / 44 / 127.
- **What the ceiling implies.** The central case uses about 50–60% of the ceiling. The high
  case (405B dense with full attention) reaches or exceeds it at long context. Either Opus is
  not that large with full attention, or it is served more efficiently than assumed here, or
  maxed-out subscribers are subsidised (see below). The ceiling therefore weighs against the
  high-GQA combination, and does not constrain the low or central cases.

### Is the assumption sound?

The common claim that labs "subsidise tokens" mixes up three things:

1. **Below electricity cost.** Unlikely. At the central estimate, electricity for a
   maxed-out $20 plan is about $10–12; for typical use it is a fraction of that.
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
  These changes fit a compute-constrained company pushing heavy use towards higher-paying
  API demand, more than they prove losses.

**Caveat.** Plans are priced on average use, so a single maxed-out subscriber could cost
more than the fee without breaking the assumption on average. Treat the ceiling as a soft
bound on typical use, not a hard per-token limit.

---

## 6. Interpreting the output

- **What the figures are.** cc-energy reports **marginal, bottom-up, facility-level**
  estimates.
- **Real-world serving.** Multiply by about 1–2 for production serving below benchmark
  efficiency (smaller batches for agent traffic, imperfect utilisation).
- **Attributional share of fleet energy.** Multiply the central figure by about 1.5–2 for
  heavy long-context use, more for light use.
- **Context dominates at long context.** With GQA, the attention terms exceed the weights
  terms above about 25–75k tokens of context for the low and central models, and about
  130–190k for the high model, whose weights cost is larger. Compaction and subagents with
  small fresh contexts reduce energy roughly in proportion to the context they avoid.
- **Model choice.** Smaller models are cheaper per token. Hausfather cites about 5–7×
  between Haiku-class and Opus-class models. The defaults assume frontier-scale models.
- **Carbon.** Emissions depend on the electricity source. Hausfather used the US grid
  average (341 gCO₂e/kWh); market-based accounting would be lower.

## 7. Main uncertainties

1. **Claude's architecture.** Active parameters, layers and attention design are not
   public. The best independent estimate (100–150B active for Opus 4.6) supports the
   central case, but carries ±50% uncertainty. The attention type matters as much: sparse
   attention would cut long-context estimates by about 3–4×.
2. **No disclosure from Anthropic.** There are no published per-token, per-prompt or fleet
   inference energy figures for Claude. Watershed calls this the field's largest data gap
   ([via Hausfather](https://skepticalscience.com/print.php?n=6600)).
3. **Serving efficiency for agent traffic.** Batch sizes, utilisation and the hardware mix
   across TPU, Trainium and GPU. The efficiency figures here are Blackwell-GPU based.
4. **KV-cache precision and storage.** 8-bit caches halve the decode term. Whether cached
   context stays in accelerator memory between turns or is offloaded and reloaded is
   covered by the kv term.
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

Architectures
- DeepSeek V3.2 model config: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json
- DeepSeek V3.2 paper (sparse attention): https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- Llama 3.1 405B specifications: https://apxml.com/models/llama-3-1-405b
- Unexcited Neurons, Estimating the size of Claude Opus 4.5/4.6: https://unexcitedneurons.substack.com/p/estimating-the-size-of-claude-opus
- ExplainX, Claude parameter counts debate (July 2026): https://www.explainx.ai/blog/claude-sonnet-opus-fable-parameter-counts-debate-july-2026
- 36Kr, Musk on Claude parameter counts: https://eu.36kr.com/en/p/3760679047267075
- NextBigFuture, Anthropic and xAI model parameter counts: https://www.nextbigfuture.com/2026/04/anthropic-and-xai-model-parameter-counts.html
- AIThinkerLab, evidence review of the 5T claim: https://aithinkerlab.com/claude-opus-5-trillion-parameters/

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
