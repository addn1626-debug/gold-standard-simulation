# Gold Standard Simulation

> **S(t) < Dc(t) + Di(t)**
> 
> A coupled differential equation simulation proving the mathematical impossibility of the gold standard under real-world supply and demand constraints.

---

## The Core Argument

Gold mine production is physically bounded by a logistic ceiling. Global M2 money supply grows exponentially with no physical constraint. Industrial gold demand is accelerating with AI adoption.

The exponential always overtakes the logistic. This is not a policy failure. It is a mathematical certainty.

**Crossover estimated: ~2006–2007** — consistent with the structural pressures that preceded 2008 and the subsequent explosion in central bank gold purchases.

---

## What the Simulation Does

Seven coupled differential equations track the economy across a 100-year horizon:

| Metric | Symbol | Description |
|--------|--------|-------------|
| Monetary Stability | M(t) | Currency backing ratio |
| Industrial Production | I(t) | Fraction of gold demand fulfilled |
| Price Level | P(t) | Inflation / deflation index |
| Economic Growth | G(t) | Real GDP trajectory |
| Reserve Adequacy | R(t) | Central bank reserve sufficiency |
| Technological Progress | T(t) | Cumulative technology deployment |
| Social Welfare | W(t) | Inverse-sensitivity-weighted composite |

Four allocation scenarios are tested — what happens when you try to distribute scarce gold between the central bank and industry:

| Scenario | α | Outcome |
|----------|---|---------|
| All to Central Bank | 1.0 | Monetary stability preserved, technology collapses |
| All to Industry | 0.0 | Technology advances, currency collapses |
| 50/50 Split | 0.5 | Both systems underperform simultaneously |
| Hoarding / Neither | S→0 | Sharpest welfare cliff post-shock |

No allocation scenario escapes the supply constraint.

---

## Data Sources

All data embedded directly as verified historical arrays. No API keys or external dependencies.

| Data | Source |
|------|--------|
| Gold mine production 1950–2024 | World Gold Council (2025) |
| Known reserves | USGS Mineral Commodity Summaries, January 2025 |
| Global M2 money supply 1960–2024 | CEIC / Econovis Global Broad Money (Q2 2024) |
| Industrial gold demand 1970–2024 | WGC Technology Demand Series (2024) |

**Key figures:**
- 2024 mine production: 3,661 t/yr (WGC, estimated all-time high)
- Proven reserves: ~59,000 tonnes — approximately 16 years at current pace
- Global M2 2024: ~$123.3 trillion, CAGR 6.8% since 2000
- Industrial demand 2024: 326 tonnes, up 7% driven by AI

---

## Robustness

Five systematic challenges confirm the conclusion is invariant:

- **M2 growth at 1%** (vs 6.8% actual) → crossover still occurs before 2100
- **Industrial demand at 10% of baseline** → crossover barely moves (currency demand alone is sufficient)
- **Recycling doubles to 2,740 t/yr** → delays decades only, does not prevent
- **Production scales to 5× baseline** → exponential M2 always wins
- **Reserve requirements fall to 0.1%** → abandons gold standard by definition

---

## How to Run

```bash
pip install numpy scipy matplotlib
python gold_simulation_complete.py
```

Produces seven output figures:

```
stage1_interaction_graph.png      — 7-metric coupling diagram
stage2_sensitivity_heatmap.png    — hyperparameter sensitivity matrix
stage3_weights.png                — inverse sensitivity welfare weights
stage4_data_fit.png               — real data curve fitting
stage5_core_argument.png          — S(t) < Dc(t) + Di(t) visualised
stage6_four_scenarios.png         — four allocation scenarios
stage7_robustness.png             — five reviewer challenges
```

---

## Paper

The formal write-up is included as `GoldStandard_addn_Final.pdf`

Covers the full methodology: empirical data foundations, mathematical formalisation, ODE system derivation, sensitivity analysis, scenario results, robustness battery, and discussion.

---

## Structure

```
gold-standard-simulation/
├── gold_simulation_complete.py   # Full simulation — all 7 stages
├── GoldStandard_addn_Final.pdf   # Formal paper
└── README.md
```

---

## License

MIT — use freely, cite if you build on it.

---

*addn. · June 2026*
