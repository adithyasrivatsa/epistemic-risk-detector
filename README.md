# LLM Hallucination Debugger — v0.1

A research-grade, infrastructure-oriented prototype for inspecting epistemic risk in LLM outputs.

> **Verdicts are signals, not judgments. Human review is required for high-stakes use.**

---

## What This Actually Is

A diagnostic tool for analyzing:
- Claim extraction from LLM responses
- Evidence availability within a bounded corpus
- Confidence–evidence mismatch

Designed to **expose unsupported confidence**, not to assert truth.

**Useful for:**
- Research
- Eval tooling
- Internal audits
- Developer debugging

## What This Is Not

- ❌ Not a guaranteed hallucination detector
- ❌ Not a safety filter
- ❌ Not production-reliable without calibration
- ❌ Not ground-truth aware

---

## Core Concept

This system evaluates **epistemic risk** in LLM responses by comparing model confidence against available supporting or contradicting evidence within a bounded corpus.

**Hallucination (in this system) means:**
> High confidence in the absence of strong supporting evidence — not necessarily factual falsehood.

---

## Status

| Aspect | Status |
|--------|--------|
| Architecture | Production-grade (interfaces, schemas, modularity) |
| Detection logic | Heuristic + prompt-driven |
| Validation | Demo-backed, not statistically calibrated |
| Intended use | Research, experimentation, extension |

---

## Why This Still Matters

- Makes LLM overconfidence visible
- Forces explicit reasoning boundaries
- Useful even when evidence is missing
- Model-agnostic, local-first, inspectable

---

## Quick Start

```bash
# Install
pip install -e .

# Index your corpus
epistemic-risk index ./your-documents/

# Analyze an LLM response
epistemic-risk analyze "The Python GIL was removed in version 3.12"

# Run the demo (no API needed)
python examples/gil_demo.py
```

## The Demo Case: "Python 3.12 Removed the GIL"

This is the canonical test case because it's:
- **Partially true**: PEP 703 was accepted, per-interpreter GIL exists
- **Technically false**: GIL was NOT removed in 3.12
- **Commonly hallucinated**: LLMs confidently state this incorrectly
- **High semantic similarity**: Naive embedding-only systems miss the contradiction

```
┌─────────────────────────────────────────────────────────────────┐
│ CLAIM: "Python 3.12 completely removed the GIL"                 │
├─────────────────────────────────────────────────────────────────┤
│ Verdict: HALLUCINATED                                           │
│ Risk Score: 89%                                                 │
│                                                                 │
│ Model Confidence: 0.92 (raw) → 0.32 (calibrated)               │
│ Evidence Strength: 0.12                                         │
│                                                                 │
│ Contradiction Type: DIRECT_NEGATION                             │
│                                                                 │
│ Evidence Found:                                                 │
│   [CONTRADICTS] "Python 3.12 did NOT remove the GIL..."        │
│   [WEAK_SUPPORT] "PEP 703 proposes making the GIL optional..."  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
Input Answer → ClaimExtractor → EvidenceRetriever → AlignmentEvaluator 
            → ConfidenceCalibrator → VerdictEngine → OutputRenderer
```

Each module is independently testable and swappable via clean interfaces.

```
src/
├── core/
│   ├── interfaces.py      # Abstract base classes
│   ├── schemas.py         # Pydantic models (strict typing)
│   └── config.py          # Configuration management
├── extractors/
│   └── claim_extractor.py # Decomposes answers into claims
├── retrievers/
│   └── local_vector.py    # SQLite + local embeddings
├── evaluators/
│   └── alignment.py       # Semantic + logical alignment
├── calibrators/
│   └── confidence.py      # Confidence calibration with penalties
├── verdict/
│   └── engine.py          # Final verdict computation
└── renderers/
    ├── cli.py             # Terminal output
    └── structured.py      # JSON/Web-ready output
```

---

## How Confidence Calibration Works

Raw model confidence is unreliable. We apply transparent, documented penalties:

| Condition | Penalty | Rationale |
|-----------|---------|-----------|
| No evidence found | -0.40 | Claim cannot be verified against corpus |
| Contradiction detected | -0.60 | Evidence directly opposes the claim |
| Weak evidence only | -0.15 | Partial support isn't full support |
| Vague/hedging language | -0.20 | "might", "possibly" indicate uncertainty |

**These penalties are heuristics, not calibrated values.** They require tuning for your domain.

**Verdict Formula:**
```
hallucination_risk = (0.4 × raw_confidence) + (0.6 × (1 - evidence_strength))
```

| Condition | Verdict |
|-----------|---------|
| `evidence_strength < 0.3` | HALLUCINATED |
| `evidence_strength > 0.7` and no contradictions | GROUNDED |
| Otherwise | WEAK |

---

## Known Limitations

### Claim Extraction
- Relies entirely on LLM prompt compliance
- Will fail on nested claims, sarcasm, code blocks, non-English text
- Span offsets are approximate, not guaranteed accurate

### Evidence Retrieval
- Semantic similarity ≠ factual relevance
- Small corpora → most claims flagged as unverifiable
- No temporal awareness (evidence staleness not tracked)

### Alignment Evaluation
- Different LLMs produce different labels for identical inputs
- Negation detection is regex-based, will false-positive
- "CONTRADICTS" requires explicit negation in evidence text

### Calibration
- Penalty values are arbitrary, not empirically derived
- Thresholds (0.3, 0.7) are not calibrated to ground truth
- No confidence intervals on verdicts

---

## Extending

### Add a new evidence source

```python
from src.core.interfaces import EvidenceProvider

class WebSearchProvider(EvidenceProvider):
    def retrieve(self, claim: str, top_k: int) -> list[EvidenceChunk]:
        # Your implementation
        pass
```

### Swap the LLM backend

```python
from src.core.interfaces import LLMProvider

class AnthropicProvider(LLMProvider):
    def complete(self, prompt: str) -> str:
        # Your implementation
        pass
```

---

## Configuration

```yaml
# config.yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.0

retrieval:
  chunk_size: 512
  chunk_overlap: 64
  top_k: 5
  similarity_threshold: 0.3

calibration:
  no_evidence_penalty: 0.4      # Tune for your domain
  contradiction_penalty: 0.6    # Tune for your domain
  vague_language_penalty: 0.2   # Tune for your domain

verdict:
  hallucination_threshold: 0.3  # Tune for your domain
  grounded_threshold: 0.7       # Tune for your domain
```

---

## Testing

```bash
pytest tests/ -v
```

56 tests covering schemas, extraction, alignment, calibration, verdict logic, and rendering.

---

## What Would Be Needed for Production Use

- [ ] Ground truth evaluation dataset
- [ ] Precision/recall metrics
- [ ] Structured logging with request IDs
- [ ] Rate limiting for LLM APIs
- [ ] Proper vector database (not SQLite)
- [ ] Human-in-the-loop review workflow
- [ ] Domain-specific penalty calibration
- [ ] Prompt tuning per model provider

---

## License

MIT
