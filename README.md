# agent-cost

Per-call cost tracking and attribution for LLM agents. Track costs by agent, task, and model. Set budgets, receive alerts, generate cost reports. Zero dependencies. Pure Python 3.8+.

## Install

```bash
pip install agent-cost
```

## Features

- Track every LLM call with full attribution (agent, task, model)
- Built-in pricing for 20+ models (OpenAI, Anthropic, Google, Meta, Mistral)
- Named budgets with alert thresholds and hard limits
- Aggregated cost reports with filtering
- Thread-safe for concurrent agents

## Usage

### Basic tracking

```python
from agent_cost import CostTracker

tracker = CostTracker()

# Track a call
record = tracker.track(
    model="gpt-4o",
    input_tokens=1200,
    output_tokens=400,
    agent="researcher",
    task="summarize",
)
print(f"Cost: ${record.cost_usd:.6f}")

# Get total
print(f"Total: ${tracker.total_cost():.4f}")
```

### Budgets with alerts

```python
def on_alert(budget, spent):
    print(f"WARNING: {budget.name} at {budget.utilization:.0%} (${spent:.2f})")

tracker.add_budget("monthly", limit_usd=100.0, alert_threshold=0.8, on_alert=on_alert)
tracker.add_budget("per-task", limit_usd=5.0, hard_limit=True)  # raises BudgetExceededError

tracker.track(
    model="claude-3-opus",
    input_tokens=5000,
    output_tokens=2000,
    agent="writer",
    task="draft",
    budget_names=["monthly", "per-task"],
)
```

### Cost reports

```python
report = tracker.report()
print(report.total_cost)         # total USD
print(report.by_model)           # breakdown per model
print(report.by_agent)           # breakdown per agent
print(report.top_models(5))      # top 5 models by cost

# Filter
report = tracker.report(agent="researcher", model="gpt-4o")
```

### Custom pricing

```python
tracker.add_pricing("my-fine-tuned-model", input_per_million=2.0, output_per_million=6.0)
```

## Supported Models (built-in pricing)

| Model | Input $/1M | Output $/1M |
|-------|-----------|------------|
| gpt-4o | $5.00 | $15.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-3-opus | $15.00 | $75.00 |
| claude-3-sonnet | $3.00 | $15.00 |
| gemini-1.5-pro | $3.50 | $10.50 |
| gemini-1.5-flash | $0.075 | $0.30 |
| ... and 15+ more | | |

## License

MIT
