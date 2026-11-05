# Value Wholesale GA demo content

These local source fixtures are not deployed or published automatically.

- `faqs/*.json` preserves the fictional policy answers in `data/generated/policies.jsonl` verbatim. Questions were adapted for FAQ discovery; metadata records their source. These are synthetic retailer policies, not Redis product claims or a real merchant's terms.
- `skills/shopping-comparison/SKILL.md` is an authored procedure. Agents discover its preview and load its published instructions on demand.
- `intents.json` defines the five routing intents and dispatch references consumed by the isolated onprem engine. Examples and score thresholds require evaluation with the deployed embedder.

Use the validation, planning, publishing, and intent-seeding commands in `docs/onprem-iris.md`. Publishing is an operator action using authoring credentials. Runtime keys cannot publish these fixtures. A changed source file has no effect until an operator applies and publishes it.
