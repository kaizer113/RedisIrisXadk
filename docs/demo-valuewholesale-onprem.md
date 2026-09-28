# Value Wholesale self-managed Iris demo

This is a presenter flow for the Value Wholesale comparison demo whose LangCache, Context
Retriever, and Agent Memory services run in the self-managed GKE deployment. Google ADK,
Vertex sessions, Gemini, and ADK Memory Bank remain the same as in the Redis Cloud-backed demo.

Demo URL: `http://34.48.172.111:8083`

## Before the session

1. Open the demo and wait for the greeting and service warm-up to finish.
2. Confirm **Context Retriever** is enabled in the Redis Iris services panel.
3. Select **Alex Rivera** and leave **Gemini 3.1 Flash-Lite** selected.
4. If prior activity could affect the walkthrough, use **Reset demo cache** and **Reset member
   memory** under Presenter controls.

## 1. Show Context Retriever hybrid product discovery

Click **Hybrid Product Finder**. The three bullets beside it are runnable demo prompts: clicking
one fills its query and governed filters, runs the search, writes the results into the conversation,
and records the call in the existing live agent trace.

### Travel audio with category and price filters

Prompt:

> wireless audio for private listening while traveling

- Category: `electronics`
- Maximum member price: `$80`
- Expected leading results: Northstar Wireless Earbuds Value Pack at `$49.99`, followed by the
  Family Pack at `$67.49`.

The `$80` limit is an inclusive structured pre-filter, not part of semantic relevance. Context
Retriever applies the category and member-price predicates before its lexical and vector ranking
legs. The natural-language phrase demonstrates the hybrid ranking; the price demonstrates governed
filtering. A semantic-distance guardrail removes broader wireless electronics before display.

### Sensitive-skin laundry

Prompt:

> free and clear laundry detergent for sensitive skin

- Category: `household`
- Maximum member price: `$35`
- Expected leading result: Clear Tide Laundry Pods, 152 count, at `$31.99`.

This demonstrates that the customer's wording does not need to reproduce the catalog description
exactly. The catalog describes the product as “free-and-clear concentrated laundry detergent
pods,” while the query expresses the shopping intent.

### Coffee described by flavor notes

Prompt:

> whole bean medium roast coffee with cocoa and caramel notes

- Category: `beverages`
- Maximum member price: `$30`
- Expected leading result: Rain City Medium Roast Coffee, 3 lb, at `$25.49`.

This is a useful semantic-plus-lexical example: product type, roast, format, and flavor notes all
contribute to ranking while category and price remain deterministic filters.

### Additional reliable prompt

Prompt:

> fresh vegetarian party snack with dip

- Category: `fresh-food`
- Maximum member price: `$20`
- Expected leading result: Market Garden Vegetable Tray, 4 lb, at `$14.49`.

## 2. Explain the trace

For each Hybrid Product Finder request, point to the two Context Retriever trace steps:

`Context Retriever · semantic relevance guardrail`

`Context Retriever · hybrid product search`

The semantic step shows the calibrated cosine-distance cutoff. The hybrid step shows each retained
product's hybrid rank, semantic rank and distance, and its rank in a companion lexical search.
The short trace label maps to the governed tool `search_product_by_semantic_embedding_hybrid`.
Explain that the companion searches make the two retrieval signals visible; the hybrid tool's RRF
order remains authoritative. The lexical tool does not accept the hybrid tool's structured-filter
arguments, so its rank is diagnostic: the application reapplies the same category and price
eligibility when presenting that companion list. The application does not send this request through
Gemini: the result is a direct, governed retrieval demonstration.

## 3. Compare with the normal shopping-agent path

Use the regular chat prompt:

> Find family-size pantry staples under $30 and check Portland stock.

The regular agent path uses RedisVL catalog retrieval, calls governed Context Retriever tools for
operational inventory, combines memory context, and asks Gemini to compose the response. Contrast
that multi-step trace with the Hybrid Product Finder's single governed Context Retriever search.

## 4. Exercise the self-managed services

Use these prompts in the regular chat to demonstrate parity with the Redis Cloud-backed demo:

- LangCache policy miss/hit pair:
  - `What is the electronics return policy?`
  - `How long is the return window for electronics?`
- Agent Memory retrieval:
  - `What household products and pickup options do I prefer?`
- Context Retriever order lookup:
  - `What do you know about me?`
- Combined memory, catalog, and inventory:
  - `Based on what you remember, what laundry option should I consider and can I pick it up in Portland?`

## Presenter notes

- The original Value Wholesale and Norling's demos do not expose Hybrid Product Finder. The
  capability is gated by `CONTEXT_HYBRID_PRODUCT_FINDER_ENABLED` and is enabled only here.
- Redis remains the operational source for catalog, price, and inventory data. Agent Memory is
  used for conversational and durable user context, not as the source of truth for commerce data.
- Hybrid RRF scores are relative and are not used as a portable relevance threshold. The demo uses
  a semantic guardrail: cosine distance must be at most `0.35` and within `0.06` of the best match.
  Those values are calibrated for this demo dataset and embedding model and require evaluation
  before reuse elsewhere.
- If a service call fails, verify the GKE services and private VM-to-GKE connectivity before
  changing application behavior.
