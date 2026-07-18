# Value Wholesale recommended demo flow

This is an 8–10 minute presenter flow for the Value Wholesale shopping agent. It starts with a
customer need, reveals the live agent trace as evidence, and finishes with the two independent
memory paths and Gemini model selector.

## Before the session

1. Open [http://34.182.213.82](http://34.182.213.82), scroll to **Presenter controls** at the
   bottom of the page, and click **Reset demo cache**. Confirm the reset and wait for the ready
   message. This flushes the managed LangCache so each scripted pair starts with a miss followed
   by a semantic hit. It does not delete seeded or learned long-term memories. The Value Wholesale
   brand at the top-left opens this guide in a new tab.
2. Scroll back to the top of the page.
3. Confirm that all eight service indicators are blue:
   Redis database, Context Retriever, Semantic Router, Embedding Cache, LangCache, Agent Memory,
   ADK Memory Bank, and ADK Agent Sessions.
4. Select **Alex Rivera** in the **Shop as** dropdown.
5. Leave **Gemini 2.5 Flash** selected.

The demo uses the fictional member `member-1001`, Alex Rivera, whose home warehouse is the
Portland Harbor location.

For the high-cardinality memory scenario, switch to `member-1005`, Taylor Morgan. Taylor has
exactly 500 pre-seeded memories: 20 durable preferences and 480 episodic distractors.

During page warm-up, the application loads the shared local
`redis/langcache-embed-v3-small` model and primes routing before Vale generates a short member
greeting. When a member is selected, the application deterministically loads the authoritative
member profile from Context Retriever, caches it for the new shopping session, and supplies it to
the greeting agent. The greeting agent may optionally retrieve Redis Agent Memory; it does not
retrieve the profile again. Wait for the greeting to finish before starting the scripted prompts.
The first shopping turn silently reuses the cached profile, so no member-profile hydration row
should appear in its trace. Changing the selected member clears the visible chat and creates a new
session with its own profile hydration.

The Context Retriever tool catalog is discovered once during page warm-up and cached by the
application. Profile hydration during member selection and later agent tool selection reuse it, so
the trace intentionally hides the redundant `discover MCP tools` step. Reload the page when you
want to refresh the catalog.

## 1. Grounded product discovery and live inventory

Prompt:

> Find family-size pantry staples under $30 and check Portland stock.

Expected answer:

- North Trail Organic Oats, 10 lb — member price `$15.99`, Portland quantity `31`.
- Harbor Select Extra Virgin Olive Oil, 2 x 1L — member price `$21.99`, Portland quantity `42`.

Point to the live trace while the request runs. It should show:

1. RedisVL Semantic Router allow/cache decision.
2. LangCache eligibility.
3. Redis short-term session retrieval.
4. Redis long-term memory retrieval.
5. ADK short-term session read and ADK Memory Bank search.
6. RedisVL catalog search.
7. Two governed `get_inventory_by_id` calls, each with its own latency.
8. `ADK Runner + Gemini` and total request time.

Talk track: the model reasons about the request, but product, price, and stock claims come from
Redis-backed tools. Inventory is accessed through the governed Context Retriever surface rather
than being invented by the model.

## 2. Show operational context and order history

Prompt:

> What do you know about me?

Expected result: the agent combines Alex's membership profile with live order context. It should
lead with order `VH-ORD-1048`, which is ready for pickup at Portland, and may briefly mention the
recent delivered order `VH-ORD-1026`. The trace should expose the Context Retriever order lookup
rather than stopping after the preloaded profile or using a hidden database call.

Talk track: Context Retriever gives the agent a controlled tool contract over live commerce
entities such as members, inventory, orders, and order lines.

## 3. Demonstrate semantic response caching

LangCache uses three versioned semantic scopes in the same managed cache: `policy:v1`,
`product-education:catalog-v1`, and `shopping-guide:v1`. The scope is prefixed inside the prompt
to keep semantically similar questions from different workloads distinct without relying on
undeclared preview attributes.

### Policy scope

First prompt:

> What is the electronics return policy?

Expected grounded answer: electronics have a 90-day return window. The first request should show
a LangCache miss and normal policy retrieval/generation.

Then ask:

> How long is the return window for electronics ?

The second request should show a semantic LangCache hit and skip `ADK Runner + Gemini`. Explain
that the RedisVL Semantic Router allows reusable ecommerce answers into LangCache while
personalized and live-data requests bypass it. Out-of-domain requests are blocked before cache,
memory, or model execution. Expand the LangCache hit to compare the current query with the cached
query that matched it.

### Product-education scope

First prompt:

> What flavor notes does Rain City Medium Roast Coffee have?

Then ask in a new session or after clearing the trace:

> How would you describe the taste of your whole-bean medium roast?

The first request generates a stable, catalog-grounded description. The paraphrase should hit
the `product-education:catalog-v1` scope. Cached product education excludes price, availability,
orders, member preferences, and other volatile or personalized fields.

### Shopping-guide scope

First prompt:

> How should I store a large bag of rolled oats after opening?

Then ask:

> What is the best way to keep bulk oats fresh?

The paraphrase should hit `shopping-guide:v1`. This demonstrates reusable guidance rather than
only policy FAQs. Guides remain generic and cannot contain member or live-commerce data.

For every cacheable example, expand the Semantic Router and LangCache trace rows. Point out the
versioned scope, the first miss, the semantic hit, the current-versus-cached query comparison, and
`Total request (0 llm calls)` on the hit.

## 4. Show both memory systems on every request

Prompt:

> What household products and pickup options do I prefer?

Expected memories:

- Alex prefers fragrance-free household and laundry products.
- Alex prefers warehouse pickup at the Portland Harbor location.

Expand both memory steps in the trace. The same query is sent to Redis Agent Memory long-term
memory and ADK Memory Bank, and the retrieved facts and wall-clock latency are shown independently.
Only Redis results are included in Gemini's context. ADK session and Memory Bank reads are marked
telemetry-only and can finish after the answer because they do not block generation.

Talk track:

- Redis Agent Memory also holds the independent, append-only short-term event stream.
- Agent Platform Sessions hold the ADK conversation session.
- Redis and ADK long-term memory remain separate retrieval systems, making their behavior visible
  in the same customer request.

Then use the intentionally noisy evaluation prompt:

> Which laundry detergent fits my preferences?

The checked-in corpus contains the same ten Alex facts in both systems, including unrelated
shopping and episodic memories. With the current managed services, Redis Agent Memory's
similarity threshold returns only the fragrance-free laundry fact. ADK Memory Bank returns that
fact plus top-k distractors such as snack and receipt preferences. Expand both trace steps to
show the precision difference; this compares retrieval behavior, not answer quality.

## 5. Save and recall a new preference

Prompt:

> Remember that I prefer fragrance-free household products and Portland pickup.

The trace should show the explicit Redis Agent Memory write. ADK queues the completed session for
Memory Bank generation after the turn. Redis promotion and Memory Bank generation are eventually
consistent, so do not promise that newly generated long-term facts appear synchronously.

Follow with:

> Based on what you remember, what laundry option should I consider and can I pick it up in Portland?

Expected behavior: the agent recalls the preference, finds Clear Tide Laundry Pods, and reports
that Portland inventory is `0`. This is a useful proof that personalization does not override live
stock truth.

## 6. Switch models without losing the session

Keep the same conversation, change the composer dropdown to **Gemini 2.5 Pro**, and ask:

> Plan the best pantry purchase under $40 using my preferences and explain the trade-off.

Point out that the generation trace names `gemini-2.5-pro`. Both model-specific ADK runners share
Agent Platform Sessions and ADK Memory Bank for persistence and measurement, while Redis continues
to provide the conversation context sent to Gemini. ADK prior-session contents remain excluded.

## 7. Close on transparency

Summarize the trace from top to bottom:

- RedisVL Semantic Router allow/block and cache decision;
- short-term context;
- Redis long-term memories and facts;
- ADK Memory Bank memories and facts;
- governed MCP and commerce tool calls;
- selected Gemini model;
- `ADK Runner + Gemini` and total latency.

The key message is that Value Wholesale does not merely produce an answer: it exposes where context
came from, which actions ran, which memory system returned each fact, and how long each step took.

## Reliable fallback prompt

If a live product flow is interrupted, use:

> What is the electronics return policy?

It is non-personalized, deterministic, and exercises the policy grounding plus LangCache path.
