# ValueHarbor recommended demo flow

This is an 8–10 minute presenter flow for the ValueHarbor shopping agent. It starts with a
customer need, reveals the live agent trace as evidence, and finishes with the two independent
memory paths and Gemini model selector.

## Before the session

1. Open [http://34.182.213.82](http://34.182.213.82).
2. Confirm that all seven service indicators are green:
   Redis database, Context Retriever, Semantic Router, LangCache, Agent Memory, ADK Memory Bank,
   and Agent Sessions.
3. Select **Alex Rivera** in the **Shop as** dropdown.
4. Leave **Gemini 2.5 Flash** selected.
5. Use a fresh browser reload so the visible conversation starts clean and the governed Context
   Retriever tool catalog is refreshed.

The demo uses the fictional member `member-1001`, Alex Rivera, whose home warehouse is the
Portland Harbor location.

For the high-cardinality memory scenario, switch to `member-1005`, Taylor Morgan. Taylor has
exactly 500 pre-seeded memories: 20 durable preferences and 480 episodic distractors.

After the page warm-up, Vale generates a short member greeting. The greeting agent can choose to
use Redis Agent Memory, Context Retriever, both, or neither. Wait for the greeting to finish before
starting the scripted prompts. The first shopping turn should show the authoritative member profile
being loaded from Context Retriever; later turns reuse the application session cache. Changing the
selected member clears the visible chat and creates a new session for that member.

The Context Retriever tool catalog is discovered once during page warm-up and cached by the
application. Profile hydration and agent tool selection reuse it, so the trace intentionally hides
the redundant `discover MCP tools` step. Reload the page when you want to refresh the catalog.

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

> Do I have a recent order ready for pickup, and where should I collect it?

Expected result: the agent finds order `VH-ORD-1048`, which is ready for pickup at Portland.
The trace should expose the Context Retriever order lookup rather than a hidden database call.

Talk track: Context Retriever gives the agent a controlled tool contract over live commerce
entities such as members, inventory, orders, and order lines.

## 3. Demonstrate semantic response caching

First prompt:

> What is the electronics return policy?

Expected grounded answer: electronics have a 90-day return window. The first request should show
a LangCache miss and normal policy retrieval/generation.

Then ask:

> How long do I have to return a laptop?

The second request should show a semantic LangCache hit and skip `ADK Runner + Gemini`. Explain
that the RedisVL Semantic Router allows reusable ecommerce answers into LangCache while
personalized and live-data requests bypass it. Out-of-domain requests are blocked before cache,
memory, or model execution.

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

The key message is that ValueHarbor does not merely produce an answer: it exposes where context
came from, which actions ran, which memory system returned each fact, and how long each step took.

## Reliable fallback prompt

If a live product flow is interrupted, use:

> What is the electronics return policy?

It is non-personalized, deterministic, and exercises the policy grounding plus LangCache path.
