# Value Wholesale pickup-rescue demo flow

This is an alternative 8–10 minute presenter path for the Value Wholesale shopping agent. It
follows one customer story: Alex has an order waiting in Portland, discovers that a preferred
laundry item is unavailable there, finds it at another warehouse, and adds the grounded result to
the cart. Along the way, the trace exposes governed commerce access, semantic caching, two memory
systems, session continuity, model switching, and an out-of-domain guardrail.

Use this path when the audience will respond better to a continuous shopping journey than to the
feature-by-feature flow in [demo.md](demo.md).

## Before the session

1. Open [http://34.182.213.82](http://34.182.213.82), scroll to **Presenter controls**, and click
   **Reset demo cache**. Confirm the reset and wait for the ready message. This clears LangCache
   but does not delete seeded or learned long-term memories.
2. Return to the top and confirm that all eight service indicators are blue: Redis database,
   Context Retriever, Semantic Router, Embedding Cache, LangCache, Agent Memory, ADK Memory Bank,
   and ADK Agent Sessions.
3. Select **Alex Rivera** in **Shop as** and **Gemini 2.5 Flash** in **Model**.
4. Wait for the greeting to finish before sending the first prompt.

Alex is the fictional member `member-1001`. The deterministic dataset gives Alex a Portland home
warehouse, a ready-for-pickup order, a fragrance-free household preference, and a preference for
Portland pickup. Portland has no Clear Tide Laundry Pods in stock; Seattle has 24.

Changing the selected member creates a new session and clears the visible conversation. Keep Alex
selected throughout this flow so later turns can demonstrate short-term context and cart state.

## 1. Begin with the member's active order

Prompt:

> Give me an account overview and tell me if I have anything to pick up.

Expected behavior:

- Vale identifies Alex as a Harbor Plus member with an `$86.42` reward balance.
- It leads with order `VH-ORD-1048`, ready for pickup at Portland Harbor.
- It may briefly mention delivered order `VH-ORD-1026`.

Expand the Context Retriever rows. The authoritative member profile was hydrated when Alex was
selected, while the current order state comes from a governed order lookup. The agent should not
pretend that identity data alone is a complete account view.

Talk track: profile and order state have different freshness and access patterns. The agent reuses
the session's profile but retrieves operational order context through the controlled commerce
surface.

## 2. Inspect the pickup without repeating the account lookup

Prompt:

> What is in the order that is ready for pickup?

Expected contents of `VH-ORD-1048`:

- North Trail Organic Oats, 10 lb — `$15.99`.
- Family Dock Paper Towels, 12 rolls — `$23.99`.
- Rain City Medium Roast Coffee, 3 lb — `$25.49`.

Point out the order-item lookup in the trace. The previous turn established which order is active;
this turn drills into that single order instead of fetching unrelated order lines.

## 3. Answer and cache the pickup policy

First prompt:

> How long will Value Wholesale hold a pickup order after it is ready?

Expected grounded answer: pickup orders are held for three calendar days after the ready
notification. The named member must show photo identification at the pickup desk.

The trace should show a Semantic Router allow/cache decision, a LangCache miss, policy retrieval,
and `ADK Runner + Gemini`.

Then ask:

> How many days do I have to collect an order once pickup is ready?

The paraphrase should hit the versioned `policy:v1` LangCache scope and skip generation. Expand the
Semantic Router and LangCache rows to show the current query, cached query, semantic match, and
`Total request (0 llm calls)`.

Talk track: reusable policy answers can be cached semantically, while the surrounding order and
member turns remain outside the cache because they contain live or personalized data.

## 4. Ask for a personalized gap-fill

Prompt:

> Using my preferences and recent purchases, what laundry option should I add to this pickup, and
> is it in stock in Portland?

Expected behavior:

- Redis Agent Memory supplies Alex's preference for fragrance-free household and laundry
  products.
- The recent-order context shows that Alex previously purchased Clear Tide Laundry Pods.
- Catalog search validates Clear Tide Laundry Pods, 152 count, at the member price of `$31.99`.
- The governed Portland inventory lookup reports quantity `0`.
- Vale does not add anything to the cart because the request asked for a recommendation, not a
  cart mutation.

Expand both long-term-memory rows. Redis Agent Memory and ADK Memory Bank receive the same query
and report their facts and latency independently. Only Redis memory is included in Gemini's
context; the ADK read is telemetry-only and can complete after the answer.

Also point out that the agent can reuse the order-history snapshot already captured in Redis
short-term session events. It should not need to repeat the broad recent-order lookup before
searching the catalog.

Talk track: personalization proposes a relevant product, but it cannot override live inventory or
authorize a purchase.

## 5. Recover with live stock at another warehouse

Prompt:

> Check whether those Clear Tide pods are available in Seattle instead.

Expected answer: Seattle has `24` units of Clear Tide Laundry Pods. The trace should show a
governed lookup for inventory ID `seattle-vh-2002`.

Contrast this with Portland's quantity of `0`. The product is unchanged, but warehouse-specific
inventory is treated as live data and retrieved again rather than inferred from conversation
memory.

## 6. Make the cart mutation explicit

Prompt:

> Add one Clear Tide Laundry Pods to my cart.

Expected behavior: the `add_item_to_cart` tool records one unit of SKU `VH-2002` for Alex. Vale may
confirm the cart update, but it must not claim that checkout, payment, or an order placement
occurred.

Follow with:

> Show me my cart.

Expected behavior: `view_cart` returns one Clear Tide Laundry Pods entry. Point out that a read-only
recommendation, a cart mutation, and an order submission are separate actions with different
authorization boundaries.

## 7. Switch models without losing the journey

Keep the same conversation, switch **Model** to **Gemini 2.5 Pro**, and ask:

> Summarize my pickup plan, the stock issue we found, and what is in my cart.

The generation trace should name `gemini-2.5-pro`. Redis short-term session events retain the
shopping journey, and the cart tool supplies current cart state. Both model-specific ADK runners
share Agent Platform Sessions and ADK Memory Bank for persistence and measurement; Redis remains
the conversation context sent to Gemini.

Talk track: changing the reasoning model does not force the member to restart the shopping task or
discard grounded state.

## 8. Finish with the domain guardrail

Prompt:

> Where is Dagestan?

Expected behavior: the RedisVL Semantic Router blocks the out-of-domain request before LangCache,
memory retrieval, governed commerce tools, or Gemini run. Expand the trace to show the block and
the skipped downstream stages.

Talk track: routing is not only a cache optimization. It also prevents unrelated requests from
consuming model and retrieval work through this commerce agent.

## Close

Recap the customer outcome and its evidence:

- Context Retriever separated authoritative profile, order, order-item, and live inventory reads.
- Redis session memory carried the multi-turn journey forward.
- Redis Agent Memory personalized the recommendation while ADK Memory Bank provided an independent
  retrieval comparison.
- LangCache reused a safe, grounded policy answer but bypassed personalized and volatile turns.
- The cart changed only after an explicit instruction.
- The model could change without losing session state.
- The Semantic Router stopped an out-of-domain request before downstream work.

The closing message is that Vale did more than produce fluent shopping advice: it exposed the
source, freshness, authorization boundary, and latency of each decision in the journey.

## Reliable fallback prompt

If the live shopping path is interrupted, use:

> How long will Value Wholesale hold a pickup order after it is ready?

It is deterministic, non-personalized, and exercises policy grounding plus the semantic-cache
path.
