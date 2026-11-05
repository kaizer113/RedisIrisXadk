# Value Wholesale Iris GA presenter walkthrough

Use this sequence for the self-managed Value Wholesale application on port `8083`. Its isolated
runtime uses a bounded Gemini tool loop through Vertex AI. The default Value Wholesale,
Norling's, and Value Travel applications retain their existing behavior.

**Acceptance status, November 4, 2026:** the GA suite/license and GKE recovery remain deployment
prerequisites. These are intended demonstrations and verification checks, not claims that the
prompts have passed against live GA services. Product matches, tool names, inventory counts,
prices, scores, and latencies must come from the actual run.

Setup and operator commands are in [onprem-iris.md](onprem-iris.md).

## Before presenting

1. Obtain the VM's current external address after startup and open `http://VM_IP:8083`.
   Its address can change; do not reuse a historical IP from the preview deployment.
2. Select a synthetic member, such as Alex, and enable **Context Gateway**.
3. Refresh service status. Core probes must pass. **Gemini Configured** indicates client
   configuration; **Lens Initialized** indicates SDK setup. Exercise a real turn and inspect
   its export before presenting inference and Lens ingest as verified.
4. Open **Explore tools** to confirm the intended shopping surface. Search mode may return a
   discovered subset, not every tool from every upstream.
5. Have the operator publish the intended FAQs and comparison skill, apply evaluated routing
   intents, and disable result caching for current inventory, prices, orders, and balances.

The selector uses synthetic personas, not customer sign-in. The demonstration performs no real
purchase, payment, cancellation, or outbound message.

## 1. An approved answer on the first request

Ask:

> What is the electronics return policy?

Inspect **Routing decision**. Its intent must dispatch to `published_guidance`. When the
published FAQ clears the application's `0.85` score floor and the question is eligible, Vale
returns that approved answer directly. The response shows the entry key, version, and discovery
score. Its answer event reports `model_calls: 0`, with no Gemini-call step for that answer.
Memory still records the exchange.

Try the paraphrase:

> How long is the return window for electronics?

This retrieves an approved publication. It does not depend on first generating and caching an
answer. If either route or FAQ threshold is not met, explain the observed trace instead of
claiming that a direct answer occurred. Scores require evaluation with the deployed embedder.

For governance, have the operator prepare a draft revision and ask again before publication.
The serving revision should remain unchanged. Deliberate publication should make the new version
visible. Accepting a proposal creates a draft; publishing is a separate action. Runtime keys
cannot author or publish. Restore the intended demo publication after this exercise.

## 2. Current data through one Gateway

Ask:

> Find dark chocolate and check current inventory at my home warehouse.

Follow the trace: Intent Router returns a scored dispatch target; Playbooks discovers published
guidance; Agent Memory reads preferences and session context; Gemini discovers business tools,
then calls exact returned names and schemas through Context Gateway. Current commercial and
account facts must come from those live results. A failed inventory call means data is
unavailable, not that stock is zero. Displayed call times are measured client timings, not
server-side performance benchmarks.

Then ask:

> Show my recent orders and explain their current status.

Confirm the returned records belong to the selected member. The application binds supported
member arguments to that persona. Use **Explore tools** to show the actual schemas; do not
assume a tool name from the older deployment.

Disable **Context Gateway**, then ask about current inventory. The answer should explain that
the necessary live data is unavailable. It must not substitute a remembered fact, an approved
FAQ, or a stale business result. Re-enable the Gateway for the remaining steps.

## 3. A skill guides the comparison

With `shopping-comparison` published and discoverable, ask:

> Compare dark chocolate options for a family gathering and explain the tradeoffs.

The published menu contains the skill preview. If Gemini selects it, `load_published_skill`
loads the exact published entry from this turn's menu; look for **Load published skill**.
Its instructions guide the comparison while current product facts still come from business
tools. A skill grants no permission to access another member or perform an external action.

For large-result handling, request a broader comparison supported by the configured Gateway.
The agent can use `business_result_handle`, then `handle_operation` to describe, page, query,
count, aggregate, or release the result using the advertised helper schemas. Only handles
created in the current turn can be used. Verify the actual handle operations in the trace;
a long-looking answer alone does not demonstrate compression or paging.

## 4. A preference survives a fresh session

Ask:

> Remember that I prefer dark chocolate and avoid milk chocolate.

Look for **Remember shopping preference**, then open **Member memory**. The write uses the
member's owner and configured GA namespace. This explicit write does not require waiting for
background memory extraction.

Select another member and verify their memory inventory does not include the preference.
Select the original member again to create a fresh session, then ask:

> What chocolate preference do you remember for me? Use it to find suitable options.

Durable preferences can survive the new session; current prices and stock still need live
retrieval. Session continuity and long-term memory are separate capabilities.

## 5. A persistent shopping shortlist

Ask:

> Find suitable dark chocolate, check live stock, and add two of one available option to my cart.

The model must discover the SKU in this turn before `add_to_cart` accepts it. The cart stores
member-scoped SKU quantities. It is a shortlist, not a purchase, quote, or stock reservation.
Select the same member in another session and ask to view the cart. Verify persistence without
inferring a current price or availability from the saved quantities.

## 6. Disconnect while Vale works

Start a request requiring several business-tool calls. While events arrive, select
**Disconnect**, wait briefly, then select **Reconnect**.

The run ID stays the same and the cursor advances as retained events replay. Reconnect does
not submit another chat request. The producer can continue without a connected client, and
another serving worker can deliver the retained stream.

Reload during an active run to exercise saved run/cursor recovery within the same browser
session. If the event stream has expired, the application attempts a member-scoped snapshot,
retained for 24 hours by default. Stream TTLs are configured separately.

This recovers event delivery. Killing the worker executing Gemini can interrupt the computation;
reconnecting does not restart it. Do not describe this as agent execution failover or exactly-once
business operations.

## 7. Inspect the same turn in Lens

Copy the displayed trace ID and open the corresponding turn using the operator's Lens read
access. Compare recalled memories, prompt assembly, tool calls, Gemini spans, and outcome.
The application records Gemini spans explicitly; it does not rely on automatic Google-client
instrumentation.

Content capture defaults to off. Enable it only for the synthetic demonstration when captured
content is needed. Model-usage attributes contain only token counts returned by Gemini; missing
usage is not estimated. A displayed trace ID alone does not establish successful export.
Lens and Agent Mesh remain beta features.

## Keep the evidence

| Demonstration | Record from the actual run |
|---|---|
| Published FAQ | Entry key/version, score, intent target, and zero-model-call answer event when eligible. |
| Publication governance | Draft remained hidden; deliberately published revision became visible. |
| Business context | Discovered schemas, member binding, and current inventory/order results. |
| Skill and handles | Published skill version and the handle operations that actually ran. |
| Durable memory | Preference recovered in a fresh session for its member and absent for another. |
| Shortlist | Discovered SKUs persisted as quantities without a price/stock guarantee. |
| Mesh | Same run ID, advancing cursor, one producer request, and replay or explicit snapshot recovery. |
| Lens | Matching exported trace, capture configuration, and any returned model usage. |

Local tests include native Agent Mesh replay against temporary Redis 8.10.1 and two HTTP
replicas: disconnect/reconnect through the second replica, one producer, completed snapshot,
and denial to another member passed. A browser audit with mocked endpoints also passed cursor
and reload recovery. The existing project's lint and 86 tests passed. Deployed-network replay,
GA service data access, real Vertex inference, and Lens-ingest acceptance remain pending until
the GA deployment passes this checklist. Add actual outcomes to the shared Iris feature/migration
record. Historical preview timings and expected prices are not GA measurements.

Product references: [Agent Playbooks](https://redis.io/docs/latest/develop/ai/context-engine/agent-playbooks/),
[Intent Router](https://redis.io/docs/latest/develop/ai/context-engine/intent-router/),
[Context Gateway](https://redis.io/docs/latest/develop/ai/context-engine/context-gateway/),
[Lens](https://redis.io/docs/latest/develop/ai/context-engine/lens/), and
[Agent Mesh](https://redis.io/docs/latest/develop/ai/context-engine/agent-mesh/).
