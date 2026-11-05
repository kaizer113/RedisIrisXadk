---
name: shopping-comparison
description: Compare Value Wholesale products for a member using current catalog and inventory tools, remembered preferences, and clear tradeoffs. Use for product comparisons, family gatherings, and bulk shopping decisions.
---

# Compare shopping options with current evidence

1. Establish the member's goal, constraints, quantity, and preferred warehouse. Ask a focused question if a necessary detail is missing.
2. Use `discover_business_tools` to discover relevant read-only catalog, member-profile, and inventory tools. Call only exact discovered names with their returned input schemas through `call_business_tool`.
3. Fetch current candidate products and stock in this turn. Use durable memories only for preferences. Use the current member profile for operational facts such as warehouse and membership. Published policy guidance supplies general rules, never current prices or account balances.
4. Compare a small set of relevant candidates. State only product attributes, quantities, prices, and stock actually returned by live tools. Explain which tradeoff matters for the member's stated goal. An upstream error means data is unavailable, not that stock is zero.
5. If a result is too large, retain it with `business_result_handle`. Inspect the advertised helper schemas and use `handle_operation` to page, query, count, aggregate, describe, or release the result. Distinguish items read from items not yet inspected. Release temporary handles when finished.
6. On an explicit request to add an option, use `add_to_cart` only for a SKU discovered in this turn. The cart is a shortlist of SKU quantities; it is not a purchase, quote, payment, stock reservation, or fulfillment commitment.
7. Save a durable shopping preference with `remember_preference` only when the member explicitly asks. Do not store prices, inventory, order status, balances, or payment information as memory.

Explain the recommendation and its evidence in plain language. If no supported option can be established, state what is missing instead of inventing a result. This skill grants no additional permissions and authorizes no external action.
