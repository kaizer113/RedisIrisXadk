"""Context Retriever entity model for the Value Wholesale shopping demo."""

from __future__ import annotations

from typing import Any

from context_surfaces.context_model import ContextField, ContextModel, ContextRelationship


class Product(ContextModel):
    __redis_key_template__ = "valueharbor:context:product:{sku}"

    sku: str = ContextField(description="Unique product SKU", is_key_component=True)
    name: str = ContextField(description="Product name", index="text", weight=2.0)
    category: str = ContextField(description="Product category", index="tag")
    price: float = ContextField(description="Regular price", index="numeric", sortable=True)
    member_price: float = ContextField(
        description="Value Wholesale member price", index="numeric", sortable=True
    )
    description: str = ContextField(description="Product description", index="text")
    tags: list[str] = ContextField(description="Product discovery tags", index="tag")

    inventory: Any = ContextRelationship(
        description="Warehouse inventory for this product",
        target="Inventory",
        source_field="sku",
    )
    order_items: Any = ContextRelationship(
        description="Order lines containing this product",
        target="OrderItem",
        source_field="sku",
    )


class Warehouse(ContextModel):
    __redis_key_template__ = "valueharbor:context:warehouse:{warehouse_id}"

    warehouse_id: str = ContextField(description="Warehouse identifier", is_key_component=True)
    name: str = ContextField(description="Warehouse name", index="text", weight=2.0)
    city: str = ContextField(description="Warehouse city", index="tag")
    state: str = ContextField(description="US state code", index="tag")

    inventory: Any = ContextRelationship(
        description="Products stocked at this warehouse",
        target="Inventory",
        source_field="warehouse_id",
    )
    orders: Any = ContextRelationship(
        description="Orders fulfilled by this warehouse",
        target="Order",
        source_field="warehouse_id",
    )


class Inventory(ContextModel):
    # Separate hash records preserve the application's O(1) String inventory keys.
    __redis_key_template__ = "valueharbor:context:inventory:{inventory_id}"

    inventory_id: str = ContextField(description="Inventory record ID", is_key_component=True)
    warehouse_id: str = ContextField(description="Warehouse identifier", index="tag")
    sku: str = ContextField(description="Product SKU", index="tag")
    quantity: int = ContextField(description="Available units", index="numeric", sortable=True)
    updated_at: str = ContextField(description="Inventory update timestamp", index="tag")

    product: Any = ContextRelationship(
        description="Product represented by this stock record",
        target="Product",
        source_field="sku",
    )
    warehouse: Any = ContextRelationship(
        description="Warehouse holding this stock",
        target="Warehouse",
        source_field="warehouse_id",
    )


class Member(ContextModel):
    __redis_key_template__ = "valueharbor:context:member:{member_id}"

    member_id: str = ContextField(description="Member identifier", is_key_component=True)
    name: str = ContextField(description="Member name", index="text", weight=2.0)
    tier: str = ContextField(description="Membership tier", index="tag")
    home_warehouse: str = ContextField(description="Preferred warehouse", index="tag")
    reward_balance: float = ContextField(
        description="Current reward balance", index="numeric", sortable=True
    )
    joined_at: str = ContextField(description="Membership start date", index="tag")

    orders: Any = ContextRelationship(
        description="Orders placed by this member",
        target="Order",
        source_field="member_id",
    )
    warehouse: Any = ContextRelationship(
        description="Member's preferred warehouse",
        target="Warehouse",
        source_field="home_warehouse",
    )


class Order(ContextModel):
    __redis_key_template__ = "valueharbor:context:order:{order_id}"

    order_id: str = ContextField(description="Order identifier", is_key_component=True)
    member_id: str = ContextField(description="Member who placed the order", index="tag")
    status: str = ContextField(description="Current order status", index="tag")
    warehouse: str = ContextField(description="Fulfillment warehouse", index="tag")
    fulfillment: str = ContextField(description="Fulfillment method", index="tag")
    placed_at: str = ContextField(description="Order date", index="tag")
    total: float = ContextField(description="Order total", index="numeric", sortable=True)
    item_count: int = ContextField(description="Number of order lines", index="numeric")

    items: Any = ContextRelationship(
        description="Line items in this order",
        target="OrderItem",
        source_field="order_id",
    )
    member: Any = ContextRelationship(
        description="Member who placed this order",
        target="Member",
        source_field="member_id",
    )


class OrderItem(ContextModel):
    __redis_key_template__ = "valueharbor:context:order-item:{order_item_id}"

    order_item_id: str = ContextField(description="Order-line identifier", is_key_component=True)
    order_id: str = ContextField(description="Parent order", index="tag")
    line_number: int = ContextField(description="Line number", index="numeric")
    sku: str = ContextField(description="Product SKU", index="tag")
    product_name: str = ContextField(description="Product name at purchase", index="text")
    quantity: int = ContextField(description="Purchased quantity", index="numeric")
    unit_price: float = ContextField(description="Unit price", index="numeric")

    order: Any = ContextRelationship(
        description="Parent order",
        target="Order",
        source_field="order_id",
    )
    product: Any = ContextRelationship(
        description="Purchased product",
        target="Product",
        source_field="sku",
    )


class Policy(ContextModel):
    __redis_key_template__ = "valueharbor:context:policy:{id}"

    id: str = ContextField(description="Policy identifier", is_key_component=True)
    title: str = ContextField(description="Policy title", index="text", weight=2.0)
    content: str = ContextField(description="Authoritative policy text", index="text")
