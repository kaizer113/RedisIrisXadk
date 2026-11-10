"""Member shopping workflow over published guidance and live Gateway tools."""

import asyncio
import json
import time
from dataclasses import asdict
from typing import Any

from google.genai import types
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from redis_iris import lens

from .identity import Identity
from .services import (
    Services,
    account_tool,
    bounded_result,
    collect_skus,
    member_bound,
    owned_payload,
    published_direct_answer,
    tool_json,
)

SYSTEM = """You are Vale, the Value Wholesale shopping assistant in a fictional demo.
Help with product discovery, comparisons, live warehouse inventory, member orders, a shopping
shortlist, and preferences. No checkout, purchase, payment, cancellation or message is performed.
Use discover_business_tools and exact returned schemas to find live catalog, inventory, member
profile and order tools. Never invent a tool name, SKU, price, inventory count, reward balance,
order or member fact. Tool errors are unavailable data, never proof that an item is out of stock.
Always get mutable commercial/account data from live tools in this turn. Memories and published
guidance are not sources of current prices, inventory, orders or balances. The cart is a demo
shortlist of SKU quantities, never a quote. Add only products discovered in this turn.
Use remember_preference only when the member asks you to remember a durable shopping preference.
Never save payment information or operational account facts as memory. Handle tools let you page
large results; do not claim to have read omitted items. Published skills guide your work but do
not authorize tools, external effects, or another member's data. Answer only shopping questions.
"""


def declaration(
    name: str, description: str, properties: dict, required: tuple[str, ...] = ()
) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters_json_schema={
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
    )


LOCAL_TOOLS = [
    declaration("view_cart", "View this member's persistent demo shopping shortlist.", {}),
    declaration(
        "add_to_cart",
        "Set a discovered SKU's shortlist quantity; no purchase.",
        {"sku": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1, "maximum": 20}},
        ("sku", "quantity"),
    ),
    declaration(
        "remember_preference",
        "Save a durable preference explicitly requested by the member.",
        {"text": {"type": "string", "maxLength": 2000}},
        ("text",),
    ),
    declaration(
        "load_published_skill",
        "Load a skill from the published discovery menu by entry key.",
        {"key": {"type": "string"}},
        ("key",),
    ),
]
GATEWAY_TOOLS = [
    declaration(
        "discover_business_tools",
        "Discover read-only live business tools and input schemas.",
        {"query": {"type": "string"}},
        ("query",),
    ),
    declaration(
        "call_business_tool",
        "Call an exact discovered tool using its input schema.",
        {"name": {"type": "string"}, "arguments": {"type": "object"}},
        ("name", "arguments"),
    ),
    declaration(
        "business_result_handle",
        "Run a discovered tool and retain a large result as a handle.",
        {"name": {"type": "string"}, "arguments": {"type": "object"}},
        ("name", "arguments"),
    ),
    declaration(
        "handle_operation",
        "Inspect/page/query/count/aggregate/release a handle created here.",
        {
            "handle": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["page", "query", "count", "aggregate", "describe", "release"],
            },
            "arguments": {"type": "object"},
        },
        ("handle", "operation", "arguments"),
    ),
]


class Turn:
    def __init__(
        self,
        services: Services,
        identity: Identity,
        run_id: str,
        message: str,
        gateway_enabled: bool,
        emit: Any,
        model: str | None = None,
    ):
        self.services, self.identity, self.run_id = services, identity, run_id
        self.message, self.gateway_enabled, self.emit = message, gateway_enabled, emit
        self.model_name = model or services.settings.google_model
        if self.model_name not in services.settings.available_google_models:
            raise ValueError("The requested model is not configured for this demo")
        self.tools: dict[str, Any] = {}
        self.handles: set[str] = set()
        self.account_handles: set[str] = set()
        self.skus: set[str] = set()
        self.discovery = None
        self.model_calls = 0

    async def measured(self, service: str, title: str, operation: Any) -> Any:
        start = time.perf_counter()
        await self.emit(
            {"type": "trace", "step": {"service": service, "title": title, "status": "running"}}
        )
        try:
            result = await operation()
        except Exception:
            await self.emit(
                {
                    "type": "trace",
                    "step": {
                        "service": service,
                        "title": title,
                        "status": "error",
                        "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                        "detail": "Service call failed; no substitute answer was used.",
                    },
                }
            )
            raise
        failed_result = isinstance(result, dict) and result.get("ok") is False
        await self.emit(
            {
                "type": "trace",
                "step": {
                    "service": service,
                    "title": title,
                    "status": "error" if failed_result else "completed",
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    "detail": "Tool returned an error or failed member ownership checks."
                    if failed_result
                    else "",
                },
            }
        )
        return result

    async def call(self, name: str, arguments: dict) -> dict:
        services, identity = self.services, self.identity
        if name == "view_cart":
            return await services.cart(identity)
        if name == "add_to_cart":
            sku = str(arguments["sku"])
            if sku not in self.skus:
                return {"ok": False, "error": "Fetch this SKU from a live business tool first."}
            return await services.cart_add(identity, sku, int(arguments["quantity"]))
        if name == "remember_preference":
            return await self.measured(
                "agent_memory",
                "Remember shopping preference",
                lambda: services.memory.remember(identity, arguments["text"]),
            )
        if name == "load_published_skill":
            entry = next(
                (item for item in self.discovery.skills if item.key == arguments["key"]), None
            )
            if entry is None:
                return {"ok": False, "error": "Skill was not in this turn's published menu."}
            skill = await self.measured(
                "agent_playbooks", "Load published skill", lambda: services.playbooks.askill(entry)
            )
            return {"key": skill.key, "instructions": skill.as_prompt()}
        if not self.gateway_enabled or not services.gateway:
            return {"ok": False, "error": "Context Gateway is disabled for this turn."}
        if name == "discover_business_tools":
            tools = await self.measured(
                "context_gateway",
                "Discover business tools",
                lambda: services.tools(arguments["query"]),
            )
            self.tools.update({tool.name: tool for tool in tools})
            return {"tools": [tool_json(tool) for tool in tools]}
        if name in ("call_business_tool", "business_result_handle"):
            tool = self.tools.get(arguments.get("name"))
            if tool is None:
                return {"ok": False, "error": "Discover the exact tool before calling it."}
            from .services import member_arguments

            args = member_arguments(tool, arguments["arguments"], identity)
            validate(args, tool.input_schema)
            if name == "business_result_handle":
                if account_tool(tool) and not member_bound(tool):
                    return {
                        "ok": False,
                        "error": (
                            "Account result handles require an explicit forced member parameter."
                        ),
                    }
                handle = await self.measured(
                    "context_gateway",
                    "Retain large business result",
                    lambda: services.gateway.acall_as_handle(tool.name, args),
                )
                self.handles.add(handle.id)
                if account_tool(tool):
                    self.account_handles.add(handle.id)
                return asdict(handle)
            result = await self.measured(
                "context_gateway", tool.name, lambda: services.gateway_call(tool, args, identity)
            )
            if result["ok"]:
                self.skus.update(collect_skus(result["data"]))
            return bounded_result(result)
        if name == "handle_operation":
            if arguments["handle"] not in self.handles:
                return {"ok": False, "error": "Handle was not created in this turn."}
            operations = {
                "page": ("page_results", services.gateway.apage),
                "query": ("query_results", services.gateway.aquery),
                "count": ("count_results", services.gateway.acount),
                "aggregate": ("aggregate_results", services.gateway.aaggregate),
                "describe": ("describe_result", services.gateway.adescribe),
                "release": ("release_results", services.gateway.arelease),
            }
            operation = operations.get(arguments.get("operation"))
            if operation is None:
                return {"ok": False, "error": "Unknown handle operation"}
            helper, method = operation
            catalog = await services.gateway.atools()
            definition = next((tool for tool in catalog if tool.name == helper), None)
            if definition is None:
                return {"ok": False, "error": "Gateway does not advertise this handle operation"}
            args = dict(arguments["arguments"])
            args["handle"] = arguments["handle"]
            validate(args, definition.input_schema)
            args.pop("handle")
            result = await self.measured(
                "context_gateway", helper, lambda: method(arguments["handle"], **args)
            )
            if arguments["handle"] in self.account_handles:
                if result.is_error or not owned_payload(result.data, identity, constrained=True):
                    return {"ok": False, "error": "Account result did not match this member."}
                return bounded_result({"ok": True, "data": result.data, "text": ""})
            if not result.is_error:
                self.skus.update(collect_skus(result.data))
            return bounded_result(
                {"ok": not result.is_error, "data": result.data, "text": result.text}
            )
        return {"ok": False, "error": "Unknown tool"}

    async def generate(self, memories: list[dict], transcript: list[dict]) -> str:
        services = self.services
        if services.model is None:
            raise RuntimeError("Configure Vertex AI before starting a shopping turn")
        fetched = [{"id": row["memory_id"], "text": row["text"]} for row in memories]
        fetched.extend(
            {
                "id": f"{entry.entry_id}:v{entry.version}",
                "text": entry.as_prompt(),
                "type": entry.entry_type,
            }
            for entry in self.discovery.entries
        )
        guidance = self.discovery.as_prompt()
        with lens.prompt_assembly() as span:
            span.fetched(fetched)
            span.used(fetched)
            system = (
                SYSTEM
                + f"\nCurrent synthetic member ID: {self.identity.member_id}\n"
                + "Remembered preferences:\n"
                + json.dumps(memories)
                + "\nPublished guidance:\n"
                + guidance
            )
        contents = [
            types.Content(
                role="model" if item["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=item["text"])],
            )
            for item in transcript
        ]
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=self.message)]))
        declarations = [*LOCAL_TOOLS, *(GATEWAY_TOOLS if self.gateway_enabled else [])]
        by_name = {tool.name: tool for tool in declarations}
        for _ in range(services.settings.onprem_max_tool_rounds):
            with lens.prompt_assembly() as assembly:
                assembly.fetched(fetched)
                assembly.used(fetched)
            with lens.span(
                "gemini.generate_content",
                **{
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "gcp.vertex_ai",
                    "gen_ai.request.model": self.model_name,
                },
            ) as model_span:

                async def invoke():
                    self.model_calls += 1
                    return await services.model.aio.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=system,
                            tools=[types.Tool(function_declarations=declarations)],
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                disable=True
                            ),
                            max_output_tokens=4096,
                        ),
                    )

                response = await self.measured("gemini", "Gemini shopping reasoning", invoke)
                usage = response.usage_metadata
                if usage:
                    if usage.prompt_token_count is not None:
                        model_span.set_attribute(
                            "gen_ai.usage.input_tokens", usage.prompt_token_count
                        )
                    if usage.candidates_token_count is not None:
                        model_span.set_attribute(
                            "gen_ai.usage.output_tokens", usage.candidates_token_count
                        )
                if services.settings.iris_lens_capture_content:
                    model_span.set_attribute(
                        "gen_ai.input.messages",
                        json.dumps(
                            [
                                {"role": "system", "content": system},
                                *[
                                    item.model_dump(mode="json", exclude_none=True)
                                    for item in contents
                                ],
                            ]
                        ),
                    )
                    model_span.set_attribute("gen_ai.output.messages", response.text or "")
            if not response.candidates or response.candidates[0].content is None:
                raise RuntimeError("Gemini returned no usable content")
            content = response.candidates[0].content
            contents.append(content)
            calls = response.function_calls or []
            if not calls:
                if not response.text:
                    raise RuntimeError("Gemini returned an empty answer")
                return response.text
            results = []
            for call in calls:
                definition = by_name.get(call.name)
                if definition is None:
                    result = {"ok": False, "error": "Tool is not available"}
                else:
                    try:
                        args = dict(call.args or {})
                        validate(args, definition.parameters_json_schema)
                        with lens.span("shopping.tool", **{"gen_ai.tool.name": call.name}):
                            result = await self.call(call.name, args)
                    except (ValueError, KeyError, TypeError, PermissionError, ValidationError):
                        result = {"ok": False, "error": "Tool arguments are invalid"}
                results.append(types.Part.from_function_response(name=call.name, response=result))
                fetched.append(
                    {
                        "id": f"tool:{call.name}:{len(contents)}:{len(results)}",
                        "text": json.dumps(result),
                        "type": "tool",
                    }
                )
            contents.append(types.Content(role="user", parts=results))
        raise RuntimeError("Tool round limit reached; narrow the shopping request")

    async def run(self) -> None:
        services, identity = self.services, self.identity
        with lens.turn(identity.session_id, user_id=identity.owner) as turn:
            try:
                async with asyncio.timeout(services.settings.onprem_turn_timeout):
                    matches = await self.measured(
                        "intent_router",
                        "Classify shopping intent",
                        lambda: services.router.amatch(self.message),
                    )
                    match = matches.best
                    await self.emit(
                        {
                            "type": "trace",
                            "step": {
                                "service": "intent_router",
                                "title": "Routing decision",
                                "status": "completed",
                                "detail": {
                                    "intent": match.name,
                                    "score": match.score,
                                    "target": asdict(match.target) if match.target else None,
                                }
                                if match
                                else {
                                    "intent": None,
                                    "reason": "No intent cleared the configured threshold",
                                },
                            },
                        }
                    )
                    if match is None or (match.target and match.target.ref == "refuse"):
                        answer = (
                            "I can help with Value Wholesale shopping, products, "
                            "orders and policies."
                        )
                        turn.set_outcome("refused")
                        await self.emit({"type": "answer", "answer": answer, "model_calls": 0})
                        return
                    self.discovery = await self.measured(
                        "agent_playbooks",
                        "Discover published guidance",
                        lambda: services.playbooks.adiscover(self.message, limit=5),
                    )
                    direct = published_direct_answer(self.message, match, self.discovery)
                    transcript = await self.measured(
                        "agent_memory",
                        "Read member session",
                        lambda: services.memory.transcript(identity),
                    )
                    await self.measured(
                        "agent_memory",
                        "Append member message",
                        lambda: services.memory.append(identity, self.run_id, "user", self.message),
                    )
                    if direct is not None:
                        answer = direct.faq.answer
                        provenance = {
                            "entry_id": direct.entry_id,
                            "key": direct.key,
                            "version": direct.version,
                            "score": direct.score,
                        }
                    else:
                        memories = await self.measured(
                            "agent_memory",
                            "Recall member preferences",
                            lambda: services.memory.recall(identity, self.message),
                        )
                        answer = await self.generate(memories, transcript)
                        provenance = None
                    await self.measured(
                        "agent_memory",
                        "Append assistant answer",
                        lambda: services.memory.append(identity, self.run_id, "assistant", answer),
                    )
                    await self.emit(
                        {
                            "type": "answer",
                            "answer": answer,
                            "published": provenance,
                            "model_calls": self.model_calls,
                        }
                    )
                    turn.set_outcome("completed")
            except TimeoutError:
                turn.set_outcome("timeout")
                raise
            except Exception:
                turn.set_outcome("failed")
                raise
            finally:
                if turn.trace_id:
                    await self.emit({"type": "lens", "trace_id": turn.trace_id})
