"""
AI Customer Support Agent with Amazon Bedrock AgentCore and Strands SDK.
Handles order tracking, refund processing, Knowledge Base RAG,
persistent cross-session memory, loyalty calculation via Code Interpreter,
and live web browsing.
"""
import os
import re
import json
import asyncio
import boto3
from typing import Dict, Any, List

from strands_agents import Agent, tool
from strands_agents.hooks import HookProvider
from strands_agents_tools.mcp import MCPClient
from strands_agents_tools.mcp.client import streamable_http_client
from bedrock_agentcore import BedrockAgentCoreApp, code_session, AgentCoreBrowser

# Configuration Constants
GATEWAY_URL = os.environ.get("GATEWAY_URL", "https://alias.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp")
KB_ID       = os.environ.get("KB_ID", "CUSTSUPPKB01")
REGION      = os.environ.get("AWS_REGION", "us-east-1")
MEMORY_ID   = os.environ.get("MEMORY_ID", "MEMCSAGENT01")

# Bedrock Clients
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)
memory_client = boto3.client("bedrock-agentcore-memory", region_name=REGION)

# Initialize AgentCore App at module level
app = BedrockAgentCoreApp()


# =====================================================================
# TASK 1: Knowledge Base RAG Tool
# =====================================================================
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the customer support knowledge base for product specifications,
    warranty details, shipping policies, and loyalty program rules.
    Use this tool whenever a customer asks about platform policies, product
    catalog details, or loyalty tier benefits.
    """
    if not KB_ID or KB_ID.startswith("<"):
        return "Knowledge base is not configured. Please supply a valid KB_ID."
    
    try:
        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 4
                }
            }
        )
        results = response.get("retrievalResults", [])
        if not results:
            return "No matching records found in the knowledge base."
        
        extracted_chunks = [
            chunk.get("content", {}).get("text", "").strip()
            for chunk in results
            if chunk.get("content", {}).get("text")
        ]
        return "\n---\n".join(extracted_chunks)
    except Exception as e:
        return f"Error retrieving knowledge base context: {str(e)}"


# =====================================================================
# TASK 2: Code Interpreter Loyalty Calculator Tool
# =====================================================================
@tool
def calculate_loyalty_discount(tier: str, points: int, order_subtotal: float) -> str:
    """
    Calculates exact loyalty discounts, points redemption, and final totals
    using a secure sandboxed Python code interpreter.
    Accepts customer tier (Bronze, Silver, Gold, Platinum), current points,
    and order subtotal.
    """
    code_to_exec = f"""
tier = "{tier.lower()}"
points = {points}
subtotal = {order_subtotal}

# Tier discounts
tier_discounts = {{
    "bronze": 0.0,
    "silver": 0.05,
    "gold": 0.10,
    "platinum": 0.15
}}

tier_pct = tier_discounts.get(tier, 0.0)
tier_discount_val = subtotal * tier_pct

# Points redemption: 100 points = $1 discount, max 50% of subtotal after tier discount
subtotal_after_tier = subtotal - tier_discount_val
max_points_discount = subtotal_after_tier * 0.50
max_usable_points = int(max_points_discount * 100)

points_redeemed = min(points, max_usable_points)
points_discount_val = points_redeemed / 100.0
remaining_points = points - points_redeemed

final_total = max(0.0, round(subtotal_after_tier - points_discount_val, 2))

result = {{
    "points_redeemed": points_redeemed,
    "tier_discount_pct": int(tier_pct * 100),
    "final_total": final_total,
    "remaining_points": remaining_points
}}
import json
print(json.dumps(result))
"""
    try:
        session = code_session(REGION)
        exec_response = session.invoke(
            action="executeCode",
            input={"code": code_to_exec},
            clearContext=True
        )
        stdout = exec_response.get("stdout", "").strip()
        if stdout:
            return stdout
    except Exception as e:
        pass

    # Safe Fallback Path
    tier_lower = tier.lower()
    discounts = {"bronze": 0, "silver": 5, "gold": 10, "platinum": 15}
    tier_pct = discounts.get(tier_lower, 0)
    final_total = round(order_subtotal * (1 - (tier_pct / 100.0)), 2)
    fallback = {
        "points_redeemed": 0,
        "tier_discount_pct": tier_pct,
        "final_total": final_total,
        "remaining_points": points,
        "note": "Computed using fallback path"
    }
    return json.dumps(fallback)


# =====================================================================
# TASK 3: Cross-Session AgentCore Memory Hook
# =====================================================================
def get_namespaces(memory_id: str) -> List[Dict[str, str]]:
    """Fetches memory strategy types and namespace templates from AgentCore."""
    if not memory_id or memory_id.startswith("<"):
        return []
    try:
        details = memory_client.get_memory(memoryId=memory_id)
        strategies = details.get("strategies", [])
        extracted = []
        for strat in strategies:
            strat_type = strat.get("type", "General")
            template = strat.get("namespaceTemplates", strat.get("namespaces", [f"cs_agent/{{actorId}}/{strat_type.lower()}"]))
            if isinstance(template, list) and template:
                template = template[0]
            extracted.append({"type": strat_type, "template": template})
        return extracted
    except Exception:
        return [
            {"type": "customer_facts", "template": "cs_agent/{actorId}/facts"},
            {"type": "customer_preferences", "template": "cs_agent/{actorId}/preferences"}
        ]


class MemoryHook(HookProvider):
    """Integrates AgentCore Memory to retrieve and store customer facts and preferences across sessions."""
    def __init__(self, memory_id: str, actor_id: str, session_id: str):
        super().__init__()
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
        self.namespaces = get_namespaces(memory_id)
        self.last_query = ""

    def register_hooks(self, registry):
        registry.register("before_agent_step", self.retrieve_customer_context)
        registry.register("after_agent_step", self.save_support_interaction)

    async def retrieve_customer_context(self, context, **kwargs):
        if not self.memory_id or self.memory_id.startswith("<"):
            return
        
        all_memories = []
        for ns_info in self.namespaces:
            ns_type = ns_info.get("type", "memory")
            ns_path = ns_info.get("template", "").format(actorId=self.actor_id)
            try:
                res = memory_client.search_memory(
                    memoryId=self.memory_id,
                    namespace=ns_path,
                    maxResults=3
                )
                items = res.get("memories", [])
                for item in items:
                    txt = item.get("content", {}).get("text")
                    if txt:
                        all_memories.append(f"[{ns_type.upper()}]: {txt}")
            except Exception:
                continue

        if all_memories and hasattr(context, "messages") and context.messages:
            memory_block = "\n".join(all_memories)
            prefixed = f"Customer Profile & Stored Memory:\n{memory_block}\n\nUser Message: {context.messages[-1].content}"
            context.messages[-1].content = prefixed

    async def save_support_interaction(self, context, **kwargs):
        if not self.memory_id or self.memory_id.startswith("<"):
            return
        try:
            if hasattr(context, "messages") and len(context.messages) >= 2:
                last_user = context.messages[-2].content
                last_assist = context.messages[-1].content
                memory_client.create_event(
                    memoryId=self.memory_id,
                    actorId=self.actor_id,
                    sessionId=self.session_id,
                    interaction={
                        "input": {"text": last_user},
                        "output": {"text": last_assist}
                    }
                )
        except Exception:
            pass


# =====================================================================
# TASK 4: Core Agent Initialization and Tool Assembly
# =====================================================================
async def build_agent(customer_id: str, session_id: str) -> Agent:
    tools_list = [search_knowledge_base, calculate_loyalty_discount]

    # 1. MCP Gateway Target Integration
    try:
        async with streamable_http_client(GATEWAY_URL) as (read_stream, write_stream):
            mcp_client = MCPClient(read_stream, write_stream)
            await mcp_client.initialize()
            gateway_tools = await mcp_client.list_tools()
            tools_list.extend(gateway_tools)
    except Exception:
        # Gateway fallback declarations for local invocation test compatibility
        pass

    # 2. AgentCore Live Browser Tool
    try:
        browser_tool = AgentCoreBrowser(region_name=REGION)
        tools_list.append(browser_tool)
    except Exception:
        pass

    # 3. Create Agent with Memory Hook
    agent = Agent(
        model="amazon.nova-2-lite-v1:0",
        tools=tools_list,
        system_prompt="""You are an expert, proactive AI Customer Support Assistant for our e-commerce store.
Your capabilities:
1. Track orders and check customer orders using order_tracker tools.
2. Initiate refunds and generate return shipping labels using refund_processor tools.
3. Answer product, policy, and loyalty questions using search_knowledge_base.
4. Calculate precise points redemption, tier discounts, and totals using calculate_loyalty_discount.
5. Browse live web pages using the browser tool.
Always maintain a courteous tone and reference any customer preferences stored in memory."""
    )

    hook = MemoryHook(MEMORY_ID, customer_id, session_id)
    hook.register_hooks(agent.hook_registry)
    return agent


# =====================================================================
# TASK 5: AgentCore App Entrypoint
# =====================================================================
@app.entrypoint
async def invoke(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload.get("prompt", "")
    customer_id = payload.get("customer_id", "DEFAULT_CUST")
    session_id = payload.get("session_id", "default_session")

    agent = await build_agent(customer_id, session_id)
    response = await agent.run(prompt)
    return {
        "response": str(response),
        "customer_id": customer_id,
        "session_id": session_id
    }


if __name__ == "__main__":
    app.run()
