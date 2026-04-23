from backend.sdk import BlockCost, BlockCostType

# Ayrshare is a subscription proxy ($149/mo Business). Per-post credit charges
# prevent a single heavy user from absorbing the fixed cost and align with the
# upload cost of each post variant.
# cost_filter matches on is_video (default False in BaseAyrshareInput). First
# match wins in block_usage_cost, so list the video tier first.
AYRSHARE_POST_COSTS = (
    BlockCost(
        cost_amount=5, cost_type=BlockCostType.RUN, cost_filter={"is_video": True}
    ),
    BlockCost(
        cost_amount=2, cost_type=BlockCostType.RUN, cost_filter={"is_video": False}
    ),
)
