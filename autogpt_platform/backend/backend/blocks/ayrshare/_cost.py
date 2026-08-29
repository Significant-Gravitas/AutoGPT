from backend.sdk import BlockCost, BlockCostType

# Ayrshare is a subscription proxy ($149/mo Business). Per-post credit charges
# prevent a single heavy user from absorbing the fixed cost and align with the
# upload cost of each post variant.
# cost_filter matches on input_data.is_video BEFORE run() executes, so the flag
# has to be correct at input-eval time. Video-only platforms (YouTube, Snapchat)
# override the base default to True; platforms that accept both (TikTok, etc.)
# rely on the caller setting is_video explicitly for accurate billing.
# First match wins in block_usage_cost, so list the video tier first.
AYRSHARE_POST_COSTS = (
    BlockCost(
        cost_amount=5, cost_type=BlockCostType.RUN, cost_filter={"is_video": True}
    ),
    BlockCost(
        cost_amount=2, cost_type=BlockCostType.RUN, cost_filter={"is_video": False}
    ),
)
