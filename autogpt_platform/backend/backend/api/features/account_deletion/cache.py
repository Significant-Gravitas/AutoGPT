from backend.api.features.store.cache import forget_store_listings
from backend.data.user import forget_refusal


def invalidate_account_caches(user_id: str, *, listings_changed: bool) -> None:
    """Run both invalidations in one worker after the account change commits."""
    forget_refusal(user_id)
    if listings_changed:
        forget_store_listings()
