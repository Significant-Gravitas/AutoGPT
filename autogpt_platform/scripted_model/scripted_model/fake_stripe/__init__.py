"""A stateful fake of the Stripe API for journeys.

The platform's Stripe SDK calls it instead of api.stripe.com
(``STRIPE_API_BASE``), a browser pays on its hosted checkout page, and it
sends the platform signed webhooks. ``fake.py`` routes; README.md lists what
is served.
"""
