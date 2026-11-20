"""The onboarding voice interview's relay: the ``voice-relay`` process.

The browser talks to Otto over one WebSocket; the relay holds the Gemini Live
session on the other side, so it is the only client the vendor sees. It
forwards audio both ways and owns every control of a call: the caps, the idle
cutoff, the scope guard, the spend cap, metering, re-attach after a dropped
socket, reconnecting a dropped vendor session, and draining on deploys.

The REST API's start route (``onboarding_dump.voice``) decides who may call
and mints the signed start token the relay trusts (:mod:`.tokens`).
"""
