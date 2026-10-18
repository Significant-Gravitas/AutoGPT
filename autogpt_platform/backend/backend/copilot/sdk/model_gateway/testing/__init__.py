"""Test support for the model gateway: a scripted upstream and a CLI driver.

Imported by the gateway's tests only. Nothing here reaches the network: the
upstream binds 127.0.0.1 on a free port and the CLI is pointed at the gateway.
"""
