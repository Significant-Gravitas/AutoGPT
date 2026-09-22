---
name: "enterprise-identity-sso-support"
description: "Use when logins fail at the identity layer — SAML, SSO, SCIM, OIDC, or API auth: triage the handshake, find the break, and verify every AI-drafted word before it ships."
triggers: ["SSO login failed", "SAML error", "SCIM provisioning", "OIDC claims", "can't log in with SSO", "certificate expired", "API token expired"]
version: "1"
---

# Enterprise identity SSO support

Use this when logins fail at the identity layer. Triage the handshake,
find the break, and verify every AI-drafted word before it ships.

## What you need first

The failure report with timestamps, the identity setup (the identity
provider or IdP, the app, and the SCIM user-provisioning path), recent
config changes, and the auth logs when available.

## Triage the handshake

1. Repro the failure on the customer's exact path first: IdP-initiated or
   SP-initiated, which app, which user. No repro, no diagnosis — collect
   the timestamp, the error string verbatim, and the affected users.
2. Walk the handshake in order: user exists and is licensed, group and role
   mapping, the SAML assertion or OIDC (OpenID Connect) claims, certificate
   and expiry, SCIM provisioning state, API token scope and expiry. Test one
   layer before moving to the next; name each layer FACT or UNKNOWN.
3. Rank the causes with a confirm step each: expired cert, mis-mapped
   attribute, deprovisioned user, clock skew, wrong audience address. The top
   cause gets the fix draft; the rest stay listed with their tests. Never
   edit IdP config yourself — draft the change for the owner's admin.
4. Verify every AI-drafted reply and KB step against the live setup before
   it goes out: quote the doc or log line behind each instruction, cut what
   you cannot verify. A login guide with an unverified step is worse than
   none.
5. Log the break, the fix, and the verify trail. Offer to mint the confirmed
   fix as a knowledge-base (KB) article for the next auth failure.

## Output

The ranked causes with confirm steps, the staged fix draft, the verified
customer reply, and the case log.

## When the records are thin

No log or admin access means an UNKNOWN layer with the access need named,
plus the questions for their IdP admin. Never guess at certs, claims, or
mappings.
