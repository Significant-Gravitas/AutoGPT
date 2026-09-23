---
name: "logistics-shipment-and-customs"
description: "Use when goods are moving: trace and expedite shipments, file customs entries, and liaise with carriers and warehouses."
triggers: ["track this shipment", "late delivery", "customs entry", "HTS code", "carrier escalation", "delivery ETA", "warehouse escalation"]
version: "1"
---

# Logistics shipment and customs

Use this when goods are moving: trace and expedite shipments, file
customs entries, and liaise with carriers and warehouses.

You need the tracking or order IDs, the carrier and warehouse portals
from prefs, the commercial invoice for customs, and the delivery
promise.

## Trace first

Pull the live status from the carrier, name the last scan with its
time, and say whether the promise still holds. A trace with no carrier
read is UNKNOWN, never "on the truck".

## Expedite stalled or late freight

Work in order: carrier escalation with the reference IDs, reroute or
reschedule, then the customer draft with the new ETA and its source.
Never promise a date the carrier has not given.

## Classify and file customs by the book

File each line item with its HTS code — the Harmonized Tariff Schedule
classification — plus value and origin from the invoice, through the
broker or portal. A classification you cannot
source goes to the broker — never guess a duty code.

## Liaise in one thread per shipment

Keep carriers, forwarders, and warehouses to one thread per shipment:
the ask, the reference IDs, the needed-by time. Log every commitment
with who made it; chase once past the time, then escalate with the
paper trail.

Deliver the trace read, the expedite or entry packet, the staged
customer draft, and the shipment log. Offer to flag the lane or
carrier pattern when failures repeat.

## What not to do

No portal access means tracing from the customer's screenshots, marked
INFERENCE, with the access need named. Never invent scan events or
entry numbers.
