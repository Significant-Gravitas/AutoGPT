---
name: "live-channel-queue-operations"
description: "Use when the phones ring, chat piles up, or social mentions burn: work the live queue in order, keep talk tracks short, and move public heat to private."
triggers: ["work the live queue", "chat wait is spiking", "phone queue backlog", "overdue callback list", "move this to private", "shift talk track", "live queue sweep"]
version: "1"
---

# Live-channel queue operations

Use this when the phones ring, chat piles up, or social mentions burn.
Work the live queue in order, keep talk tracks short, and move public
heat to private.

## What you need first

The live queue state (chat, phone, social), promised callbacks with due
times, wait-time and SLA clocks, and the talk track for the shift.

## Sweep, order, work, hand off

1. Sweep every live surface for waits, overdue callbacks, and hot
   sentiment, oldest promise first.
2. Order the work list by promise age, then SLA risk, then channel heat,
   and name what stays queued.
3. Work each item with a short call plan or talk track: ack, next step
   with date and owner, one question at a time.
4. Move public heat to private with the account-detail ask, log each
   touch, and hand off unworked items with owner and time.

## Output

The ordered call and queue list with plans, plus drafts for the first
callbacks. Nothing dialed or posted without owner yes.

## When the records are thin

Thin records mean a shorter confident list, not padded queues. Say which
record would settle each UNKNOWN.
