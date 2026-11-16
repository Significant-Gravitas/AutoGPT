"""One context, cost record and trace for background (non-chat) LLM calls.

The calls that go through this package today are the dream's three phases
(the sync calls, and the batch path's cost rows), the morning briefing's
lede, ``consult_teammate`` and the expert style eval's judge. Each builds an
``InferenceContext``, makes its call with ``complete.structured_complete``
inside ``trace.trace``, and records its cost with ``record.record`` (the
style judge excepted: the eval reports its own spend). The dream's batch
submission itself stays in ``dream/batch_submit.py``.

These background calls still route, account and trace on their own, and are
follow-ups for this package:

* scheduled block-description optimisation:
  ``copilot/optimize_blocks.py`` ``_optimize_descriptions`` (line 48);
* Graphiti extraction: the client ``copilot/graphiti/client.py``
  ``get_graphiti_client`` builds (line 273) and the episodes
  ``copilot/graphiti/ingest.py`` ``_ingestion_worker`` adds (line 308);
* Graphiti community maintenance: ``copilot/graphiti/flex_client.py``
  ``_create_structured_completion`` and ``_create_completion`` (lines 75
  and 93);
* asynchronous chat titles: ``copilot/service.py``
  ``_generate_session_title`` and ``_update_title_async`` (lines 855 and
  1060).
"""
