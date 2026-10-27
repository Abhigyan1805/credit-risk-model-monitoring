"""AI-drafted, checker-verified monthly monitoring commentary.

A local open-weight LLM drafts an analyst-style commentary from one month's monitoring results;
``commentary.checker`` verifies every number, metric, month, direction word and RAG status against
those results and blocks the draft if anything is unsupported. A blocked draft is retried with the
checker's feedback, then replaced by a deterministic template. See ``docs/AI_COMMENTARY.md``.
"""
