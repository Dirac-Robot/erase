# ERASE: Explicit Retention And Selective Erasure

A structured memory system for LLM summarization that explicitly models **what to remember** and **what to forget**.

## Concept

Traditional summarization focuses only on "what to keep." ERASE introduces a **dual-scored** approach:

- **Retention Score**: How important is this information to remember?
- **Erasure Score**: How safe is it to forget this information?

By explicitly modeling both dimensions, ERASE enables more nuanced, structured memory management for long-context LLM applications.
