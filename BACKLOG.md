# Deferred Technical Debt & Polish Backlog

This backlog tracks non-blocking improvements and technical debt identified during the Milestone 2 review. These items are intentionally deferred to prevent scope creep while keeping the M2 baseline frozen.

**Current Baseline**: M2 Frozen (`commit 5af31ad`)  
**Status**: Tracked / Non-blocking for M2

---

## 1. Provider Lifecycle / Caching
* **Description**: The current provider and client lifecycle uses lazy initialization but can be improved for session reuse, connection pooling, and multi-turn efficiency.
* **Resolution Plan**: Revisit when designing the M3 multi-agent orchestration and provider lifecycle.
* **Priority**: Medium (Deferred to M3 / Orchestration)

## 2. `_LazyProviderProxy` and Legacy Compatibility Shim Cleanup
* **Description**: `src/server.py` and `src/openai_client.py` contain legacy aliases and proxy shims (`openai_client`, `set_openai_client`, `_LazyProviderProxy`) maintained to ensure 100% backward compatibility with M1 tests.
* **Resolution Plan**: Re-evaluate and cleanly deprecate/remove during a future refactor milestone once all legacy callers and tests are updated.
* **Priority**: Low (Deferred)

## 3. Hard-coded Windows Project Paths in Tests & Scripts
* **Description**: Integration tests (`tests/test_mcp_stdio_integration.py`) and verification scripts (`scripts/verify_m2_*.py`) currently specify absolute Windows paths (e.g., `C:\Project\ai-agent-bridge`).
* **Resolution Plan**: Make tests and scripts cross-platform and environment-agnostic by dynamically resolving project root via `Path(__file__).resolve().parent...` and runtime Python executable via `sys.executable`.
* **Priority**: Medium (Deferred)

## 4. Gemini Model Documentation
* **Description**: `.env.example` and some documentation references specify `gemini-2.5-flash`, whereas Google API now recommends `gemini-3.8-flash` for current accounts.
* **Resolution Plan**: Update configuration templates and documentation examples to reflect currently supported and recommended models.
* **Priority**: Low (Deferred)

---

*Note: None of these items block M2 completion or M3 design. M2 remains CLOSED & FROZEN.*
