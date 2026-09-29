# AGENTS.md

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Project
cabIQ is an end-to-end NYC taxi prediction system.

Primary goals:
- train and evaluate ML models on NYC taxi data
- expose predictions through an API
- keep data processing, model code, API code, and tests separated
- move toward production-quality engineering practices

## Working rules
- Make the smallest change required for the task.
- Do not refactor unrelated files.
- Preserve existing behavior unless the task explicitly changes it.
- Prefer existing dependencies over adding new ones.
- Never commit secrets, credentials, API keys, datasets, model artifacts, or local environment files.
- Explain architectural changes before making large structural changes.

## Before editing
- Inspect the relevant files and their dependencies.
- Follow the existing project structure and naming conventions.
- For architectural work, use Graphify when it materially helps understand dependencies.

## Python
- Use clear type hints where appropriate.
- Prefer small functions with single responsibilities.
- Keep business logic separate from API/router code.
- Keep model-loading logic separate from request handling.
- Avoid unnecessary abstractions.

## Testing
For behavior changes:
- add or update tests
- run the narrowest relevant test suite first
- run the broader test suite when the change affects shared components
- report any tests that were not run

## ML changes
When modifying model or feature logic:
- avoid data leakage
- keep training and inference preprocessing consistent
- document important feature changes
- evaluate against an appropriate validation/test set
- do not overwrite trained artifacts unless explicitly requested

## API changes
When modifying prediction APIs:
- preserve backwards compatibility unless explicitly changing the contract
- validate request inputs
- return clear error responses
- add/update API tests

## Generated/local files
Do not commit or modify generated local tooling output unless explicitly requested:
- .codex/
- graphify-out/
- .env
- __pycache__/
- .pytest_cache/
- model artifacts
- large datasets

## Verification
Before completing a coding task, report:
- what changed
- what was tested
- whether tests passed
- any remaining risks or follow-up work