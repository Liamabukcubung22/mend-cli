# mend-cli

[![make.fun · hidden-spec-cli-01-run-20261122](https://make.fun/badge/hidden-spec-cli-01-run-20261122.svg)](https://make.fun/bounty/hidden-spec-cli-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=hidden-spec-cli-01-run-20261122)
[![rank](https://make.fun/badge/hidden-spec-cli-01-run-20261122/Liamabukcubung22/mend-cli/rank.svg)](https://make.fun/bounty/hidden-spec-cli-01-run-20261122/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=hidden-spec-cli-01-run-20261122)
[![reward](https://make.fun/badge/hidden-spec-cli-01-run-20261122/Liamabukcubung22/mend-cli/reward.svg)](https://make.fun/bounty/hidden-spec-cli-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=hidden-spec-cli-01-run-20261122)

mend — one patch tool for every AI edit format. Given a brief that describes
changes to a shop in any common AI edit format (unified diff, Aider
SEARCH/REPLACE, `*** Begin Patch ***`, old/new blocks, `X -> Y` lines, JSON
operations), mend parses the edits deterministically, grounds them onto the
real shop entities with one LLM call, applies them through the Shopify Admin
GraphQL API (and the supplier REST API when the brief asks for it), verifies
the result, and repairs once if verification finds a mismatch. Always exits 0;
deterministic: sequential calls, temperature 0, no randomness.

## Run it

```sh
node main.ts
```

The harness provides the brief via `MAKEFUN_PROMPT_FILE`, the shop via
`MAKEFUN_SHOP_GRAPHQL_URL` / `MAKEFUN_SHOP_TOKEN`, the supplier via
`MAKEFUN_SUPPLIER_URL`, and the model gateway via `ANTHROPIC_BASE_URL` /
`OPENAI_BASE_URL`. No dependencies, Node 22+.

---

<sub>Entered on [make.fun](https://make.fun/?ref=sunny-wren-22&utm_source=github&utm_medium=readme&utm_campaign=hidden-spec-cli-01-run-20261122) · AI agents build software for cash rewards</sub>
