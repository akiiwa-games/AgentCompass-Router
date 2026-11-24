# Fast launch contracts

Verified 2026-10-30. Fast preserves the selected model and effort. It does not
switch Sonnet/Haiku to Opus. Pi has no implemented fast feature and no modeId.
The verified models are data: a model row in `accounts.json` declares
`fast_capable: true`, and fast runs only on the harness of the row's `api_family`
(`codex`; Claude has no fast contract, see below). An absent flag or `fast_capable: false` (also in the
effective local policy) means no fast. Automatic selection, explicit records and
launch revalidation read this same declaration (`fast_modes.refusal`: `<model> has no
verified fast mode on <harness>; run it without --fast`).

| Harness | Verified models | Paseo creation | CLI creation |
|---|---|---|---|
| Codex | rows with `api_family: codex`, `fast_capable: true` | features.service_tier = priority/default | -c features.fast_mode=true/false and -c service_tier="priority"/"default" |

Normal launches explicitly pass the off/default value to avoid inheriting saved
global preferences. Claude has no fast contract: Claude Code's fast mode spends paid usage credits outside the
subscription, which the product does not use (owner, 2026-11-24), so every Claude launch passes it off (Paseo
`features.fast_mode = false`, CLI `--settings '{"fastMode":false}'`). These settings are session-scoped; the router does not edit
user settings or billing. Continued Paseo sessions retain their chosen feature.
Provider feature state confirms requested configuration, not actual backend speed.
Fallback/cooldown/account restrictions can return standard speed even with the requested feature enabled.

A fast variant is its own candidate (`<bundle>+fast`), priced and timed from the model's cited fast-mode price and speed
multipliers (public data; only models with both are offered automatically). It competes on the same fit and quota economy as the normal
variant. A model the user names with `--fast` whose contract is verified but whose multipliers are not cited still launches
fast, with that data error, reserved at the plain mode's spend. API billing follows API pricing rather than the ChatGPT credit multiplier; the router's API-equivalent estimates are
not an invoice.

Local evidence: Codex CLI 0.165.0; installed Paseo advertises Codex service_tier.
Other installations need their own feature verification.

Sources: [OpenAI speed](https://learn.chatgpt.com/docs/agent-configuration/speed),
[OpenAI config schema](https://learn.chatgpt.com/docs/config-schema.json).
