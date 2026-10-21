# Acme Cloud — API Rate Limits

## Default limits

| Plan | Requests / minute | Burst |
|---|---|---|
| Starter | 60 | 120 |
| Growth | 300 | 600 |
| Enterprise | 2000 | 4000 |

Rate-limit responses return HTTP 429 with a `Retry-After` header.

## Idempotency

Write endpoints accept an `Idempotency-Key` header. Keys are retained for 24 hours.

## Versioning

The current stable API version is `v1`.
Deprecated endpoints remain available for at least 12 months after the deprecation announcement.
Breaking changes are published in the changelog and emailed to account owners.
