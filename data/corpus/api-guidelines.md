# Internal API Guidelines

## Resource naming

URLs are lowercase, hyphen-separated, and plural for collections:
`/v1/accounts`, `/v1/payment-orders`. A single resource is addressed by its
identifier: `/v1/accounts/{account_id}`. Verbs do not appear in paths; the HTTP
method carries the intent.

## Versioning

The major version is part of the path (`/v1/...`). A new major version is
introduced only for breaking changes. Additive changes — a new optional field, a
new endpoint — do not require a version bump. Deprecated fields are announced at
least two releases before removal and marked in the OpenAPI description.

## Status codes

Use the standard meanings: 200 for a successful read, 201 for a created resource,
204 for a successful delete with no body, 400 for a malformed request, 401 when
credentials are missing or invalid, 403 when the caller is authenticated but not
permitted, 404 for an unknown resource, 409 for a state conflict, 422 for a
well-formed request that fails validation, and 429 when a rate limit is exceeded.

## Rate limiting

Rate limits are enforced per API key, not per IP address. The default limit is
600 requests per minute with a burst allowance of 100. Responses include
`X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`. A 429
response must include a `Retry-After` header in seconds.

## Pagination

Collections are paginated with a cursor, not an offset. The request accepts
`limit` (default 50, maximum 200) and `cursor`. The response contains `data`,
`next_cursor` (null on the final page), and `has_more`. Offset pagination is
prohibited because it produces duplicates and gaps when rows are inserted
concurrently.

## Idempotency

Any endpoint that creates a financial resource must accept an
`Idempotency-Key` header. Replaying a request with the same key and the same body
returns the original result rather than creating a duplicate. Keys are honoured
for 24 hours.

## Errors

Errors use a single envelope:

```json
{
  "error": {
    "code": "insufficient_funds",
    "message": "The source account has insufficient available balance.",
    "request_id": "req_8f2c1a",
    "details": []
  }
}
```

`code` is a stable machine-readable string; clients must branch on `code`, never
on `message`, because messages may be reworded or translated.
