# API Request Examples

This folder contains runnable request examples for every currently exposed endpoint.

## Endpoints Covered

- GET /health
- POST /run-task
- POST /compare
- POST /leaderboard
- GET /leaderboard
- GET /recommend

## Payload Files

- `payloads/run-task.json`
- `payloads/compare.json`
- `payloads/leaderboard-prompt.json`
- `payloads/recommend-all.json`
- `payloads/recommend-live.json`
- `payloads/recommend-experiment.json`

## How To Use

### Option 1: VS Code REST Client

1. Install the `humao.rest-client` extension.
2. Open `examples/requests.http`.
3. Click `Send Request` above each endpoint block.

### Option 2: curl

```bash
curl -X POST http://localhost:8000/run-task -H "Content-Type: application/json" -d @examples/payloads/run-task.json
```

Replace path as needed for other payload files in `examples/payloads/`.

## Notes

- Default base URL in examples is `http://localhost:8000`.
- `aggregation` for leaderboard currently supports only `latest`.
- Recommendations can read from `live`, `experiment`, or `all` logs via the `source` query parameter.
- Recommendation payload files are query presets for `GET /recommend` (they are not request bodies).
