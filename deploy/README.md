# Deploying the API

The image contains only the API, the trained bundle (~3 MB), CPU-only PyTorch and the demo page: no raw data, no
notebook dependencies. It runs as a non-root user and has a Docker health check.

## Local

```bash
docker compose up --build          # http://127.0.0.1:8090  (demo page at /, API docs at /docs)
```

## A small VPS with HTTPS

1. Create an Ubuntu server (2 GB RAM is comfortable: PyTorch plus a day's graph in memory) and point a DNS `A`
   record, e.g. `aml.example.com`, at its IP.
2. Install Docker and clone the repository:

   ```bash
   curl -fsSL https://get.docker.com | sh
   git clone <your repo url> graphaml && cd graphaml
   ```

3. Start the API behind Caddy, which obtains a Let's Encrypt certificate on first request:

   ```bash
   export DOMAIN=aml.example.com
   export AML_API_KEY=$(openssl rand -hex 24)   # keep this; clients send it as X-API-Key
   docker compose -f deploy/docker-compose.prod.yml up -d --build
   ```

4. Smoke test:

   ```bash
   curl https://aml.example.com/health
   curl -X POST https://aml.example.com/alerts -H "X-API-Key: $AML_API_KEY" -F file=@examples/transactions_example.csv
   ```

Only ports 80 and 443 need to be open (`ufw allow 80,443/tcp`). `/health`, `/docs`, `/demo/*` and the demo page are
public (the demo only ever generates synthetic data); `/alerts`, `/score/batch` and `/drift` require the key.

## Updating the model

Train on IBM data in Colab (notebook step 11 downloads `graphaml_models.zip`), unzip it into the repository, then:

```bash
pytest -q                             # includes the training/serving skew test
git commit -am "retrain on IBM HI-Small" && git push
# on the server:
git pull && docker compose -f deploy/docker-compose.prod.yml up -d --build
```

## Operating it

- **Every morning:** send yesterday's transactions together with enough history (the model uses each account's
  activity, so 30+ days is a good window) to `POST /alerts?score_from=<yesterday>`. The response is the alert queue,
  highest risk first, with the reasons for each alert.
- **Weekly:** `POST /drift` with a recent batch. `overall: "drift"` means payments no longer look like the training
  period (new product, new payment rail, a merger); check alert precision with the analysts before trusting the ranking.
- **Feedback loop:** analysts' decisions on alerts (true / false positive) are new labels. Retrain monthly on them;
  keep a small random sample of non-alerted transactions reviewed as well, otherwise the model only learns from its own
  choices.
