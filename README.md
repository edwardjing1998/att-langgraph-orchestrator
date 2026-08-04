# AT&T-style LangGraph AI Search Orchestrator

A learning project that coordinates three governed search tools:

1. Databricks policy/knowledge AI Search
2. Databricks billing-knowledge AI Search
3. Snowflake Cortex Search for curated billing transactions

The application is a regular Python/FastAPI service. It is not a PySpark job.
LangGraph manages request-time routing, retrieval, evidence normalization, and
answer generation.

## Architecture

```text
Client
  -> FastAPI /api/v1/ask
  -> LangGraph classify
  -> selected search tools (parallel for mixed questions)
  -> normalized evidence
  -> optional Databricks LLM synthesis
  -> cited response
```

## Project structure

```text
src/att_agent/
  api.py                 FastAPI routes and API-key protection
  config.py              Environment configuration
  graph.py               LangGraph workflow
  models.py              Request, response, state and evidence models
  routing.py             Deterministic initial router
  clients/
    databricks.py        Databricks AI Search and model-serving clients
    snowflake.py         Snowflake Cortex Search client
openshift/
  buildconfig.yaml       OpenShift internal image build
  deployment.yaml        Runtime deployment template
  service.yaml           Cluster service
  route.yaml             External HTTPS route
.github/workflows/
  deploy-openshift.yml   Tests, builds and deploys
```

## Prerequisites

- Python 3.12+
- A working Databricks workspace and token
- A working Snowflake PAT
- The three indexes/services already deployed
- An OpenShift project/namespace
- `oc` access for manual testing (GitHub installs `oc` automatically)

## Existing search services

```text
Databricks policy index:
workspace.staging_ai.ai_knowledge_index

Databricks billing index:
workspace.staging_ai.billing_knowledge_index

Snowflake Cortex Search:
EDU_AI_APP.AI.BILLING_CORTEX_SEARCH
```

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Copy the non-secret example configuration:

```bash
cp .env.example .env
```

Export secrets without writing them to `.env`:

```bash
read -s "DATABRICKS_TOKEN?Databricks token: "
export DATABRICKS_TOKEN
echo

read -s "SNOWFLAKE_PAT?Snowflake PAT: "
export SNOWFLAKE_PAT
echo
```

Load the non-secret variables from `.env`:

```bash
set -a
source .env
set +a
```

Run the service:

```bash
uvicorn att_agent.api:app --host 0.0.0.0 --port 8080 --reload
```

Test it:

```bash
curl --fail-with-body \
  --request POST \
  --url "http://localhost:8080/api/v1/ask" \
  --header "Content-Type: application/json" \
  --data '{
    "question": "I was charged for roaming. Explain the charge and tell me whether it can be reversed.",
    "billingAccountToken": "tok_demo_sas_003",
    "maxResultsPerSource": 3
  }'
```

If `APP_API_KEY` is configured, include:

```bash
--header "X-API-Key: $APP_API_KEY"
```

## Run tests and quality checks

```bash
ruff check src tests
pytest -q
```

## GitHub environment

Create a GitHub environment named `staging`:

```text
Repository -> Settings -> Environments -> New environment -> staging
```

Add these environment secrets:

```text
OPENSHIFT_SERVER
OPENSHIFT_TOKEN
OPENSHIFT_NAMESPACE
DATABRICKS_TOKEN
SNOWFLAKE_PAT
APP_API_KEY                 optional but recommended
```

Add these environment variables:

```text
DATABRICKS_HOST=https://dbc-3a95c916-417d.cloud.databricks.com
DATABRICKS_POLICY_INDEX=workspace.staging_ai.ai_knowledge_index
DATABRICKS_BILLING_INDEX=workspace.staging_ai.billing_knowledge_index
DATABRICKS_CHAT_ENDPOINT=databricks-gpt-oss-120b
SNOWFLAKE_HOST=https://usbxmys-cm03603.snowflakecomputing.com
SNOWFLAKE_CORTEX_DATABASE=EDU_AI_APP
SNOWFLAKE_CORTEX_SCHEMA=AI
SNOWFLAKE_CORTEX_SERVICE=BILLING_CORTEX_SEARCH
LLM_ENABLED=true
```

If the model-serving endpoint is unavailable, set:

```text
LLM_ENABLED=false
```

The API will then return the retrieved evidence with an extractive fallback
answer instead of calling the LLM.

## Deploy from GitHub

Push to `main`, or manually run:

```text
Actions -> Deploy LangGraph Orchestrator to OpenShift -> Run workflow
```

The pipeline:

1. Installs Python dependencies.
2. Runs Ruff and pytest.
3. Installs the OpenShift CLI.
4. Logs into the OpenShift namespace.
5. Creates/updates the runtime secret.
6. Builds the container inside OpenShift.
7. Applies the Deployment, Service and Route.
8. Waits for rollout and smoke-tests `/health/ready`.

## Verify on OpenShift

```bash
oc get pods
oc get deployment att-langgraph-orchestrator
oc get route att-langgraph-orchestrator
oc logs deployment/att-langgraph-orchestrator --tail=100
```

Get the route:

```bash
export APP_URL="https://$(oc get route att-langgraph-orchestrator -o jsonpath='{.spec.host}')"
```

Health check:

```bash
curl --fail-with-body "$APP_URL/health/ready"
```

Ask a mixed question:

```bash
curl --fail-with-body \
  --request POST \
  --url "$APP_URL/api/v1/ask" \
  --header "Content-Type: application/json" \
  --header "X-API-Key: $APP_API_KEY" \
  --data '{
    "question": "I was charged for roaming. Explain it and determine whether the charge may be reversed.",
    "billingAccountToken": "tok_demo_sas_003",
    "maxResultsPerSource": 3
  }'
```

## Security notes

- Never commit Databricks or Snowflake tokens.
- Use service identities, OAuth/workload identity, and an enterprise secret
  manager in production.
- Do not trust a billing-account token supplied by an external client. A real
  gateway must derive it from the authenticated user's entitlements.
- Cortex Search uses owner's-rights behavior; application authorization must be
  enforced before retrieval.
- Keep PII out of prompts, logs, graph state, and AI indexes.
- The deterministic router is intentionally auditable. Add LLM-based routing
  only after evaluation data and fallback rules exist.

