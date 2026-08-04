# 

curl --fail-with-body \
  --request POST \
  --url "https://att-langgraph-orchestrator-edward-jing-dev.apps.rm1.0a51.p1.openshiftapps.com/api/v1/ask" \
  --header "Content-Type: application/json" \
  --data '{
    "question": "Can an incorrect roaming charge be reversed?",
    "maxResultsPerSource": 3
  }'