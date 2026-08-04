# 

curl --fail-with-body \
  --request POST \
  --url "https://att-langgraph-orchestrator-edward-jing-dev.apps.rm1.0a51.p1.openshiftapps.com/api/v1/ask" \
  --header "Content-Type: application/json" \
  --data '{
    "question": "Can an incorrect roaming charge be reversed?",
    "maxResultsPerSource": 3
  }'



# policy

export APP_API_KEY="<THE_SAME_VALUE_STORED_IN_GITHUB>"

curl --fail-with-body \
  --request POST \
  --url "https://att-langgraph-orchestrator-edward-jing-dev.apps.rm1.0a51.p1.openshiftapps.com/api/v1/ask" \
  --header "Content-Type: application/json" \
  --data '{
    "question": "Can an incorrect roaming charge be reversed?",
    "maxResultsPerSource": 3
  }'

# billing_knowledge

curl --fail-with-body \
  --request POST \
  --url "https://att-langgraph-orchestrator-edward-jing-dev.apps.rm1.0a51.p1.openshiftapps.com/api/v1/ask" \
  --header "Content-Type: application/json" \
  --data '{
    "question": "Explain the roaming charges on my latest bill.",
    "billingAccountToken": "tok_demo_sas_003",
    "maxResultsPerSource": 3
  }'


# billing_transaction

curl --fail-with-body \
  --request POST \
  --url "https://att-langgraph-orchestrator-edward-jing-dev.apps.rm1.0a51.p1.openshiftapps.com/api/v1/ask" \
  --header "Content-Type: application/json" \
  --data '{
    "question": "Show the transaction records and amounts on my latest bill.",
    "billingAccountToken": "tok_demo_sas_003",
    "maxResultsPerSource": 3
  }'