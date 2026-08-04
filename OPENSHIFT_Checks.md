# Log in and select your OpenShift project

oc login --token=sha256~jBcjclMBC5AiwcKcCCR-EGEmTcPjBZuMAvmv7-QQ2Tk --server=https://api.rm1.0a51.p1.openshiftapps.com:6443

# Check the application logs

oc logs \
  deployment/att-langgraph-orchestrator \
  --tail=200

oc logs \
  deployment/att-langgraph-orchestrator \
  --since=30m |
grep -Ei "404|databricks|http|error|trace"

# check the environment variables in the running pod

oc exec \
  deployment/att-langgraph-orchestrator \
  -- printenv \
    DATABRICKS_HOST \
    DATABRICKS_POLICY_INDEX \
    DATABRICKS_BILLING_INDEX \
    DATABRICKS_CHAT_ENDPOINT