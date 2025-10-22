import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.tracking.MlflowClient()

# get latest versions of a known model (replace with your model name)
model_name = "CallCenterAI_Transformer_Model"
versions = client.get_latest_versions(model_name)
for v in versions:
    print(f"version={v.version}, stage={v.current_stage}, source={v.source}")
