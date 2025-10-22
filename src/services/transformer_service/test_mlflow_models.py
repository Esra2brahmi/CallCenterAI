import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

print("=" * 60)
print("MLFLOW REGISTRY CHECK")
print("=" * 60)

# 1. Check registered models
print("\n📦 REGISTERED MODELS:")
try:
    registered_models = client.search_registered_models()
    if registered_models:
        for rm in registered_models:
            print(f"\n   ✓ Model: {rm.name}")
            versions = client.search_model_versions(f"name='{rm.name}'")
            for v in versions:
                print(f"      Version {v.version}")
                print(f"      Source: {v.source}")
    else:
        print("   ❌ No registered models found!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Check experiments
print("\n🔬 EXPERIMENTS:")
try:
    experiments = client.search_experiments()
    for exp in experiments:
        print(f"\n   Experiment: {exp.name}")
        print(f"   ID: {exp.experiment_id}")
        
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=5,
            order_by=["start_time DESC"]
        )
        print(f"   Total runs: {len(runs)}")
        
        if runs:
            print(f"\n   Latest run:")
            run = runs[0]
            print(f"     ID: {run.info.run_id}")
            print(f"     Status: {run.info.status}")
            print(f"     Model URI: runs:/{run.info.run_id}/model")
            
            # Check for model artifacts
            try:
                artifacts = client.list_artifacts(run.info.run_id)
                print(f"\n     Artifacts:")
                for a in artifacts:
                    print(f"       - {a.path}")
            except Exception as e:
                print(f"     Error listing artifacts: {e}")

except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)