import ray

# Initialize Ray
try:
    ray.init(ignore_reinit_error=True)
    print("✅ Ray Successfully Initialized")
except Exception as e:
    print(f"❌ Ray Initialization Failed: {e}")
