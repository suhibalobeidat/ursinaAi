from ray.rllib.policy.policy import Policy

policy_dir = r"C:\Users\sohai\ray_results\Trainable_2023-04-04_18-55-31\Trainable_20ee7_00000_0_2023-04-04_18-55-32\checkpoint_000764\policies\default_policy"

policy = Policy.from_checkpoint(policy_dir)    

print(policy.model)
policy.export_model(r"C:\Users\sohai\Desktop\ursinaAi")
