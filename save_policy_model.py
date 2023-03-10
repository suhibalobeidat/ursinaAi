from ray.rllib.policy.policy import Policy

policy_dir = r"C:\Users\sohai\ray_results\Trainable_2023-03-06_00-42-12\Trainable_96b28_00000_0_2023-03-06_00-42-12\checkpoint_001477\policies\default_policy"

policy = Policy.from_checkpoint(policy_dir)    

print(policy.model)
policy.export_model(r"C:\Users\sohai\Desktop\ursinaAi")
