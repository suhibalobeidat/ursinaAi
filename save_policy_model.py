from ray.rllib.policy.policy import Policy

policy_dir = r"C:\Users\sohai\ray_results\Trainable_2023-03-28_14-39-53\Trainable_415de_00000_0_2023-03-28_14-39-53\checkpoint_001244\policies\default_policy"

policy = Policy.from_checkpoint(policy_dir)    

print(policy.model)
policy.export_model(r"C:\Users\sohai\Desktop\ursinaAi")
