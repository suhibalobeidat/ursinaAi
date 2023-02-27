from ray.rllib.policy.policy import Policy

policy_dir = r"C:\Users\sohai\ray_results\Trainable_2023-02-23_13-52-37\Trainable_2fcda_00000_0_2023-02-23_13-52-38\checkpoint_001472\policies\default_policy"

policy = Policy.from_checkpoint(policy_dir)    

print(policy.model)
policy.export_model(r"C:\Users\sohai\Desktop\ursinaAi\dist")
