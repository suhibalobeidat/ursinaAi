from ray.rllib.policy.policy import Policy

policy_dir = r"C:\Users\sohai\ray_results\Trainable_2023-02-27_23-14-48\Trainable_6292f_00000_0_2023-02-27_23-14-48\checkpoint_001747\policies\default_policy"

policy = Policy.from_checkpoint(policy_dir)    

print(policy.model)
policy.export_model(r"C:\Users\sohai\Desktop\ursinaAi")
