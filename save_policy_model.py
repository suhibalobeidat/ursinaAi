from ray.rllib.policy.policy import Policy

policy_dir = r"C:\Users\sohai\ray_results\tensorboard\Trainable_bde87636_5_clip_param=0.2825,entropy_coeff=0.0895,fcnet_activation=0.5979,fcnet_hiddens_layer_count=4.6484,gamma=0.8168,3\Trainable_51878_00000_0_2023-02-08_11-41-02\checkpoint_003091\policies\default_policy"

policy = Policy.from_checkpoint(policy_dir)    

print(policy.model)
policy.export_model(r"C:\Users\sohai\Desktop\ursinaAi\dist")
