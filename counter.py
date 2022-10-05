class Counter:
    def __init__(self):
        self.iter_index = 0
        self.iter_epoch = 0
        self.iter_test = 0
        self.iter_successful_ep = 0
        self.iter_ppo = 0
        self.total_time = 0
        self.epochs_counter = 0
        self.test_average_reward = 0
        self.test_average_steps = 0
        