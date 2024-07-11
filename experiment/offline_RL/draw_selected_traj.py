from offline_RL.vdice_correct_filter import *

@dataclass
class TrainConfig:
    #############################
    ######### Experiment ########
    #############################
    seed: int = 100
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    save_freq: int = int(5e3)  # How often (time steps) save the model
    update_freq: int = int(1.5e5)  # How often (time steps) we update the model
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(6e6)  # Max time steps to run environment
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    # discount: float = 0.99  # Discount factor
    discount: float = 0.99  # Discount factor

    #############################
    ######### NN Arc ############
    #############################
    vf_lr: float = 3e-4  # V function learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    layernorm: bool = False
    hidden_dim: int = 256
    tau: float = 0.005  # Target network update rate
    
    #############################
    ###### dataset preprocess ###
    #############################
    # normalize: bool = True  # Normalize states
    normalize: bool = False  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    # normalize_reward: bool = False  # Normalize reward
    
    #############################
    ###### Wandb Logging ########
    #############################
    project: str = "test"
    checkpoints_path: Optional[str] = "test"
    alg: str = "test"
    env_1: str = "antmaze-umaze-v2"  # OpenAI gym environment name
    env_2: str = "antmaze-umaze-v2"  # OpenAI gym environment name

    #############################
    #### DICE Hyperparameters ###
    #############################
    device: str = "cuda"
    vdice_type: Optional[str] = "semi"
    semi_dice_lambda: float = 0.3
    true_dice_alpha: float = 2.0
    env_name: str = "EmptyRoom"
    reward_type: str = "sparse"

def create_dataset(file_path="dataset.npy"):

    env = PointMassEnv(start=np.array([12.5, 4.5], dtype=np.float32), 
                                goal=np.array([4.5, 12.5], dtype=np.float32), 
                                goal_radius=0.8,
                                env_name=config.env_name,
                                reward_type=config.reward_type,)
    dataset = np.load(file_path, allow_pickle=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset["id"] = np.arange(dataset["actions"].shape[0])

    # dataset = change_reward(dataset)

    return dataset, state_dim, action_dim, env

def get_weights(dataset, trainer):

    semi_s_weight = []
    semi_a_weight = []
    semi_s_or_a_weight = []
    semi_s_and_a_weight = []
    true_sa_weight = []

    for i in range(0,len(dataset["observations"]),8192):
        end_idx = min(i+8192,len(dataset["observations"]))

        obs = torch.tensor(dataset["observations"][i:end_idx], dtype=torch.float32).to(trainer.device)
        next_obs = torch.tensor(dataset["next_observations"][i:end_idx], dtype=torch.float32).to(trainer.device)
        actions = torch.tensor(dataset["actions"][i:end_idx], dtype=torch.float32).to(trainer.device)
        rewards = torch.tensor(dataset["rewards"][i:end_idx], dtype=torch.float32).to(trainer.device)
        terminals = torch.tensor(dataset["terminals"][i:end_idx], dtype=torch.float32).to(trainer.device)

        semi_s, semi_a, true_sa = trainer.get_weights(obs, 
                                                        next_obs, 
                                                        actions, 
                                                        rewards, 
                                                        terminals)
        semi_s_weight.append(semi_s.cpu().numpy())
        semi_a_weight.append(semi_a.cpu().numpy())
        semi_s_or_a_weight.append(torch.max(semi_s, semi_a).cpu().numpy())
        semi_s_and_a_weight.append((semi_s * semi_a).cpu().numpy())
        true_sa_weight.append(true_sa.cpu().numpy())

    semi_s_weight = np.hstack(semi_s_weight)
    semi_s_weight = semi_s_weight > 0
    
    semi_a_weight = np.hstack(semi_a_weight)
    semi_a_weight = semi_a_weight > 0
    
    semi_s_or_a_weight = np.hstack(semi_s_or_a_weight)
    semi_s_or_a_weight = semi_s_or_a_weight > 0
    
    semi_s_and_a_weight = np.hstack(semi_s_and_a_weight)
    semi_s_and_a_weight = semi_s_and_a_weight > 0

    true_sa_weight = np.hstack(true_sa_weight)
    true_sa_weight = true_sa_weight > 0


    return semi_s_weight, semi_a_weight, semi_s_or_a_weight, semi_s_and_a_weight, true_sa_weight

def draw_traj(weights, dataset, env, save_path=None):
    # select the observation in dataset with weights > 0
    selected_obs = dataset["observations"][weights]
    selected_next_obs = dataset["next_observations"][weights]
    selected_terminals = dataset["terminals"][weights]
    selected_traj_img = env.get_env_frame_with_selected_traj(start=env._start, 
                                                            goal=env._goal, 
                                                            obs=selected_obs, 
                                                            next_obs=selected_next_obs,
                                                            terminals=selected_terminals,
                                                            save_path=save_path)
                                                            # save_path="full_traj.png")
    


    

@pyrallis.wrap()
def main(config: TrainConfig):

    dataset, state_dim, action_dim, env = create_dataset("dataset.npy")
    expert_dataset, state_dim, action_dim, env = create_dataset("expert_dataset.npy")
    dataset = expert_dataset
    max_action = float(env.action_space.high[0])

    actor = GaussianPolicy(
        state_dim, action_dim, max_action, dropout=config.actor_dropout
    ).to(config.device)
    true_v_network = ValueFunction(state_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)
    semi_v_network = ValueFunction(state_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)
    q_network = TwinQ(state_dim, action_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)
    mu_network = ValueFunction(state_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)
    U_network = ValueFunction(state_dim, layernorm=config.layernorm, hidden_dim=config.hidden_dim).to(config.device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    true_v_optimizer = torch.optim.Adam(true_v_network.parameters(), lr=config.vf_lr)
    semi_v_optimizer = torch.optim.Adam(semi_v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.vf_lr)
    mu_optimizer = torch.optim.Adam(mu_network.parameters(), lr=config.vf_lr)
    U_optimizer = torch.optim.Adam(U_network.parameters(), lr=config.vf_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "true_v_network": true_v_network,
        "true_v_optimizer": true_v_optimizer,
        "semi_v_network": semi_v_network,
        "semi_v_optimizer": semi_v_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "mu_network": mu_network,
        "mu_optimizer": mu_optimizer,
        "U_network": U_network,
        "U_optimizer": U_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # VDICE
        "vdice_type": config.vdice_type,
        "semi_dice_lambda": config.semi_dice_lambda,
        "true_dice_alpha": config.true_dice_alpha,
        "max_steps": config.max_timesteps,
    }

    # Initialize actor
    trainer = VDICE(**kwargs)
    
    
    modify_reward(dataset, config.env_1)

    # only take the first 1000 samples from the dataset
    dataset["observations"] = dataset["observations"][:2000]
    dataset["actions"] = dataset["actions"][:2000]
    dataset["rewards"] = dataset["rewards"][:2000]
    dataset["next_observations"] = dataset["next_observations"][:2000]
    dataset["terminals"] = dataset["terminals"][:2000]

    # i = 4999
    # while i < 3_000_000:
    #     trainer.load_state_dict(

    #         torch.load("sub_semi_nns_nr_l_3/sub_semi_nns_nr_l_3-antmaze-umaze-v2-23b0e99f/model/checkpoint_"+ str(int(i))+".pt")
    #         )
    #     print("load model at checkpoint_" + str(int(i)))
    #     semi_s_weight, semi_a_weight, semi_s_or_a_weight, semi_s_and_a_weight, true_sa_weight = get_weights(dataset, trainer)
    #     # semi_s_or_a_weight = np.ones_like(semi_s_or_a_weight)
    #     # import pdb; pdb.set_trace()
    #     weights_dict = {
    #         "semi_s_weight": semi_s_weight,
    #         "semi_a_weight": semi_a_weight,
    #         "semi_s_or_a_weight": semi_s_or_a_weight,
    #         "semi_s_and_a_weight": semi_s_and_a_weight,
    #         "true_sa_weight": true_sa_weight
    #     }
    #     for key, weights in weights_dict.items():
    #         save_dir = "sub_semi_nns_nr_l_3/selected_traj/"+key
    #         os.makedirs(save_dir, exist_ok=True)
    #         draw_traj(weights, dataset, env, save_path = save_dir + "/" + str(int(i)) + ".png")
    #         # draw_traj(semi_s_or_a_weight, dataset, env, save_path = "img/full.png")

    #     i += 5000


    trainer.load_state_dict(
            torch.load("empty_room/empty_room_normalize_reward_True_true_dice_alpha_2_discount_0_99_semi_dice_lambda_0_3_seed_100/test-antmaze-umaze-v2-a79f1170/model/checkpoint_999999.pt")
            )
    semi_s_weight, semi_a_weight, semi_s_or_a_weight, semi_s_and_a_weight, true_sa_weight = get_weights(dataset, trainer)
    # semi_s_weight = np.ones_like(semi_s_weight)
    # import pdb; pdb.set_trace()
    draw_traj(true_sa_weight, dataset, env, save_path = "true_sa_weight.png")
    

    
    


if __name__ == "__main__":
    config = pyrallis.parse(config_class=TrainConfig)
    main()