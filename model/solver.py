import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import json
from tqdm import tqdm, trange

from .layers import Summarizer, Discriminator, Actor, Critic
from .fragments import compute_fragments
from .utils import TensorboardWriter


def compute_returns(next_value, rewards, masks, gamma=0.99):
    """Function that computes the return z_i following the equation (6) of the paper"""
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates AC-SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        self._build()

    def _build(self):
        device = self.config.device
        action_state_size = self.config.action_state_size

        # Build Modules
        self.linear_compress = nn.Linear(
            self.config.input_size, self.config.hidden_size
        ).to(device)
        self.summarizer = Summarizer(
            input_size=self.config.hidden_size, hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers
        ).to(device)
        self.discriminator = Discriminator(
            input_size=self.config.hidden_size, hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers
        ).to(device)
        self.actor = Actor(
            state_size=action_state_size, action_size=action_state_size
        ).to(device)
        self.critic = Critic(
            state_size=action_state_size, action_size=action_state_size
        ).to(device)
        self.model = nn.ModuleList([
            self.linear_compress, self.summarizer, self.discriminator, self.actor, self.critic
        ])

        if self.config.mode == "train":
            # Build Optimizers
            self.e_optimizer = optim.Adam(
                self.summarizer.vae.e_lstm.parameters(),
                lr=self.config.lr
            )
            self.d_optimizer = optim.Adam(
                self.summarizer.vae.d_lstm.parameters(),
                lr=self.config.lr
            )
            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters()) + list(self.linear_compress.parameters()),
                lr=self.config.discriminator_lr
            )
            self.optimizerA_s = optim.Adam(
                list(self.actor.parameters()) + list(self.summarizer.s_lstm.parameters()) +
                list(self.linear_compress.parameters()),
                lr=self.config.lr
            )
            self.optimizerC = optim.Adam(self.critic.parameters(), lr=self.config.lr)

            self.writer = TensorboardWriter(str(self.config.log_dir))

    def load_model_state(self, epoch_i):
        device = self.config.device
        checkpoint_path = self.config.model_dir / f"epoch-{epoch_i}.pkl"
        print("checkpoint file path:", checkpoint_path)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(device)

    def reconstruction_loss(self, h_origin, h_sum):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""
        return torch.norm(h_origin - h_sum, p=2)

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""
        return torch.abs(torch.mean(scores) - self.config.regularization_factor)

    criterion = nn.MSELoss()

    def AC(self, original_features, seq_len, action_fragments):
        """Function that makes the actor's actions, in the training steps where the actor and critic components are not trained"""
        device = self.config.device
        action_state_size = self.config.action_state_size
        termination_point = self.config.termination_point

        scores = self.summarizer.s_lstm(original_features)  # [seq_len, 1]

        fragment_scores = np.zeros(action_state_size)  # [num_fragments, 1]
        for fragment in range(action_state_size):
            frag_begin, frag_end = action_fragments[fragment] + np.asarray((0, 1))
            fragment_scores[fragment] = scores[frag_begin:frag_end].mean()
        state = fragment_scores

        previous_actions = []  # save all the actions (the selected fragments of each episode)
        reduction_factor = (action_state_size - termination_point) / action_state_size
        action_scores = (torch.ones(seq_len) * reduction_factor).to(device)
        action_fragment_scores = (torch.ones(action_state_size)).to(device)

        counter = 0
        for _ in range(termination_point):
            state = torch.FloatTensor(state).to(device)
            # select an action
            dist = self.actor(state)
            # dist.probs = dist.probs * action_fragment_scores    # prevent duplicate action
            action = dist.sample()  # returns a scalar between 0-action_state_size

            if action in previous_actions:
                # duplicate action
                pass
            else:
                previous_actions.append(action)
                action_factor = (termination_point - counter) / (action_state_size - counter) + 1

                frag_begin, frag_end = action_fragments[action] + np.asarray((0, 1))
                action_scores[frag_begin:frag_end] = action_factor
                action_fragment_scores[action] = 0

                counter = counter + 1

            next_state = state * action_fragment_scores
            next_state = next_state.cpu().detach().numpy()
            state = next_state

        weighted_scores = action_scores.unsqueeze(1) * scores
        weighted_features = weighted_scores.view(-1, 1, 1) * original_features

        return weighted_features, weighted_scores

    def train(self):
        device = self.config.device

        # labels for training the GAN part of the model
        original_label = torch.tensor(1.0).to(device)
        summary_label = torch.tensor(0.0).to(device)

        step = 0
        for epoch_i in trange(self.config.n_epochs, desc="Epoch", ncols=80):
            action_state_size = self.config.action_state_size
            self.model.train()

            recon_loss_init_history = []
            recon_loss_history = []
            sparsity_loss_history = []
            prior_loss_history = []
            g_loss_history = []
            e_loss_history = []
            d_loss_history = []
            c_original_loss_history = []
            c_summary_loss_history = []
            actor_loss_history = []
            critic_loss_history = []
            reward_history = []

            # Train in batches of as many videos as the batch_size
            num_batches = int(len(self.train_loader) / self.config.batch_size)
            iterator = iter(self.train_loader)
            for batch in range(num_batches):
                list_image_features = []
                list_action_fragments = []

                print(f"batch: {batch}")

                # ---- Train eLSTM ----#
                if self.config.verbose:
                    tqdm.write("Training eLSTM...")
                self.e_optimizer.zero_grad()
                for video in range(self.config.batch_size):
                    image_features, _ = next(iterator)
                    action_fragments = compute_fragments(image_features.shape[1], action_state_size)

                    # [batch_size, seq_len, input_size]
                    # [seq_len, input_size]
                    image_features = image_features.view(-1, self.config.input_size)
                    
                    list_image_features.append(image_features)
                    list_action_fragments.append(action_fragments)

                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).to(device)
                    seq_len = image_features_.shape[0]

                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

                    weighted_features, scores = self.AC(original_features, seq_len, action_fragments)
                    h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)

                    h_origin, original_prob = self.discriminator(original_features)
                    h_sum, sum_prob = self.discriminator(generated_features)

                    if self.config.verbose:
                        tqdm.write(f"original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}")

                    reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
                    prior_loss = self.prior_loss(h_mu, h_log_variance)

                    tqdm.write(f"recon loss {reconstruction_loss.item():.3f}, prior loss: {prior_loss.item():.3f}")

                    e_loss = reconstruction_loss + prior_loss
                    e_loss = e_loss / self.config.batch_size
                    e_loss.backward()

                    prior_loss_history.append(prior_loss.data)
                    e_loss_history.append(e_loss.data)

                # Update e_lstm parameters every "batch_size" iterations
                torch.nn.utils.clip_grad_norm_(self.summarizer.vae.e_lstm.parameters(), self.config.clip)
                self.e_optimizer.step()

                #---- Train dLSTM (decoder/generator) ----#
                if self.config.verbose:
                    tqdm.write("Training dLSTM...")
                self.d_optimizer.zero_grad()
                for video in range(self.config.batch_size):
                    image_features = list_image_features[video]
                    action_fragments = list_action_fragments[video]

                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).to(device)
                    seq_len = image_features_.shape[0]

                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)

                    weighted_features, _ = self.AC(original_features, seq_len, action_fragments)
                    h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)

                    h_origin, original_prob = self.discriminator(original_features)
                    h_sum, sum_prob = self.discriminator(generated_features)

                    tqdm.write(f"original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}")

                    reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
                    g_loss = self.criterion(sum_prob, original_label)

                    orig_features = original_features.squeeze(1)    # [seq_len, hidden_size]
                    gen_features = generated_features.squeeze(1)    #         >>
                    recon_losses = []
                    for frame_index in range(seq_len):
                        recon_losses.append(self.reconstruction_loss(orig_features[frame_index, :], gen_features[frame_index, :]))
                    reconstruction_loss_init = torch.stack(recon_losses).mean()

                    if self.config.verbose:
                        tqdm.write(f"recon loss {reconstruction_loss.item():.3f}, g loss: {g_loss.item():.3f}")

                    d_loss = reconstruction_loss + g_loss
                    d_loss = d_loss / self.config.batch_size
                    d_loss.backward()

                    recon_loss_init_history.append(reconstruction_loss_init.data)
                    recon_loss_history.append(reconstruction_loss.data)
                    g_loss_history.append(g_loss.data)
                    d_loss_history.append(d_loss.data)

                # Update d_lstm parameters every "batch_size" iterations
                torch.nn.utils.clip_grad_norm_(self.summarizer.vae.d_lstm.parameters(), self.config.clip)
                self.d_optimizer.step()

                #---- Train cLSTM ----#
                if self.config.verbose:
                    tqdm.write("Training cLSTM...")
                self.c_optimizer.zero_grad()
                for video in range(self.config.batch_size):
                    image_features = list_image_features[video]
                    action_fragments = list_action_fragments[video]

                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).to(device)
                    seq_len = image_features_.shape[0]

                    # Train with original loss
                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                    h_origin, original_prob = self.discriminator(original_features)
                    c_original_loss = self.criterion(original_prob, original_label)
                    c_original_loss = c_original_loss / self.config.batch_size
                    c_original_loss.backward()

                    # Train with summary loss
                    weighted_features, _ = self.AC(original_features, seq_len, action_fragments)
                    h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)
                    h_sum, sum_prob = self.discriminator(generated_features.detach())
                    c_summary_loss = self.criterion(sum_prob, summary_label)
                    c_summary_loss = c_summary_loss / self.config.batch_size
                    c_summary_loss.backward()

                    tqdm.write(f"original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}")
                    
                    c_original_loss_history.append(c_original_loss.data)
                    c_summary_loss_history.append(c_summary_loss.data)

                # Update c_lstm parameters every "batch_size" iterations
                torch.nn.utils.clip_grad_norm_(list(self.discriminator.parameters()) + list(self.linear_compress.parameters()), self.config.clip)
                self.c_optimizer.step()

                #---- Train sLSTM and actor-critic ----#
                if self.config.verbose:
                    tqdm.write("Training sLSTM, actor and critic...")
                self.optimizerA_s.zero_grad()
                self.optimizerC.zero_grad()
                for video in range(self.config.batch_size):
                    image_features = list_image_features[video]
                    action_fragments = list_action_fragments[video]

                    # [seq_len, input_size]
                    image_features_ = Variable(image_features).to(device)
                    seq_len = image_features_.shape[0]

                    # [seq_len, 1, hidden_size]
                    original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
                    scores = self.summarizer.s_lstm(original_features)  # [seq_len, 1]

                    fragment_scores = np.zeros(action_state_size)  # [num_fragments, 1]
                    for fragment in range(action_state_size):
                        frag_begin, frag_end = action_fragments[fragment] + np.asarray((0, 1))
                        fragment_scores[fragment] = scores[frag_begin:frag_end].mean()

                    state = fragment_scores  # [action_state_size, 1]

                    previous_actions = []  # save all the actions (the selected fragments of each step)
                    reduction_factor = (action_state_size - self.config.termination_point) / action_state_size
                    action_scores = (torch.ones(seq_len) * reduction_factor).to(device)
                    action_fragment_scores = (torch.ones(action_state_size)).to(device)

                    log_probs = []
                    values = []
                    rewards = []
                    masks = []
                    entropy = 0

                    counter = 0
                    for ACstep in range(self.config.termination_point):
                        # select an action, get a value for the current state
                        state = torch.FloatTensor(state).to(device)  # [action_state_size, 1]
                        dist, value = self.actor(state), self.critic(state)
                        action = dist.sample()  # returns a scalar between 0-action_state_size

                        if action in previous_actions:
                            reward = 0
                        else:
                            previous_actions.append(action)
                            action_factor = (self.config.termination_point - counter) / (action_state_size - counter) + 1

                            frag_begin, frag_end = action_fragments[action] + np.asarray((0, 1))
                            action_scores[frag_begin:frag_end] = action_factor
                            action_fragment_scores[action] = 0

                            weighted_scores = action_scores.unsqueeze(1) * scores
                            weighted_features = weighted_scores.view(-1, 1, 1) * original_features

                            h_mu, h_log_variance, generated_features = self.summarizer.vae(weighted_features)

                            h_origin, original_prob = self.discriminator(original_features)
                            h_sum, sum_prob = self.discriminator(generated_features)

                            tqdm.write(f"original_p: {original_prob.item():.3f}, summary_p: {sum_prob.item():.3f}")

                            rec_loss = self.reconstruction_loss(h_origin, h_sum)
                            reward = 1 - rec_loss.item()  # the less the distance, the higher the reward
                            counter = counter + 1

                        next_state = state * action_fragment_scores
                        next_state = next_state.cpu().detach().numpy()

                        log_prob = dist.log_prob(action).unsqueeze(0)
                        entropy += dist.entropy().mean()

                        log_probs.append(log_prob)
                        values.append(value)
                        rewards.append(torch.tensor([reward], dtype=torch.float, device=device))

                        if ACstep == (self.config.termination_point - 1):
                            masks.append(torch.tensor([0], dtype=torch.float, device=device)) 
                        else:
                            masks.append(torch.tensor([1], dtype=torch.float, device=device))

                        state = next_state

                    next_state = torch.FloatTensor(next_state).to(device)
                    next_value = self.critic(next_state)
                    returns = compute_returns(next_value, rewards, masks)

                    log_probs = torch.cat(log_probs)
                    returns = torch.cat(returns).detach()
                    values = torch.cat(values)

                    advantage = returns - values

                    actor_loss = -((log_probs * advantage.detach()).mean() + (self.config.entropy_coef / self.config.termination_point) * entropy)
                    sparsity_loss = self.sparsity_loss(scores)
                    critic_loss = advantage.pow(2).mean()

                    actor_loss = actor_loss / self.config.batch_size
                    sparsity_loss = sparsity_loss / self.config.batch_size
                    critic_loss = critic_loss / self.config.batch_size
                    actor_loss.backward()
                    sparsity_loss.backward()
                    critic_loss.backward()

                    reward_mean = torch.mean(torch.stack(rewards))
                    reward_history.append(reward_mean)
                    actor_loss_history.append(actor_loss)
                    sparsity_loss_history.append(sparsity_loss)
                    critic_loss_history.append(critic_loss)

                    if self.config.verbose:
                        tqdm.write("Plotting...")

                    self.writer.update_loss(original_prob.data, step, "original_prob")
                    self.writer.update_loss(sum_prob.data, step, "sum_prob")

                    step += 1

                # Update s_lstm, actor and critic parameters every "batch_size" iterations
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.linear_compress.parameters()) +
                    list(self.summarizer.s_lstm.parameters()) + list(self.critic.parameters()),
                    self.config.clip
                )
                self.optimizerA_s.step()
                self.optimizerC.step()

            recon_loss_init = torch.stack(recon_loss_init_history).mean()
            recon_loss = torch.stack(recon_loss_history).mean()
            prior_loss = torch.stack(prior_loss_history).mean()
            g_loss = torch.stack(g_loss_history).mean()
            e_loss = torch.stack(e_loss_history).mean()
            d_loss = torch.stack(d_loss_history).mean()
            c_original_loss = torch.stack(c_original_loss_history).mean()
            c_summary_loss = torch.stack(c_summary_loss_history).mean()
            sparsity_loss = torch.stack(sparsity_loss_history).mean()
            actor_loss = torch.stack(actor_loss_history).mean()
            critic_loss = torch.stack(critic_loss_history).mean()
            reward = torch.mean(torch.stack(reward_history))

            # Plot
            if self.config.verbose:
                tqdm.write("Plotting...")
            self.writer.update_loss(recon_loss_init, epoch_i, "recon_loss_init_epoch")
            self.writer.update_loss(recon_loss, epoch_i, "recon_loss_epoch")
            self.writer.update_loss(prior_loss, epoch_i, "prior_loss_epoch")    
            self.writer.update_loss(g_loss, epoch_i, "g_loss_epoch")    
            self.writer.update_loss(e_loss, epoch_i, "e_loss_epoch")
            self.writer.update_loss(d_loss, epoch_i, "d_loss_epoch")
            self.writer.update_loss(c_original_loss, epoch_i, "c_original_loss_epoch")
            self.writer.update_loss(c_summary_loss, epoch_i, "c_summary_loss_epoch")
            self.writer.update_loss(sparsity_loss, epoch_i, "sparsity_loss_epoch")
            self.writer.update_loss(actor_loss, epoch_i, "actor_loss_epoch")
            self.writer.update_loss(critic_loss, epoch_i, "critic_loss_epoch")
            self.writer.update_loss(reward, epoch_i, "reward_epoch")

            # Save parameters at checkpoint
            checkpoint_path = str(self.config.model_dir / f"epoch-{epoch_i}.pkl")
            if self.config.verbose:
                tqdm.write(f"Save parameters at {checkpoint_path}")
            torch.save(self.model.state_dict(), checkpoint_path)

            self.evaluate(epoch_i)

    def predict(self, features):
        device = self.config.device
        input_size = self.config.input_size
        assert(features.shape[-1] == input_size)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features)

        action_state_size = self.config.action_state_size
        action_fragments = compute_fragments(features.shape[0], action_state_size)

        self.model.eval()

        # [seq_len, input_size]
        features = features.view(-1, input_size).to(device)

        # [seq_len, 1, hidden_size]
        original_features = self.linear_compress(features.detach())
        original_features = original_features.unsqueeze(1)  # add a dimension for batch
        seq_len = original_features.shape[0]

        with torch.no_grad():
            _, scores = self.AC(original_features, seq_len, action_fragments)

            scores = scores.squeeze(1)
            scores = scores.cpu().detach().numpy()
            return scores

    def evaluate(self, epoch_i):
        self.model.eval()

        out_dict = {}
        for image_features, video_name in tqdm(self.test_loader, desc="Evaluate", ncols=80, leave=False):
            if self.config.verbose:
                tqdm.write(f"Predicting scores of {video_name} ...")

            scores = self.predict(image_features)
            out_dict[video_name] = scores.tolist()

            score_save_path = self.config.score_dir / f"{self.config.dataset}_{epoch_i}.json"
            with open(score_save_path, "w") as f:
                if self.config.verbose:
                    tqdm.write(f"Saving score at {str(score_save_path)}.")
                json.dump(out_dict, f, indent=4)
            score_save_path.chmod(0o777)


if __name__ == "__main__":
    pass
