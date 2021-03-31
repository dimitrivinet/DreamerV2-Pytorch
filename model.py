import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, gamma, num_action=9):
        super(WorldModel, self).__init__()
        self.gamma = gamma #discount factor
        self.num_action = num_action

        #Recurrent Model (RSSM): ((z, a), h) -> h
        self.gru = nn.GRUCell(
            input_size=1024 + num_action,
            hidden_size=512,
        )

        #q (1st part): (x) -> xembedded
        self.representation_model_conv = nn.Sequential(
            #3,64,64
            #TODO
            nn.Conv2d(3, 32, kernel_size=3, padding=2),
            nn.ELU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ELU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.ELU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.ELU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.ELU(inplace=True),
            # #512, 1, 1
        )
        #q (2nd part): (h, xembedded) -> z
        self.representation_model_mlp = nn.Sequential(
            #TODO
            nn.Linear(512+512, 32*32),
            nn.ELU(inplace=True),
            nn.Linear(32*32, 32*32),
        )

        #p: (h) -> z_hat
        self.transition_predictor = nn.Sequential(
            #TODO
            nn.Linear(512, 32*32),
            nn.ELU(inplace=True),
            nn.Linear(32*32, 32*32),
        )

        #p: (h, z) -> r_hat
        self.r_predictor_mlp = nn.Sequential(
            #TODO
            nn.Linear(512+32*32, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 1),
        )

        #p: (h, z) -> y_hat
        self.gamma_predictor_mlp = nn.Sequential(
            #TODO
            nn.Linear(512+32*32, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 1),
        )

        #p: (h, z) -> x_hat
        self.x_hat_predictor_mlp = nn.Sequential(
            #TODO
            nn.Linear(512+32*32, 32*32),
            nn.ELU(inplace=True),
        )

        self.image_predictor_conv = nn.Sequential( #64, 4, 4
            #TODO
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # out: 32, 8, 8
            nn.ELU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # out: 16, 16, 16
            nn.ELU(inplace=True),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),    # out: 8, 32, 32
            nn.ELU(inplace=True),

            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),     # out: 3, 64, 64 (image reconstruite)
        )
    
    def compute_h(self, batch_size, device, a=None, h=None, z=None):
        """
        In:
            batch_size: int
            device: torch.device of the model
            a: a_t-1
            h: h_t-1
            z: z_t-1
        Out:
            h: h_t
        """
        if h is None: #starting new sequence
            h = torch.zeros((batch_size, 512)).to(device)
        else:
            #TODO
            h = self.gru(torch.cat([a, z], d=-1), h)
        return h

    def compute_z(self, x, h):
        """
        In:
            x: x_t
            h: h_t
        Out:
            z_logit:  logits of z_t
            z_sample: z_t
        """
        #TODO
        x_embedding = self.representation_model_conv(x)
        x_embedding = x_embedding.view(-1, 512)

        z_logits = self.representation_model_mlp(
            torch.cat([x_embedding, h], dim=-1)
        )
        z_sample = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=z_logits
        ).sample()
        z_probs = nn.Softmax(z_logits, dim=-1)
        z_sample = z_sample + z_probs - z_probs.detach() # sample + probs - (probs sans gradiant)

        return z_logits, z_sample
    
    def compute_z_hat_sample(self, z_hat_logits):
        """
        In:
            z_hat_logits: logits of z_hat_t
        Out:
            z_hat_sample (with straight through gradient)
        """
        #TODO
        z_hat_sample = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=z_hat_logits.reshape(-1, 32, 32)
        ).sample()
        z_hat_probs = nn.Softmax(z_hat_logits, dim=-1)
        z_hat_sample = z_hat_sample + z_hat_probs - z_hat_probs.detach() # sample + probs - (probs sans gradiant)

        return z_hat_sample

    def compute_x_hat(self, h_z):
        """
        In:
            h_z: concat of h_t and z_t
        Out:
            x_hat: x_hat_t
        """
        #TODO
        x_hat_embedding = self.x_hat_predictor_mlp(h_z).reshape(-1, 64, 4, 4)

        return self.image_predictor_conv(x_hat_embedding)

    #using inference
    def forward_inference(self, a, x, z, h):
        """
        In:
            a: a_t-1
            x: x_t
            z: z_t-1 (sampled not logits)
            h: h_t-1
        """
        #TODO
        h = self.compute_h(a.shape[0], x.device, a, h, z)
        _, z_sample = self.compute_z(x, h)
        #no straight-though gradient
        return z_sample, h

    def dream(self, a, x, z, h):
        #TODO
        h = self.compute_h(x.shape[0], x.device, a, h, z)

        z_hat_logits = self.transition_predictor(h)
        z_hat_sample = self.compute_z_hat_sample(z_hat_logits)

        h_z = torch.cat([h, z], dim=-1)

        r_hat = self.r_predictor_mlp(h_z)
        gamma_hat = self.gamma_predictor_mlp(h_z)

        z_hat_sample = self.compute_z_hat_sample(z_hat_logits)

        gamma_hat_sample = torch.distributions.bernoulli.Bernoulli(
            torch.tensor([gamma_hat])
        ).sample() * self.gamma

        x_hat, z_logits, z_sample, r_hat_sample = None

        return z_logits, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, (z_hat_sample, r_hat_sample, gamma_hat_sample)

    def train(self, a, x, z, h):
        #TODO
        h = self.compute_h(a.shape[0], x.device, a, h, z)
        z_logits, z_sample = self.compute_z(x, h)
        z_hat_logits = self.transition_predictor(h)

        h_z = torch.cat([h, z], dim=-1)
        x_hat = self.compute_x_hat(h_z)
        r_hat = self.r_predictor_mlp(h_z)
        gamma_hat = self.gamma_predictor_mlp(h_z)

        z_hat_sample, r_hat_sample, gamma_hat_sample = None
        # z_hat_sample = self.compute_z_hat_sample(z_hat_logits)
        # r_hat_sample = torch.distributions.normal.Normal(
        #     torch.tensor([r_hat]), torch.tensor([1.0])
        # ).sample()
        # gamma_hat_sample = torch.distributions.bernoulli.Bernoulli(
        #     torch.tensor([gamma_hat])
        # ).sample()

    
        return z_logits, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, (z_hat_sample, r_hat_sample, gamma_hat_sample)


    def forward(self, a, x, z, h=None, dream=False, inference=False):
        """
        Input:
            a: a_t-1
            x: x_t
            z: z_t-1
            h: h_t-1
        """
        if inference: #only use embedding network, i.e. no image predictor
            return self.forward_inference(a, x, z, h)
        elif dream:
            return self.dream(a, x, z, h)
        else:
            return self.train(a, x, z, h)

class Actor(nn.Module):
    def __init__(self, num_actions=9):
        super(Actor, self).__init__()
        
        self.num_actions = num_actions

        self.model = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, num_actions),
            nn.Tanh()
        )

    def forward(self, z_sample):
        z_sample = z_sample.reshape(-1, 32*32)
        return self.model(z_sample)*2

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, z_sample):
        z_sample = z_sample.reshape(-1, 32*32)
        return self.model(z_sample)

class LossModel(nn.Module):
    def __init__(self, nx=1/64/64/3, nr=1, ng=1, nt=0.08, nq=0.1):
        super(LossModel, self).__init__()

        self.nx = nx
        self.nr = nr
        self.ng = ng
        self.nt = nt
        self.nq = nq

    def forward(self, x, r, gamma, z_logits, z_sample, x_hat, r_hat, gamma_hat, z_hat_logits):
        x_dist = torch.distributions.normal.Normal(
            loc=x_hat,
            scale=1.0
        )
        r_dist = torch.distributions.normal.Normal(
            loc=r_hat,
            scale=1.0
        )
        gamma_dist = torch.distributions.bernoulli.Bernoulli(
            logits=gamma_hat
        )
        z_hat_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=z_hat_logits.reshape(-1, 32, 32)
        )
        z_dist = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=z_logits.reshape(-1, 32, 32).detach()
        )

        z_sample = z_sample.reshape(-1, 32, 32)

        loss = -self.nx*x_dist.log_prob(x).mean()\
                -self.nr*r_dist.log_prob(r).mean()\
                -self.ng*gamma_dist.log_prob(gamma.round()).mean()\
                -self.nt*z_hat_dist.log_prob(z_sample.detach()).mean()\
                +self.nq*z_dist.log_prob(z_sample).mean()

        return loss

class ActorLoss(nn.Module):
    def __init__(self, ns=0.9, nd=0.1, ne=3e-3):
        super(ActorLoss, self).__init__()

        self.ns = ns
        self.nd = nd
        self.ne = ne

        self.anneal = 1e-5

    def forward(self, a, dist_a, V, ve):
        #TODO

        return loss.mean()

class CriticLoss(nn.Module):
    def __init__(self):
        super(CriticLoss, self).__init__()

        self.mse = nn.MSELoss()

    def forward(self, V, ve):
        return self.mse(V, ve)
