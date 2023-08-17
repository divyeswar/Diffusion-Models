import torch
import pytorch_lightning as pl
import torch.utils.data

def cosine_beta_schedule(timesteps,lbeta,ubeta,s):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class LitDiffusionModel(pl.LightningModule):
    def __init__(self, n_dim=3, n_steps=200, lbeta=1e-5, ubeta=1e-2):
        super().__init__()
        """
        If you include more hyperparams (e.g. `n_layers`), be sure to add that to `argparse` from `train.py`.
        Also, manually make sure that this new hyperparameter is being saved in `hparams.yaml`.
        """
        self.save_hyperparameters()

        """
        Your model implementation starts here. We have separate learnable modules for `time_embed` and `model`.
        You may choose a different architecture altogether. Feel free to explore what works best for you.
        If your architecture is just a sequence of `torch.nn.XXX` layers, using `torch.nn.Sequential` will be easier.
        
        `time_embed` can be learned or a fixed function based on the insights you get from visualizing the data.
        If your `model` is different for different datasets, you can use a hyperparameter to switch between them.
        Make sure that your hyperparameter behaves as expecte and is being saved correctly in `hparams.yaml`.
        """
        self.time_embed = torch.nn.Sequential(
                torch.nn.Linear(1,16),
                torch.nn.ReLU(),
                torch.nn.Linear(16,8),
                torch.nn.ReLU(),
                torch.nn.Linear(8,3)
        )
        self.model = torch.nn.Sequential(
                torch.nn.Linear(6,128),
                torch.nn.ReLU(),
                
                # torch.nn.Linear(128,256),
                # torch.nn.ReLU(),
                
                # torch.nn.Linear(256,512),
                # torch.nn.ReLU(),
                
                # torch.nn.Linear(512,256),
                # torch.nn.ReLU(),

                # torch.nn.Linear(256,128),
                # torch.nn.ReLU(),

                torch.nn.Linear(128,64),
                torch.nn.ReLU(),

                torch.nn.Linear(64,32),
                torch.nn.ReLU(),
                
                torch.nn.Linear(32,16),
                torch.nn.ReLU(),

                torch.nn.Linear(16,8),
                torch.nn.ReLU(),

                torch.nn.Linear(8,3),
                # torch.nn.ReLU()
        )
        self.beta=None
        self.alpha=None
        self.alpha_bar=None
        self.sigma2=None
        """
        Be sure to save at least these 2 parameters in the model instance.
        """
        self.n_steps = n_steps
        self.n_dim = n_dim

        """
        Sets up variables for noise schedule
        """
        self.init_alpha_beta_schedule(lbeta, ubeta)

    def forward(self, x, t):
        """
        Similar to `forward` function in `nn.Module`. 
        Notice here that `x` and `t` are passed separately. If you are using an architecture that combines
        `x` and `t` in a different way, modify this function appropriately.
        """
        # print(x.shape)
        if not isinstance(t, torch.Tensor):
            t = torch.LongTensor([t]).expand(x.size(0))
        # print("Tshape:", type(t[0]))
        # print("Heyo",t.dtype)
        # print("T_embed shape: ",t.shape)
        t=torch.reshape(t,(torch.numel(t),1))
        t_embed = self.time_embed(t.float())
        # t_embed=torch.rehsape(t_embed,(t_embed))
        # t_embed = t_embed.reshape((1,t_embed.shape[0]))
        # print("x_shape: ", x.shape)
        return self.model(torch.cat((x, t_embed), dim=1).float())

    def init_alpha_beta_schedule(self, lbeta, ubeta):
        """
        Set up your noise schedule. You can perhaps have an additional hyperparameter that allows you to
        switch between various schedules for answering q4 in depth. Make sure that this hyperparameter 
        is included correctly while saving and loading your checkpoints.
        """
        self.beta = torch.linspace(lbeta, ubeta, self.n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2=self.beta
        # pass

    def q_sample(self, x, t,epsilon):
        """
        Sample from q given x_t.
        """
        # val=torch.randn(x)
        # print("t",t.shape)
        alpha_bar = torch.gather(self.alpha_bar,0,t.flatten())
        # print(alpha_bar.shape)
        alpha_bar=alpha_bar.reshape((x.shape[0],1))
        mean = alpha_bar ** 0.5 * x
        # print("meanshape",mean.shape)
        # mean = torch.squeeze(self.alpha_bar[t-1],dim=1) ** 0.5 * x
        # mean = gather(self.alpha_bar, t) ** 0.5 * x
        var = 1 - alpha_bar
        # var = 1 - gather(self.alpha_bar, t)
        return mean+(var** 0.5)*epsilon


    def p_sample(self, x, t, eps):
        """
        Sample from p given x_t.
        """
        # print(self.alpha_bar[t])
        alpha_bar = torch.squeeze(self.alpha_bar[t],dim=0)
        # alpha_bar = torch.gather(self.alpha_bar,1,t)
       
        # alpha = torch.squeeze(self.alpha[t-1],dim=1)
        alpha = self.alpha[t]
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (x - eps_coef * self.forward(x,t))
        var = self.sigma2[t]
        # var = torch.squeeze(self.sigma2[t-1],dim=1)
        

        # eps = torch.randn(x.shape)
        # Sample
        return mean + (var ** .5) * eps
        # pass

    def training_step(self, batch, batch_idx):
        """
        Implements one training step.
        Given a batch of samples (n_samples, n_dim) from the distribution you must calculate the loss
        for this batch. Simply return this loss from this function so that PyTorch Lightning will 
        automatically do the backprop for you. 
        Refer to the DDPM paper [1] for more details about equations that you need to implement for
        calculating loss. Make sure that all the operations preserve gradients for proper backprop.
        Refer to PyTorch Lightning documentation [2,3] for more details about how the automatic backprop 
        will update the parameters based on the loss you return from this function.

        References:
        [1]: https://arxiv.org/abs/2006.11239
        [2]: https://pytorch-lightning.readthedocs.io/en/stable/
        [3]: https://www.pytorchlightning.ai/tutorials
        """
        loss=0
        # for i in range(0,batch.shape[0]):
        #     t=torch.randint(low=1,high=self.n_steps+1,size=(1,1))
        #     val=torch.randn((1,self.n_dim))
        #     loss=loss+(torch.norm(val-(self.forward(self.q_sample(batch[i],t,val),t))))**2
        # loss=loss/batch.shape[0]
        t=torch.randint(low=0,high=self.n_steps,size=(batch.shape[0],1))
        val=torch.randn((batch.shape[0],self.n_dim))
        loss=(torch.norm(val-(self.forward(self.q_sample(batch,t,val),t))))**2
        return loss/batch.shape[0]

    def sample(self, n_samples, progress=False, return_intermediate=False):
        """
        Implements inference step for the DDPM.
        `progress` is an optional flag to implement -- it should just show the current step in diffusion
        reverse process.
        If `return_intermediate` is `False`,
            the function returns a `n_samples` sampled from the learned DDPM
            i.e. a Tensor of size (n_samples, n_dim).
            Return: (n_samples, n_dim)(final result from diffusion)
        Else
            the function returns all the intermediate steps in the diffusion process as well 
            i.e. a Tensor of size (n_samples, n_dim) and a list of `self.n_steps` Tensors of size (n_samples, n_dim) each.
            Return: (n_samples, n_dim)(final result), [(n_samples, n_dim)(intermediate) x n_steps]
        """
        fin=torch.zeros(self.n_steps,n_samples,self.n_dim)
        interm = []
        # for i in range(self.n_steps,0,-1):
        #     for j in range(0,n_samples):
        #         ini = torch.randn(1,self.n_dim)
        #         if i>1:
        #             z = torch.randn(1,self.n_dim)
        #         else:
        #             z = 0
        #         fin[i-1][j] = self.p_sample(fin[i-2],j,z)
        #         ini = self.p_sample(fin[i-1],i,z)
        #         fin[i-1][j] = ini
        # for j in range(0,n_samples):
        ini=torch.randn(n_samples,self.n_dim) 

        for i in range(self.n_steps-1,-1,-1):
            if i>1:
                z=torch.randn(n_samples,self.n_dim)
            else:
                z=0
            ini=self.p_sample(ini,i,z) 

            fin[i]=ini
        for i in range(self.n_steps-1,-1,-1):
            interm.append(fin[i])
        if return_intermediate:
            return fin[0], interm
        return fin[0]
        # pass

    def configure_optimizers(self):
        """
        Sets up the optimizer to be used for backprop.
        Must return a `torch.optim.XXX` instance.
        You may choose to add certain hyperparameters of the optimizers to the `train.py` as well.
        In our experiments, we chose one good value of optimizer hyperparameters for all experiments.
        """
        return torch.optim.Adam(self.parameters(),lr=0.01)
        pass
    
    