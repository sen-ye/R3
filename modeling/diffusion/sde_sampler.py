import math
import torch

LOG_PI = torch.log(2*torch.tensor(math.pi))

class GaussianDistribution:
    """A class representing a Gaussian (Normal) distribution with methods for sampling and probability calculations.
    
    This class provides functionality for working with Gaussian distributions including:
    - Sampling from the distribution
    - Computing log probabilities
    - Calculating KL divergence between distributions
    
    Attributes:
        mean (torch.Tensor): Mean of the Gaussian distribution
        std (torch.Tensor): Standard deviation of the Gaussian distribution
        log_std (torch.Tensor): Log of the standard deviation
        generators (list, optional): List of random number generators for sampling
    """
    def __init__(self, mean, std, log_std, generators=None):
        """Initialize the Gaussian distribution.
        
        Args:
            mean (torch.Tensor): Mean of the distribution
            std (torch.Tensor): Standard deviation of the distribution
            log_std (torch.Tensor): Log of the standard deviation
            generators (list, optional): List of random number generators for sampling
        """
        self.mean = mean
        self.std = std
        self.log_std = log_std
        self.generators = generators

    def sample(self):
        """Sample from the Gaussian distribution.    
        Returns:
            torch.Tensor: Sampled values from the distribution
            
        Raises:
            ValueError: If generators are provided but their length doesn't match the batch size
        """
        size = self.mean.shape
        if self.generators is not None:
            if len(self.generators) == size[0]:
                rand_tensor = []
                for i in range(size[0]):
                    rand_tensor.append(torch.randn(size[1:], generator=self.generators[i]))
                rand_tensor = torch.stack(rand_tensor, dim=0)
            else:
                raise ValueError("Generators must be of the same size as the batch size.")
        else:
            rand_tensor = torch.randn(size)
        return rand_tensor.to(self.mean) * self.std + self.mean

    def log_prob(self, x, real=False):
        """Compute the log probability of samples under this distribution.
        
        Args:
            x (torch.Tensor): Input samples
            real (bool): Whether to compute the log probability of the real distribution. 
            In most cases, we only care about the relative difference between two distributions, so we set real to False.
        Returns:
            torch.Tensor: Log probabilities of the samples
        """
        dims_except_batch = list(range(1,len(self.mean.shape)))
        bias = 2 * self.log_std + LOG_PI if real else 0
        return -0.5*torch.mean(
            (x - self.mean)**2 / (self.std**2) + bias,
            dim=dims_except_batch
        )
    
    def kl_divergence(self, other:"GaussianDistribution"):
        """Compute the KL divergence between this distribution and another Gaussian distribution.
        
        Args:
            other (GaussianDistribution): Another Gaussian distribution to compute KL divergence with
            
        Returns:
            torch.Tensor: KL divergence between the two distributions
        """
        dims_except_batch = list(range(1,len(self.mean.shape)))
        return torch.mean((self.mean - other.mean)**2 / (other.std**2),dim=dims_except_batch)
    
class SDESampler:
    def __init__(self,eta_mode:str='constant', constant_eta:float=0.2,model_output_type:str='velocity'):
        """
        Args:
            eta_mode: str, mode of eta, can be 'constant' or 'monotonic'
            constant_eta: float, constant eta
            model_output_type: str, type of model output, can be 'velocity' only.
        """
        self.eta_mode = eta_mode
        self.constant_eta = constant_eta
        self.model_output_type = model_output_type
        assert self.model_output_type in ['velocity']
    
    def step(self,x_t,t,dt,model_output):
        if not isinstance(dt,torch.Tensor):
            dt = torch.tensor(dt)
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t)
        if self.model_output_type == 'velocity':
            x_prev = self.get_x_t_distribution(x_t,t,dt,model_output)
            x_prev_sample = x_prev.sample()
            return x_prev_sample.to(x_t.dtype),x_prev.log_prob(x_prev_sample)
    
    @torch.autocast('cuda',enabled=False)
    def get_x_t_distribution(self,x_t,t,dt,model_output):
        if t.ndim <2:
            t = t.reshape(-1,1)
        if dt.ndim <2:
            dt = dt.reshape(-1,1)
        pred_x_0 = x_t - t * model_output.to(x_t.device)
        score_estimate = -(x_t-pred_x_0*(1 - t))/t**2
        log_term = -0.5 * self.get_eta(t)**2 * score_estimate
        x_t_mean = x_t - (model_output.to(x_t.device) + log_term) * dt # velocity pointing from data to noise
        std_dev_t = self.get_eta(t) * torch.pow(dt,0.5)
        return GaussianDistribution(x_t_mean, std_dev_t, log_std=torch.log(std_dev_t))
    
    def get_eta(self,t):
        if self.eta_mode == 'constant':
            return self.constant_eta
        elif self.eta_mode == 'monotonic':
            return torch.sqrt(t / (1 - torch.where(t >= 0.95, 0.95, t)))*self.constant_eta
        else:
            raise NotImplementedError(f"Unknown eta mode: {self.eta_mode}")
        
    def __repr__(self):
        return f"SDESampler(eta_mode={self.eta_mode}, constant_eta={self.constant_eta}, model_output_type={self.model_output_type})"