import torch
import torch.nn as nn
import einops


class VAE(nn.Module):
    def __init__(self, in_dims: int = 16000, h_dims: int = 10, bias_last_fc: bool = True, image: bool = True):
        super().__init__()

        self.image = image
        self.bias_last_fc = bias_last_fc
        self.h_dims = h_dims

        def make_block(in_dims, out_dims, dropout):
            block = [
                nn.Linear(in_dims, out_dims),
                nn.LayerNorm(out_dims),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            return block

        self.encoder = nn.Sequential(
            *make_block(in_dims, 1024, 0.1),
            *make_block(1024, 512, 0.1),
            *make_block(512, 256, 0.1),
            *make_block(256, 128, 0.1),
        )

        # A fully connected layer that takes the output from the previous layer (of size 128) and maps it to the mean vector (mu) of size h_dims
        self.fc_mu = nn.Linear(128, h_dims)
        # Another fully connected layer that maps the same 128-dimensional input to the log variance vector (log_var) of the latent space.
        self.fc_log_var = nn.Linear(128, h_dims)

        self.in_decoder = nn.Linear(h_dims, 128)
        self.decoder = nn.Sequential(
            *make_block(128, 256, 0.1),
            *make_block(256, 512, 0.1),
            *make_block(512, 1024, 0.1),
            nn.Linear(1024, in_dims, bias=self.bias_last_fc)
        )

        self.init_weights()

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
    # samples from the latent distribution using the mean and log variance
    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.in_decoder(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        if self.image:
            x = einops.rearrange(x, "B C H W -> B C (H W)")
        x = x.squeeze()
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        print('Initialization complete.')

    @staticmethod
    def get_fake_sound_data(B, C, SR):
        return torch.randn(B, C, SR)
    
    @staticmethod
    def get_fake_img_data(B, C, H, W):
        return torch.randn(B, C, H, W)



if __name__ == "__main__":

    sound = VAE.get_fake_sound_data(4, 1, 8000)

    model = VAE(
        in_dims=sound.shape[2],
        h_dims=10, 
        bias_last_fc=True, 
        image=False
    )
    
    out, mu, log_var = model(sound)
    print(out)
