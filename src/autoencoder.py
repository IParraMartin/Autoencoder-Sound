import torch
import torch.nn as nn
import einops


class Autoencoder(nn.Module):
    def __init__(self, in_dims: int = 16000, h_dims: int = 10, bias_last_fc: bool = True, image: bool = True):
        super().__init__()

        self.image = image
        self.bias_last_fc = bias_last_fc

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
            *make_block(128, h_dims, 0.1)
        )

        self.latent_space = nn.Linear(h_dims, h_dims)

        self.decoder = nn.Sequential(
            *make_block(h_dims, 128, 0.1),
            *make_block(128, 256, 0.1),
            *make_block(256, 512, 0.1),
            *make_block(512, 1024, 0.1),
            nn.Linear(1024, in_dims, bias=self.bias_last_fc)
        )

        self.init_weights()

    def forward(self, x):

        if self.image:
            x = einops.rearrange(x, "B C H W -> B C (H W)")

        x = x.squeeze()

        encoded = self.encoder(x)
        latent = self.latent_space(encoded)
        decoded = self.decoder(latent)
        return decoded, latent
    
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

    img = Autoencoder.get_fake_img_data(4, 1, 16, 16)
    sound = Autoencoder.get_fake_sound_data(4, 1, 16000)

    model = Autoencoder(in_dims=16000, h_dims=10, bias_last_fc=True, image=False)
    
    out, latent = model(sound)
    print(latent)
    print(out)
