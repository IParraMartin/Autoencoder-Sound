import torch
import torch.nn as nn
import einops


class Autoencoder(nn.Module):

    def __init__(self, in_dims: int = 16000, h_dims: int = 10, is_image: bool = True):
        super().__init__()

        self.is_image = is_image

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

        self.latent = nn.Linear(128, h_dims)
        self.latent_relu = nn.ReLU()

        self.decoder = nn.Sequential(
            *make_block(h_dims, 128, 0.1),
            *make_block(128, 256, 0.1),
            *make_block(256, 512, 0.1),
            *make_block(512, 1024, 0.1),
        )

        nn.Linear(1024, in_dims)

        self.init_weights()

    def forward(self, x):
        if self.is_image:
            x = einops.rearrange(x, "B C H W -> B C (H W)")
        encoded = self.encoder(x)
        latent = self.latent_relu(self.latent(encoded))
        decoded = self.decoder(latent)
        return decoded
    
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

    sound = Autoencoder.get_fake_sound_data(4, 1, 8000)

    model = Autoencoder(
        in_dims=sound.shape[2],
        h_dims=10,
        is_image=False
    )
    
    decoded = model(sound)
    print(decoded)
