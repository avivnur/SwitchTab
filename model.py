import torch
from torch import nn
from utils import Encoder, Projector, Decoder, Predictor, feature_corruption

class SwitchTabModel(nn.Module):
    def __init__(self, feature_size, num_classes, num_heads=2):
        super(SwitchTabModel, self).__init__()
        self.encoder = Encoder(feature_size, num_heads)
        self.projector_s = Projector(feature_size)
        self.projector_m = Projector(feature_size)
        self.decoder = Decoder(2 * feature_size, feature_size)  # Assuming concatenation of salient and mutual embeddings
        self.predictor = Predictor(feature_size, num_classes)

    def forward(self, x1, x2):
        # Feature corruption is not included in the model itself and should be applied to the data beforehand
        z1_encoded = self.encoder(x1)
        z2_encoded = self.encoder(x2)

        s1_salient = self.projector_s(z1_encoded)
        m1_mutual = self.projector_m(z1_encoded)
        s2_salient = self.projector_s(z2_encoded)
        m2_mutual = self.projector_m(z2_encoded)

        x1_reconstructed = self.decoder(torch.cat((m1_mutual, s1_salient), dim=1))
        x2_reconstructed = self.decoder(torch.cat((m2_mutual, s2_salient), dim=1))
        x1_switched = self.decoder(torch.cat((m2_mutual, s1_salient), dim=1))
        x2_switched = self.decoder(torch.cat((m1_mutual, s2_salient), dim=1))

        return x1_reconstructed, x2_reconstructed, x1_switched, x2_switched

    def get_salient_embeddings(self, x):
        z_encoded = self.encoder(x)
        s_salient = self.projector_s(z_encoded)
        return s_salient
