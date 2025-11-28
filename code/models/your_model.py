from models.base_model import ClassificationModel
import torch
import torch.nn as nn
from functools import partial

from models.mae import MaskedAutoencoder
from timm.models.vision_transformer import Block

# Namespace(architecture='transformer', exp_setting_key='pt_syn01', ssl='mae', 
# pretrained_weight_key=None, num_lead=1, target_task=None, target_freq=500, 
# max_duration=10, n_workers=4, scale_type='per_sample', batch_size=512, 
# optimizer='adam', optimizier_patience=5, scheduler='cosine-01', 
# aug_mask_ratio=0.5, max_shift_ratio=0.5, backbone_out_dim=64, 
# emb_dim=128, dataset='syn_ecg-04', learning_rate=0.0001, 
# eval_every=1000000, save_model_every=1000000, total_samples=1000000000, 
# dump_every='1*1e6', data_lim=1000000, val_lim=5000, mae_mask_ratio=0.75, 
# depth=10, heads=8, mlp_ratio=4.0, dec_emb_dim=128, dec_depth=2, dec_heads=16, 
# chunk_len=50, src_freq=500, downsample=1, seed=7, host='syn_ecg-shono', 
# device='cuda:1')

class SimECGMAE(ClassificationModel):
    
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape):
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape

        self.model = self._prep_model()

    def _set_weight(self, model):
        weight_file = "../data/net.pth"

        state_dict = dict(torch.load(weight_file, map_location="cpu")) # OrderedDict -> dict
        
        old_keys = list(state_dict.keys())
        for key in old_keys:
            new_key = key.replace("backbone.", "")

            if key.startswith("fc."):
                state_dict.pop(key)
                continue

            state_dict[new_key] = state_dict.pop(key)

        model.backbone.load_state_dict(state_dict)
        return model

    def _prep_model(self):
        seqlen = 5000
        chunk_len = 50
        emb_dim = 128

        mae = MaskedAutoencoder(
            Block=Block,
            seqlen=seqlen, 
            chunk_size=chunk_len,
            in_channels=1,
            emb_dim=emb_dim, 
            depth=10, 
            num_heads=8,
            decoder_emb_dim=128, 
            decoder_depth=2,
            decoder_num_heads=16,
            mlp_ratio=4, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        )

        head = HeadModule(emb_dim)
        model = Predictor(
            mae, 
            head, 
        )
        model = self._set_weight(model)
        return model

    def fit(self, X_train, y_train, X_val, y_val):
        pass

    def predict(self, X):
        pass



class Predictor(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module,
        head: nn.Module
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _, _ = self.backbone.forward_encoder(x, mask_ratio=0)
        h = self._select_features(h)
        h = self.fc(h)
        return self.head(h)

    def _select_features(self, h: torch.Tensor) -> torch.Tensor:
        return h[:, 0]

class HeadModule(nn.Module):
    def __init__(self, in_dim: int, head_dim: int, n_head_layer: int):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_head_layer == 0:
            head_dim = in_dim
        else:
            self.layers.append(self._create_layer(in_dim, head_dim))        
            for _ in range(n_head_layer - 1):
                self.layers.append(self._create_layer(head_dim, head_dim))
        
        self.fc_final = nn.Linear(head_dim, 1)

    def _create_layer(self, in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.fc_final(x)
