"""Full author-reference architectures with the analysis PF feature contract."""
import importlib.util
from pathlib import Path


def architecture(name, profile='full'):
    if profile not in ['full', 'compact']:
        raise ValueError('Unknown model profile: '+profile)
    if profile == 'compact':
        if name == 'ParticleNet':
            return dict(input_dims=12, num_classes=2,
                        conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                        fc_params=[(128, .1)], use_fusion=True, use_fts_bn=False,
                        use_counts=True, for_inference=False)
        if name == 'ParticleTransformer':
            return dict(input_dim=12, num_classes=2, pair_input_dim=4,
                        use_pre_activation_pair=False, embed_dims=[64, 128, 64],
                        pair_embed_dims=[32, 32], num_heads=4, num_layers=2,
                        num_cls_layers=1, fc_params=[(64, .1)], activation='gelu',
                        trim=False, use_amp=False)
    if name == 'ParticleNet':
        return dict(input_dims=12, num_classes=2,
                    conv_params=[(16, (64, 64, 64)), (16, (128, 128, 128)),
                                 (16, (256, 256, 256))],
                    fc_params=[(256, .1)], use_fusion=False, use_fts_bn=True,
                    use_counts=True, for_inference=False)
    if name == 'ParticleTransformer':
        return dict(input_dim=12, num_classes=2, pair_input_dim=4,
                    use_pre_activation_pair=False, embed_dims=[128, 512, 128],
                    pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8,
                    num_cls_layers=2, fc_params=[], activation='gelu',
                    cls_block_params=dict(dropout=0., attn_dropout=0., activation_dropout=0.),
                    trim=False, use_amp=False)
    raise ValueError('Unknown architecture: '+name)


def model_for(name, directory, profile='full'):
    path = Path(directory)/(name+'.py')
    spec = importlib.util.spec_from_file_location('full_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)(**architecture(name, profile))


def forward(name, model, x, v, mask, points):
    return model(points, x, mask.float()) if name == 'ParticleNet' else model(x, v, mask)
