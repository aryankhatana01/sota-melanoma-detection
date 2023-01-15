import torch

class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    batch_size = 16
    image_size = 512
    n_epochs = 10
    lr = 1e-4
    use_meta = True
    out_dim = 4
    use_scheduler = True
    scheduler = 'CosineAnnealingLR'
    scheduler_params = dict(
        T_max=10,
        eta_min=1e-6,
        last_epoch=-1,
    )
    warmup_epochs = 1
    warmup_factor = 10
    warmup_method = 'linear'
    n_fold = 5
    seed = 42
    model_name = 'tf_efficientnet_b6_ns'
    model_params = dict(
        pretrained=True,
        drop_rate=0.5,
        drop_path_rate=0.2,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam
    optimizer_params = dict(
        lr=lr,
        weight_decay=1e-6,
    )
    verbose = True