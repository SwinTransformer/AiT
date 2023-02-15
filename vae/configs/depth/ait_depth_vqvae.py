image_size = 480
model = dict(
    image_size=image_size,
    num_resnet_blocks=2,
    downsample_ratio=32,
    num_tokens=128,
    codebook_dim=512,
    hidden_dim=16,
    use_norm=False,
    channels=1,
    train_objective='regression',
    max_value=10.,
    residul_type='v1',
    loss_type='mse_ignore_zero',
)

train_setting = dict(
    output_dir='outputs',
    data=dict(
        is_train=True,
        data_path='data/nyu_depth_v2',
        filenames_path='./dataset/filenames',
        mask=True,
        mask_ratio=0.5,
        mask_patch_size=16,
        crop_size=(image_size, image_size),
    ),
    opt_params=dict(
        epochs=20,
        batch_size=8,
        learning_rate=3e-4,
        lr_decay_rate=0.98,
        schedule_step=500,
        schedule_type='exp',
    )
)

test_setting = dict(
    data=dict(
        data_path='data/nyu_depth_v2',
        filenames_path='./dataset/filenames',
    ),
)
