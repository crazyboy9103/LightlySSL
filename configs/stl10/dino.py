model_config = dict(
    projection_head_kwargs = dict(hidden_dim=512, bottleneck_dim=64, output_dim=2048),
    loss_kwargs = dict(warmup_teacher_temp_epochs=5)
)