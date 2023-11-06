model_config = dict(
    projection_head_kwargs = dict(
        hidden_dim = 2048,
        bottleneck_dim = 256,
        output_dim = 2048,
        batch_norm = True,
        freeze_last_layer = -1,
        norm_last_layer = True
    ),
    loss_kwargs = dict(
        warmup_teacher_temp = 0.04,
        teacher_temp = 0.04,
        warmup_teacher_temp_epochs = 30,
        student_temp = 0.1,
        center_momentum = 0.9
    )
)