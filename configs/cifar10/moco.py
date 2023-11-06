model_config = dict(
    projection_head_kwargs = dict(
        hidden_dim=512, 
        output_dim=128
    ),
    loss_kwargs = dict(
        temperature=0.1,
        memory_bank_size=4096
    )
)