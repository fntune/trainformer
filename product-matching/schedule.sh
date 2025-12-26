python3 run.py -m \
    dataloader.image_size=384 \
    loss=arcface \
    optimizer=lamb \
    model.model_name=swin_base_patch4_window12_384_in22k \
    optimizer.lr=0.0001 \
    scheduler.lr_scheduler.warmup_factor=0.2 \
    max_epochs=20 \
    loss.margin=33.3,40.1,45.8

python3 run.py dataloader.image_size=384 loss=arcface optimizer=lamb model.model_name=swin_base_patch4_window12_384_in22k optimizer.lr=0.0001 scheduler.lr_scheduler.warmup_factor=0.2