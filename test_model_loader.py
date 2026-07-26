from utils.model_loader import load_checkpoint_model


for checkpoint_day in [7, 14, 21, 30]:
    preprocessing, model = load_checkpoint_model(
        checkpoint_day
    )

    print(
        f"Day {checkpoint_day}: "
        f"{type(preprocessing).__name__}, "
        f"{type(model).__name__}, "
        f"{model.n_features_in_} features"
    )