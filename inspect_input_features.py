from utils.model_loader import load_checkpoint_model


for checkpoint_day in [7, 14, 21, 30]:
    preprocessing, _ = load_checkpoint_model(
        checkpoint_day
    )

    print("\n" + "=" * 60)
    print(f"DAY {checkpoint_day} RAW INPUT FEATURES")
    print("=" * 60)

    feature_names = preprocessing.feature_names_in_

    print(f"Number of raw features: {len(feature_names)}")

    for feature_name in feature_names:
        print(feature_name)