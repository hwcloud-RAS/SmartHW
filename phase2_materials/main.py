import argparse


def generate_features(data_path: str, feature_path: str):
    """
    Generate features from the original data.

    :param data_path: Path to the data file.
    :param feature_path: Path to the generated features.
    """
    print(f"Generate features from {data_path} and save it to {feature_path}")


def training_pipeline(feature_path: str, model_path: str):
    """
    Train a model using the features.

    :param feature_path: Path to the generated features.
    :param model_path: Path to the trained model.
    """
    print(f"Train a model using {feature_path} and save it to {model_path}")


def inference_pipeline(feature_path: str, model_path: str, output_path: str):
    """
    Make predictions using the trained model.

    :param feature_path: Path to the generated features.
    :param model_path: Path to the trained model.
    :param output_path: Path to the output predictions.
    """
    print(f"Make predictions using {feature_path} and {model_path} and save it to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument(
        "--process",
        type=str,
        default="generate_features",
        choices=["generate_features", "train_model", "inference"],
        help="Process to run",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="smartmem/stage2_feather",
        help="Path to the data file",
    )
    parser.add_argument(
        "--feature_path",
        type=str,
        default="smartmem/stage2_features",
        help="Path to the feature file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="smartmem/stage2_model",
        help="Path to the model file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="smartmem/submission.csv",
        help="Path to the output file",
    )

    args = parser.parse_args()

    if args.process == "generate_features":
        generate_features(args.data_path, args.feature_path)
    elif args.process == "train_model":
        training_pipeline(args.feature_path, args.model_path)
    elif args.process == "inference":
        inference_pipeline(args.feature_path, args.model_path, args.output_path)
