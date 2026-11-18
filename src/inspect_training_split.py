
from sklearn.model_selection import GroupShuffleSplit

import train_models


def main():
    data = train_models.load_data()

    X = train_models.prepare_features(data)
    y = data["label"].astype(int)
    groups = data["trip_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=train_models.TEST_SIZE,
        random_state=train_models.RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    training_trips = sorted(
        data.iloc[train_indices]["trip_id"].unique()
    )

    validation_trips = sorted(
        data.iloc[test_indices]["trip_id"].unique()
    )

    print("\n" + "=" * 60)
    print("PATHSNAP - ORIGINAL MODEL TRAINING SPLIT")
    print("=" * 60)

    print("\nTraining trips:")
    print(training_trips)

    print("\nValidation trips:")
    print(validation_trips)

    print("\nNumber of training trips:", len(training_trips))
    print("Number of validation trips:", len(validation_trips))

    trip_one = "00000001"

    print("\n--- TRIP 1 MEMBERSHIP ---")

    if trip_one in training_trips:
        print("Trip 1 was used for TRAINING.")
    elif trip_one in validation_trips:
        print("Trip 1 was used for VALIDATION.")
    else:
        print("Trip 1 was not included in the model dataset.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()