import pandas as pd
from sklearn import model_selection
from . import config


if __name__ == "__main__":
    df = pd.read_pickle(config.TRAINING_FILE)
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, "kfold"] = fold

    df.to_csv(config.TRAINING_FOLDS, index=False)
