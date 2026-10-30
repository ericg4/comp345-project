"""Write the shared 70/15/15 nevus vs. melanoma split, grouped by lesion_id.

Already run and committed; every model uses it.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import CLASSES, HAM_DIR, SPLIT_CSV

SEED = 0


def main():
    metadata = pd.read_csv(HAM_DIR / "HAM10000_metadata.csv")
    data = metadata[metadata["dx"].isin(CLASSES)].reset_index(drop=True)
    data["label"] = data["dx"].map({name: i for i, name in enumerate(CLASSES)})

    # 20 class-stratified folds of whole lesions; 14 / 3 / 3 folds give 70 / 15 / 15.
    fold_names = ["train"] * 14 + ["val"] * 3 + ["test"] * 3
    folds = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    data["split"] = ""
    for fold, (_, rows) in enumerate(folds.split(data, data["label"], data["lesion_id"])):
        data.loc[rows, "split"] = fold_names[fold]
    assert (data.groupby("lesion_id")["split"].nunique() == 1).all(), "lesion in multiple splits"

    split = data[["image_id", "lesion_id", "dx", "label", "split"]].sort_values(["split", "image_id"])
    SPLIT_CSV.parent.mkdir(exist_ok=True)
    split.to_csv(SPLIT_CSV, index=False)
    print(f"Saved {SPLIT_CSV}")
    print(pd.crosstab(split["split"], split["dx"], margins=True))


if __name__ == "__main__":
    main()
