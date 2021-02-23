import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def load_data(path_csv, sep=";"):
     dataset = pd.DataFrame(pd.read_csv(path_csv,sep=sep))
     print("\nShape dataset {}\n".format(dataset.shape))
     print(dataset.head())
     return dataset


def shuffle_split_data(path_csv, x_col, y_col, sep=";", validation=True, val_size=0.20, test_size=0.10):

    dataset = load_data(path_csv, sep)

    X = dataset[x_col]
    y = dataset[y_col]

    sss1 = StratifiedShuffleSplit(n_splits=1, random_state=0, test_size=test_size)
    for train_idx, test_idx in sss1.split(X, y):
        X_train, X_test =X.iloc[list(train_idx)], X.iloc[list(test_idx)]
        y_train, y_test =y.iloc[list(train_idx)], y.iloc[list(test_idx)]

    if validation:
        sss2 = StratifiedShuffleSplit(n_splits=1,  random_state=0, test_size=val_size)
        for train_idx, val_idx in sss2.split(X_train, y_train):
            X_train, X_val =X.iloc[list(train_idx)], X.iloc[list(val_idx)]
            y_train, y_val =y.iloc[list(train_idx)], y.iloc[list(val_idx)]

        print("Training set: {}\n".format(len(y_train)))
        print("Validation set: {}\n".format(len(y_val)))
        print("Test set: {}\n".format(len(y_test)))


        return X_train, y_train, X_val, y_val, X_test, y_test

    print("Training set: {}\n".format(len(y_train)))
    print("Test set: {}\n".format(len(y_test)))

    return X_train, y_train, X_test, y_test


def split_data(path_csv, x_col, y_col, sep=";", validation=True, val_size=0.20, test_size=0.10):

    dataset = load_data(path_csv, sep)

    X = dataset[x_col]
    y = dataset[y_col]

    sss1 = train_test_split(n_splits=1, random_state=0, test_size=test_size)
    for train_idx, test_idx in sss1.split(X, y):
        X_train, X_test =X.iloc[list(train_idx)], X.iloc[list(test_idx)]
        y_train, y_test =y.iloc[list(train_idx)], y.iloc[list(test_idx)]

    if validation:
        sss2 = train_test_split(n_splits=1,  random_state=0, test_size=val_size)
        for train_idx, val_idx in sss2.split(X_train, y_train):
            X_train, X_val =X.iloc[list(train_idx)], X.iloc[list(val_idx)]
            y_train, y_val =y.iloc[list(train_idx)], y.iloc[list(val_idx)]

        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_train, y_train, X_test, y_test
