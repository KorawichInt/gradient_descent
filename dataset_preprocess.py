import pandas as pd

# Filter columns 
def filtering_dataset(csv_path):
    df = pd.read_csv(csv_path)
    filtered_df = df[["Hours Studied", "Sample Question Papers Practiced", "Previous Scores"]]
    filtered_df.columns = ["x1", "x2", "y"]
    # print(filtered_df.head(5))
    return filtered_df

# Split dataset
def spliting_data(filtered_df):
    # print(filtered_df.head(5))
    shuffled_df = filtered_df.sample(frac=1)
    # print(shuffled_df.head(5))

    total_rows = shuffled_df.shape[0]
    train_size = int(total_rows*0.8)
 
    # Split data into test and train
    train_df = shuffled_df[0:train_size]
    # print(train.shape[0])
    test_df = shuffled_df[train_size:]
    # print(test.shape[0])
    train_df.to_csv("train_dataset", index=False) 
    test_df.to_csv("test_dataset", index=False) 


if __name__ == "__main__":
    csv_path = "dataset.csv"
    filtered_df = filtering_dataset(csv_path)
    spliting_data(filtered_df)