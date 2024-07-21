import pandas as pd

# Filter columns 
def filtering_dataset(input_path, selected_columns):
    df = pd.read_csv(input_path)
    filtered_df = df[selected_columns]
    filtered_df.columns = ["x1", "x2", "y"]
    # print(filtered_df.head(5))
    return filtered_df

# Split dataset
def spliting_data(filtered_df, full_dir):
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
    train_df.to_csv(f"{full_dir}/train_dataset.csv", index=False) 
    test_df.to_csv(f"{full_dir}/test_dataset.csv", index=False) 


if __name__ == "__main__":
    save_path = "dataset"
    num_dir = input("Enter which number of num directory you want to prepocess : ")
    full_dir = f"{save_path}{num_dir}"
    input_path = f"{full_dir}/dataset.csv"
    # selected_columns = ["Hours Studied", "Sample Question Papers Practiced", "Previous Scores"]
    selected_columns = ["number_courses", "time_study", "Marks"]
    filtered_df = filtering_dataset(input_path, selected_columns)
    spliting_data(filtered_df, full_dir)