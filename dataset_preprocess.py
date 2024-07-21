import pandas as pd

# Filter columns 
def filtering_dataset(csv_path):
    df = pd.read_csv(csv_path)
    filtered_df = df[["Hours Studied", "Sample Question Papers Practiced", "Previous Scores"]]
    filtered_df.columns = ["x1", "x2", "y"]
    print(filtered_df.head(5))

def spliting_data(filtered_df):
    pass

if __name__ == "__main__":
    csv_path = "dataset.csv"
    filtered_df = filtering_dataset(csv_path)