import sys
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def main(csv_path, output_path="scaler.pkl"):
    df = pd.read_csv(csv_path)
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])
    joblib.dump(scaler, output_path)
    print(f"Saved scaler fit on {len(df)} rows to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_scaler.py path/to/Mall_Customers.csv [output.pkl]")
        sys.exit(1)
    csv_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) > 2 else "scaler.pkl"
    main(csv_arg, output_arg)
