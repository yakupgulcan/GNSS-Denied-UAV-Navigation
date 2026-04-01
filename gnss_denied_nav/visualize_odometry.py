import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.interpolate import interp1d


class VisualReportGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.df = None
        self.df_real = None
        self.df_vis = None

    def load_data(self):
        if not os.path.exists(self.filename):
            print(f"Error: File not found -> {self.filename}")
            return False

        try:
            self.df = pd.read_csv(self.filename)
            print(f"Data loaded. Total rows: {len(self.df)}")

            # Split by type
            self.df_real = self.df[self.df['type'] == 'REAL'].sort_values('timestamp')
            self.df_vis = self.df[self.df['type'] == 'VISUAL'].sort_values('timestamp')

            if self.df_real.empty or self.df_vis.empty:
                print("Error: REAL or VISUAL dataset is empty.")
                return False

            return True
        except Exception as e:
            print(f"CSV read error: {e}")
            return False

    def remove_outliers_iqr(self, df, column, factor=1.5):
        """Simple IQR-based outlier removal."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        return df[(df[column] >= lower) & (df[column] <= upper)]

    def synchronize_and_calculate_error(self):
        """
        Interpolate ground-truth positions at the timestamps of the visual estimates
        and compute Euclidean error.
        """
        # Zero the time axis relative to the earliest timestamp (in seconds)
        start_time = min(self.df_real['timestamp'].min(), self.df_vis['timestamp'].min())

        t_real = self.df_real['timestamp'] - start_time
        x_real = self.df_real['x']
        y_real = self.df_real['y']

        t_vis = self.df_vis['timestamp'] - start_time
        x_vis = self.df_vis['x']
        y_vis = self.df_vis['y']

        # Build interpolation functions for X and Y over time
        # fill_value="extrapolate" handles small overruns at the boundaries
        f_x = interp1d(t_real, x_real, kind='linear', fill_value="extrapolate")
        f_y = interp1d(t_real, y_real, kind='linear', fill_value="extrapolate")

        # Ground-truth position at each visual estimate timestamp
        x_truth_at_vis_time = f_x(t_vis)
        y_truth_at_vis_time = f_y(t_vis)

        # Euclidean error
        errors = np.sqrt((x_vis - x_truth_at_vis_time)**2 + (y_vis - y_truth_at_vis_time)**2)

        return t_vis, x_vis, y_vis, x_real, y_real, errors

    def generate_report(self):
        if not self.load_data():
            return

        t_vis, x_vis, y_vis, x_real, y_real, errors = self.synchronize_and_calculate_error()

        # Convert Pandas Series to NumPy arrays for matplotlib compatibility
        x_real_np = x_real.to_numpy()
        y_real_np = y_real.to_numpy()
        x_vis_np = x_vis.to_numpy()
        y_vis_np = y_vis.to_numpy()

        # Statistics
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)
        max_error = np.max(errors)

        print("-" * 30)
        print("     PERFORMANCE REPORT")
        print("-" * 30)
        print(f"Processed frames : {len(x_vis)}")
        print(f"Mean error       : {mean_error:.2f} m")
        print(f"RMSE             : {rmse:.2f} m")
        print(f"Max error        : {max_error:.2f} m")
        print("-" * 30)

        # --- Visualization ---
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2)

        # 1. Trajectory comparison (left, spanning both rows)
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.plot(x_real_np, y_real_np, 'k-', linewidth=2, label='Ground Truth (GPS)', alpha=0.7)
        ax1.plot(x_vis_np, y_vis_np, 'r--.', markersize=4, linewidth=1, label='Visual estimate')
        ax1.scatter(x_real_np[0], y_real_np[0], c='g', marker='^', s=100, label='Start', zorder=5)
        ax1.scatter(x_real_np[-1], y_real_np[-1], c='b', marker='s', s=100, label='End', zorder=5)
        ax1.set_title(f"Trajectory Comparison (RMSE: {rmse:.2f}m)")
        ax1.set_xlabel("East [m]")
        ax1.set_ylabel("North [m]")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.axis('equal')

        # 2. Position error over time (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(t_vis.to_numpy(), errors.to_numpy(), 'r-', alpha=0.7)
        ax2.fill_between(t_vis.to_numpy(), 0, errors.to_numpy(), color='red', alpha=0.1)
        ax2.set_title("Position Error over Time")
        ax2.set_ylabel("Error [m]")
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.axhline(mean_error, color='blue', linestyle='--', label=f'Mean: {mean_error:.2f}m')
        ax2.legend()

        # 3. Error histogram (bottom right)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(errors.to_numpy(), bins=20, color='gray', edgecolor='black', alpha=0.7)
        ax3.set_title("Error Distribution (Histogram)")
        ax3.set_xlabel("Error [m]")
        ax3.set_ylabel("Count")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Usage: python visualize_odometry.py <log_file.csv>
    file_path = ''
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    reporter = VisualReportGenerator(file_path)
    reporter.generate_report()