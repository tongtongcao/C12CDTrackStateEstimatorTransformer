import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy.stats import norm
import pandas as pd
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2.0 * sigma ** 2))


class Plotter:
    def __init__(self, print_dir="", end_name=""):
        self.print_dir = print_dir
        self.end_name = end_name
        self.state_parameter_name = ["d0", "phi0", "kappa", "z0", "tandip"]
        os.makedirs(self.print_dir, exist_ok=True)

    # ------------------------------------------------------------
    # training loss
    # ------------------------------------------------------------
    def plotTrainLoss(self, tracker):

        train_losses = tracker.train_losses
        val_losses = tracker.val_losses

        print("Final train loss:", train_losses[-1])
        print("Final val loss:", val_losses[-1])

        plt.figure(figsize=(10, 6))

        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        #plt.yscale("log")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        outname = os.path.join(self.print_dir, f"loss_{self.end_name}.png")
        plt.savefig(outname, dpi=200)
        plt.close()

    # ------------------------------------------------------------
    # residual
    # ------------------------------------------------------------
    def plot_residuals(self, preds, targets, scale=1.0, fit_range_factor={"d0": 0.8, "phi0": 0.8, "kappa": 0.8, "z0": 0.8, "tandip": 0.8}):

        targets = np.asarray(targets)
        preds = np.asarray(preds)

        # Momentum magnitude is scaled by the scale factor of magnetic field
        preds = preds.copy()
        preds[:, 2] /= scale

        plot_ranges = {
            "d0": (-0.1, 0.1),
            "phi0": (-0.3, 0.3),
            "kappa": (-1, 1),
            "z0": (-1.5, 1.5),
            "tandip": (-0.2, 0.2)
        }

        # --------------------------------------------------------
        # Fit range factors
        # --------------------------------------------------------
        if isinstance(fit_range_factor, (int, float)):
            fit_range_factors = {
                name: fit_range_factor
                for name in self.state_parameter_name
            }
        else:
            fit_range_factors = {
                name: fit_range_factor.get(name,2.0)
                for name in self.state_parameter_name
            }

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()

        fit_results = []

        for i, name in enumerate(self.state_parameter_name):

            # ---------------------------------------
            # Residuals
            # ---------------------------------------
            diff = preds[:, i] - targets[:, i]

            if i == 1:
                diff = np.arctan2(
                    np.sin(diff),
                    np.cos(diff)
                )

            diff = diff[np.isfinite(diff)]

            plot_min, plot_max = plot_ranges[name]

            # Initial estimates from all residuals
            mu0 = np.mean(diff)
            sigma0 = np.std(diff)

            factor = fit_range_factors[name]

            fit_min = mu0 - factor * sigma0
            fit_max = mu0 + factor * sigma0

            # ---------------------------------------
            # Histogram
            # ---------------------------------------
            counts, bins, _ = axes[i].hist(
                diff,
                bins=60,
                range=(plot_min, plot_max),
                color="royalblue",
                alpha=0.7,
                label="Residuals"
            )

            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # Fit only within mean ± 2σ
            mask = (
                    (bin_centers >= fit_min)
                    & (bin_centers <= fit_max)
                    & (counts > 0)
            )

            x_fit = bin_centers[mask]
            y_fit = counts[mask]

            if len(x_fit) < 5:
                fit_results.append((name, np.nan, np.nan))
                axes[i].set_title(f"{name} residual")
                continue

            # Initial guess
            A0 = np.max(y_fit)

            # Poisson uncertainties
            sigma_y = np.sqrt(np.maximum(y_fit, 1.0))

            try:
                popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=[A0, mu0, sigma0], sigma=sigma_y, absolute_sigma=True,
                                       bounds=([0.0, fit_min, 1e-8], [np.inf, fit_max, np.inf]))

                A, mu, sigma = popt

                fit_results.append((name, mu, sigma))

                # Draw fitted Gaussian
                x_curve = np.linspace(plot_min, plot_max, 500)
                y_curve = gaussian(x_curve, A, mu, sigma)

                axes[i].plot(x_curve, y_curve, "r--", lw=3,
                             label=(f"Gaussian Fit:\n" f"μ={mu:+.3e}\n" f"σ={sigma:.3e}")
                             )

                # Show fit window
                axes[i].axvline(fit_min, color="green", ls=":", lw=1.5)
                axes[i].axvline(fit_max, color="green", ls=":", lw=1.5)

            except RuntimeError:

                fit_results.append((name, np.nan, np.nan))

                axes[i].text(0.05, 0.95, "Fit failed", transform=axes[i].transAxes, color="red", fontsize=11,
                             verticalalignment="top")

            axes[i].set_title(f"{name} residual")
            axes[i].set_xlabel("Prediction − Truth")
            axes[i].set_ylabel("Counts")
            axes[i].set_xlim(plot_min, plot_max)
            axes[i].legend(loc="upper left")

        # Hide unused subplot
        for j in range(len(self.state_parameter_name), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        outname = os.path.join(
            self.print_dir,
            f"track_diff_{self.end_name}.png"
        )
        plt.savefig(outname, dpi=200)
        plt.close()

        # --------------------------------------------------
        # Save Gaussian fit results to CSV
        df = pd.DataFrame(fit_results, columns=['Variable', 'Mean (μ)', 'Sigma (σ)'])
        csv_path = os.path.join(self.print_dir, f"track_diff_fit_{self.end_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved fit results to {csv_path}")

    # ------------------------------------------------------------
    # prediction vs target
    # ------------------------------------------------------------
    def plot_pred_target(self, preds, targets, scale=1.0, bins=100):
        targets = np.asarray(targets)
        preds = np.asarray(preds)

        # Momentum magnitude is scaled by the scale factor of magnetic field
        preds = preds.copy()
        preds[:, 2] /= scale

        if preds.ndim != 2 or targets.ndim != 2:
            raise ValueError("preds and targets must be 2D arrays")

        if preds.shape[1] != 5 or targets.shape[1] != 5:
            raise ValueError(f"Expected 5 state parameters, got preds={preds.shape[1]}, targets={targets.shape[1]}")

        # Five plots
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))

        axes = axes.flatten()

        # Loop over 5 parameters
        for dim, name in enumerate(self.state_parameter_name):
            target = targets[:, dim]
            pred = preds[:, dim]

            if dim == 1:
                delta = pred - target
                delta = np.arctan2(
                    np.sin(delta),
                    np.cos(delta)
                )
                pred = target + delta

            # Remove invalid values
            valid = (np.isfinite(target) & np.isfinite(pred))

            target = target[valid]
            pred = pred[valid]

            # Determine plotting range
            xmin = min(target.min(), pred.min())
            xmax = max(target.max(), pred.max())

            # Avoid zero-width range
            if xmax <= xmin:
                center = 0.5 * (xmin + xmax)
                xmin = center - 1.0
                xmax = center + 1.0

            if dim == 0:
                xmin = -0.15
                xmax = 0.15

            # 2D histogram
            h = axes[dim].hist2d(target, pred, range=[[xmin, xmax], [xmin, xmax]], bins=bins, norm=colors.LogNorm())

            # y = x
            axes[dim].plot([xmin, xmax], [xmin, xmax], "r--", linewidth=2)

            axes[dim].set_xlabel("Target")
            axes[dim].set_ylabel("Prediction")
            axes[dim].set_title(name)

            fig.colorbar(h[3], ax=axes[dim])

        # Hide unused sixth subplot
        for j in range(len(self.state_parameter_name), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        outname = os.path.join(self.print_dir, f"pred_vs_target_{self.end_name}.png")
        plt.savefig(outname, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved prediction-vs-target plot to {outname}")
