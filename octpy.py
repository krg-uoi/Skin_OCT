import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from skimage import io, feature, measure
import os
import csv
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


class Image:
    def __init__(self, path, canny_thresh=(0.15, 0.25)):
        self.path = os.path.abspath(path)
        self.canny_thresh = canny_thresh
        self.read_grayscale()
        self.shape = self.img.shape
        self.size = self.img.size
        self.fit = False

    def read_grayscale(self):
        self.img = io.imread(self.path, as_gray=True)

    def properties(self):
        props = {
            "Name": os.path.basename(self.path),
            "Path": self.path,
            "Dimensions": self.shape,
            "Pixels": self.size,
        }

        return props

    def print_properties(self):
        for key, val in self.properties().items():
            print(f"{key}: {val}")
        print()

    def plot_image(self):
        self.fig, self.ax = plt.subplots()

        aspect_ratio = self.shape[0] / self.shape[1]
        fig_width, fig_height = figaspect(aspect_ratio)
        self.fig.set_size_inches(fig_width, fig_height)

        self.ax.imshow(self.img, cmap="gray")
        self.ax.set_title(f"Image: {os.path.basename(self.path)}")
        self.ax.set_xlabel("x (pixels)")
        self.ax.set_ylabel("y (pixels)")

        return self.fig, self.ax

    def show(self):
        plt.show()

    def draw_point(self, coords, color="y"):
        try:
            self.fig
        except AttributeError:
            self.plot_image()

        if self.are_coords_int(coords):
            self.ax.plot(coords[0], coords[1], "o", color=color, label="point")

    def draw_line(self, p1, p2, linewidth=1, color="y", alpha=1):
        try:
            self.fig
        except AttributeError:
            self.plot_image()

        if all(self.are_coords_int(p) for p in (p1, p2)):
            self.ax.plot(
                (p1[0], p2[0]),
                (p1[1], p2[1]),
                label="line",
                linewidth=linewidth,
                color=color,
                alpha=alpha,
            )

    def get_canny_edge(self):
        edge = feature.canny(
            self.img,
            sigma=3,
            low_threshold=self.canny_thresh[0],
            high_threshold=self.canny_thresh[1],
        )
        edge_points = np.argwhere(edge > 0)
        # swap columns
        edge_points[:, [0, 1]] = edge_points[:, [1, 0]]
        # sort array
        edge_points_sorted = edge_points[edge_points[:, 0].argsort()]

        return edge_points_sorted

    def draw_canny(self, color="r"):
        try:
            self.fig
        except AttributeError:
            self.plot_image()

        edge_points = self.get_canny_edge()
        self.ax.plot(
            edge_points[:, 0],
            edge_points[:, 1],
            "o",
            color=color,
            markersize=1,
            label="Canny",
        )

    def are_coords_int(self, coords):
        for c in coords:
            if isinstance(c, int):
                return True
            elif c.is_integer():
                return True
            else:
                raise ValueError(
                    "The coordinates correspond to image pixels. They must be integers."
                )

    def get_line_angle(self, p1, p2):
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / math.pi
        return angle

    def get_line_profile(self, p1, p2, linewidth=1, reduce_func=np.mean):
        if all(self.are_coords_int(p) for p in (p1, p2)):
            y = measure.profile_line(
                self.img,
                (p1[1], p1[0]),
                (p2[1], p2[0]),
                linewidth=linewidth,
                reduce_func=reduce_func,
            )

        x = [i for i in range(len(y))]

        self.x_lprofile = x
        self.y_lprofile = y

        return self.x_lprofile, self.y_lprofile

    def plot_line_profile(self, x, y, color="y", label=None):
        fig, ax = plt.subplots()

        ax.plot(x, y, color=color, label=label)
        ax.set_title("Line profile")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("Normalized intensity (a.u.)")
        ax.grid(linestyle=":", alpha=0.75)

        return fig, ax

    def line_profile(
        self,
        p1,
        p2,
        linewidth=1,
        color="y",
        alpha=1,
        reduce_func=np.mean,
        fit_exp=False,
        bounds=(-np.inf, np.inf),
    ):
        try:
            self.fig
        except AttributeError:
            self.plot_image()

        self.draw_line(p1, p2, linewidth=linewidth, color=color, alpha=alpha)
        x, y = self.get_line_profile(
            p1, p2, linewidth=linewidth, reduce_func=reduce_func
        )
        fig, ax = self.plot_line_profile(x, y, color=color, label="Profile")

        if fit_exp:
            self.popt, self.pcov = self.fit_exp(x, y, bounds=bounds)
            a, b, c = self.popt
            self.y_fit = self.exp(x, *self.popt)

            label_text = rf"$f(x) = {a:.3f} \cdot e^{{- { b:.3f} x}} + {c:.3f}$"
            ax.plot(x, self.y_fit, label=label_text, color="r", linestyle="--")

        ax.legend()

        self.profile_fig = fig
        self.profile_ax = ax

        return self.profile_fig, self.profile_ax

    def exp(self, x, a, b, c):
        return a * np.exp(-b * np.array(x)) + c

    def fit_exp(self, x, y, bounds=(-np.inf, np.inf)):
        self.fit = True
        popt, pcov = curve_fit(self.exp, x, y, p0=[1, 0, 0], bounds=bounds, maxfev=5000)

        return popt, pcov

    def fit_params(self):
        params_dict = {}
        a, b, c = self.popt
        std_a, std_b, std_c = np.sqrt(np.diag(self.pcov))
        r, _ = pearsonr(self.y_lprofile, self.y_fit)
        r_squared = r**2

        for i, p in enumerate(["a", "b", "c"]):
            params_dict[p] = self.popt[i]

        for i, p in enumerate(["a", "b", "c"]):
            params_dict[f"std_{p}"] = np.sqrt(np.diag(self.pcov))[i]

        params_dict["R2"] = r_squared

        return params_dict

    def pp_fit_params(self):
        params_dict = self.fit_params()

        print("Fit function: a * exp(-b) + c")
        print()
        print("Fit parameters:")
        # print()
        # items = [f'{key} = {value}' for key, value in params_dict.items()]
        # end_str = '-' * 10
        # result = end_str.join(items[i:i+3] for i in range(0, len(items), 3))
        for i, (key, value) in enumerate(params_dict.items(), 1):
            print(f"{key} = {value}")
            if i % 3 == 0:
                print("-" * 20)
        print()
        # print(result)

    def fit_exp_residuals(self):
        residuals = self.y_lprofile - self.y_fit

        return residuals

    def plot_fit_exp_residuals(self, linewidth=None, color=None, alpha=1):
        residuals = self.fit_exp_residuals()
        fig, ax = plt.subplots()

        ax.plot(
            self.x_lprofile, residuals, linewidth=linewidth, color=color, alpha=alpha
        )
        ax.set_title("Fit residuals")
        ax.grid(linestyle=":", alpha=0.75)

        self.residuals_fig = fig
        self.residuals_ax = ax
        return self.residuals_fig, self.residuals_ax

    def line_profile_from_canny(
        self,
        x,
        length=100,
        offset=0,
        if_multiple=0,
        linewidth=1,
        color="y",
        alpha=1,
        reduce_func=np.mean,
        fit_exp=False,
        bounds=(-np.inf, np.inf),
    ):
        edge = self.get_canny_edge()
        edge_point = edge[edge[:, 0] == x]
        edge_point_sorted = edge_point[edge_point[:, 1].argsort()]

        if edge_point_sorted[:, 1].shape[0] > 1:
            y = edge_point_sorted[if_multiple, 1]
        else:
            y = edge_point_sorted[0, 1]

        p1 = (x, y + offset)
        p2 = (x, p1[1] + length)

        fig, ax = self.line_profile(
            p1,
            p2,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
            reduce_func=reduce_func,
            fit_exp=fit_exp,
            bounds=bounds,
        )

        return fig, ax

    def save_line_profile(self, path, overwrite=False):
        write_file = True
        if self.file_exists(path):
            if not overwrite:
                print("Overwrite? [Y/n]")
                write_file = self.yes_no()
            else:
                print("File has been overwritten.")

        if write_file:
            if self.fit:
                data_header = [
                    "Pixel",
                    "Normalized intensity (a.u.)",
                    "Fit",
                    "Fit residuals",
                ]
                data = zip(
                    self.x_lprofile,
                    self.y_lprofile,
                    self.y_fit,
                    self.fit_exp_residuals(),
                )
            else:
                data_header = ["Pixel", "Normalized intensity (a.u.)"]
                data = zip(self.x_lprofile, self.y_lprofile)

            with open(path, "w") as f:
                writer = csv.writer(f, delimiter=";", lineterminator="\r")

                writer.writerow(["[ Image properties ]"])
                for key, val in self.properties().items():
                    writer.writerow([f"{key}: {val}"])
                    # print(f"{key}: {val}")

                if self.fit:
                    a, b, c = self.popt
                    std_a, std_b, std_c = np.sqrt(np.diag(self.pcov))

                    writer.writerow([])
                    writer.writerow(["[ Fit parameters ]"])
                    writer.writerow([f"a = {a} +/- {std_a}"])
                    writer.writerow([f"a = {b} +/- {std_b}"])
                    writer.writerow([f"a = {c} +/- {std_c}"])
                    writer.writerow([f"R^2 = {self.r_squared}"])

                writer.writerow([])
                writer.writerow(["[ Data ]"])
                writer.writerow(data_header)
                for row in data:
                    writer.writerow(row)

    def file_exists(self, path):
        full_path = os.path.abspath(path)
        basename = os.path.basename(full_path)
        dirname = os.path.dirname(full_path)

        if os.path.isfile(full_path):
            print(f"File {basename} already exists in {dirname}.")
            return True
        else:
            return False

    def yes_no(self, default="y"):
        yes_list = ["y", "Y"]
        no_list = ["n", "N"]

        while True:
            response = input()
            if response == "":
                return True
            elif response in yes_list:
                return True
            elif response in no_list:
                return False
            else:
                print("Invalid input. Please enter [Y]es or [n]o.")

    def crop(self, p1, p2):
        self.img = self.img[p1[1] : p2[1], p1[0] : p2[0]]

    def save_image(self, path, dpi="figure"):
        self.fig.savefig(path, dpi=dpi)

    def save_line_profile_plot(self, path, dpi="figure"):
        self.profile_fig.savefig(path, dpi=dpi)

    def save_fit_exp_residual_plot(self, path, dpi="figure"):
        self.residuals_fig.savefig(path, dpi=dpi)


if __name__ == "__main__":
    import pandas as pd
    from cycler import cycler
    import itertools
    from matplotlib import cm
 
    output_dir = "/images_with_fit/healthy_only_multiple_lines/waterfall_test"
    image = Image(
        "/gg_images/healthy-2787_Denoised_FNL.jpg"
    )
    # image.plot_image()
    image.crop((500, 150), (601, 451))
    print(image.img)
    image.plot_image()

    df = pd.read_excel(
        os.path.join(output_dir, "500-600.xlsx"),
        header=0,
        index_col=0,
    )
    # print(df)
    # df.plot(legend=False)
    offset = 0
    fig, ax = plt.subplots(figsize=(20, 10))
    y2 = 0
    # print(df.index)
    print(df.to_numpy())
    plt.imshow(df.to_numpy(), cmap="gray")

    add_image = np.add(image.img, 100 * df.to_numpy())
    # print(add_image)
    plt.imshow(add_image, cmap="gray")
    plt.show()

