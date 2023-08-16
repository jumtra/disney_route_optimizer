from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from disney_route_optimize.common.config_maneger import ConfigManeger
from disney_route_optimize.wait_predction.dataclass.do_predict import Predictor

dict_weekday = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def visualize(config_maneger: ConfigManeger, predictor: Predictor):
    path_visualize = Path(config_maneger.config.output.wp_output.path_visualize_dir)
    path_visualize.mkdir(exist_ok=True, parents=True)

    path_allfig = path_visualize / "actual_pred_all_valid.png"
    path_each_day_fig = path_visualize / "actual_pred_each_day_valid.png"
    do_visualize = config_maneger.config.tasks.wp_task.do_visualize

    df_train = predictor.df_train
    df_valid = predictor.df_valid

    path_allfig = path_visualize / "actual_pred_all_valid.png"
    path_each_day_fig = path_visualize / "actual_pred_each_day_valid.png"
    get_plot(
        df=df_valid,
        path_allfig=path_allfig,
        path_each_day_fig=path_each_day_fig,
        do_visualize=do_visualize,
        fig_size=20,
    )

    path_allfig = path_visualize / "actual_pred_all_train.png"
    path_each_day_fig = path_visualize / "actual_pred_each_day_train.png"
    get_plot(
        df=df_train,
        path_allfig=path_allfig,
        path_each_day_fig=path_each_day_fig,
        do_visualize=do_visualize,
        fig_size=30,
    )


def get_plot(
    df: pd.DataFrame,
    path_allfig: Path,
    path_each_day_fig: Path,
    do_visualize: bool,
    fig_size: int = 15,
) -> None:
    if do_visualize or (not path_allfig.exists()) or (not path_each_day_fig.exists()):
        # 実績と分布で比較

        ser_predict = df["pred"]
        ser_actual = df["target"]
        mae = round(mean_absolute_error(ser_actual, ser_predict), 2)
        rmse = round(mean_squared_error(ser_actual, ser_predict) ** 0.5, 2)
        fig, ax = plt.subplots()
        plot_predict_actual(ser_predict=ser_predict, ser_actual=ser_actual, ax=ax)
        plt.title(f"mae:{mae},rmse:{rmse}")
        fig.savefig(path_allfig)

        plt.close()
        # 日付毎
        valid_day = len(df["date"].unique())

        max_col = 7  # 一行一週間
        num_plot = int(valid_day / max_col) + valid_day % max_col + 1
        count = 0
        fig, ax = plt.subplots(
            nrows=num_plot,
            ncols=max_col,
            figsize=(fig_size, fig_size),
        )
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        for date, df in df.groupby("date"):
            ser_predict = df["pred"]
            ser_actual = df["target"]
            mae = round(mean_absolute_error(ser_actual, ser_predict), 2)
            rmse = round(mean_squared_error(ser_actual, ser_predict) ** 0.5, 2)
            plot_predict_actual(
                ser_predict=ser_predict,
                ser_actual=ser_actual,
                ax=ax[int(count / max_col)][count % max_col],
            )
            ax[int(count / max_col)][count % max_col].set(
                title=f"{str(date)}({dict_weekday[pd.to_datetime(date).weekday()]}\nmae:{mae},rmse:{rmse})"
            )
            count += 1
        fig.savefig(path_each_day_fig)
        plt.close()


def plot_predict_actual(ser_predict: pd.Series, ser_actual: pd.Series, ax: plt.Axes) -> plt.Axes:
    """予実プロットの作成"""
    label_predict = ser_predict.name
    label_actual = ser_actual.name

    max_actual = ser_actual.max()
    max_predict = ser_predict.max()
    min_actual = ser_actual.min()
    min_predict = ser_predict.min()

    minmax = min(max_actual, max_predict)
    maxmin = max(min_actual, min_predict)

    line = np.linspace(0, 100, 100)
    sub_line = np.linspace(maxmin, minmax, 100)

    df = pd.concat([ser_predict, ser_actual], axis=1)

    ax.scatter(
        data=df,
        x=label_actual,
        y=label_predict,
        s=20,
        alpha=0.7,
    )

    ax.set(xlabel=label_actual, ylabel=label_predict, aspect="equal")
    ax.plot(
        sub_line,
        sub_line,
        color="black",
        linestyle="dashed",
        linewidth=0.3,
    )
    ax.plot(line, line, color="black", linestyle="solid", linewidth=1)

    return ax
