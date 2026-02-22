import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def numericalData(df):
    # data load time impacted by dataset id, but vary little variance otherwise
    print(df.groupby(["Dataset ID"]).agg({"Data Load Time (s)": ["min", "max", "mean", "std"]}))

    # runtime impacted substantially by batch size and architecture, but minimally by dataset
    print(df.groupby(["Dataset ID", "Batch Size", "Architecture ID"]).agg({"Runtime (s)": "mean"}))

    # optimizer has an effect on runtime, but learning rate does not
    print(df.groupby(["Optimizer ID", "Learning Rate"]).agg({"Runtime (s)": "mean"}))

    # batch size decreases accuracy as it increases, but has no effect or even increases accuracy if using Adam
    print(df.groupby(["Batch Size", "Optimizer ID"]).agg({"Accuracy (Training)": "mean", "Accuracy (Testing)": "mean"}))

    # best learning rate consistently varied based on dataset and optimizer
    print(df.groupby(["Dataset ID", "Optimizer ID", "Learning Rate"]).agg({"Accuracy (Training)": "mean", "Accuracy (Testing)": "mean"}))

    # brief informal view of the best accuracy and runtime models
    print(df.sort_values("Accuracy (Testing)", ascending=False).groupby("Dataset ID").head(10))
    print(df.sort_values("Runtime (s)", ascending=True).groupby("Dataset ID").head(10))
    print(df.sort_values("Accuracy (Testing)", ascending=False).groupby("Dataset ID").head(100).melt().value_counts().head(10))

def hypothesisTesting(df):
    df_clean = df.rename(columns={
        "Runtime (s)": "Runtime",
        "Accuracy (Testing)": "Accuracy_Testing",
        "Batch Size": "Batch_Size",
        "Dataset ID": "Dataset_ID",
        "Learning Rate": "Learning_Rate",
        "Optimizer ID": "Optimizer_ID",
        "Architecture ID": "Architecture_ID"
    })

    # full testing accuracy anova
    formula = 'Accuracy_Testing ~ C(Batch_Size) + C(Dataset_ID) + C(Learning_Rate) + C(Optimizer_ID) + C(Architecture_ID)'
    model = ols(formula, data=df_clean).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table['sum_sq_perc'] = (anova_table['sum_sq'] / anova_table['sum_sq'].sum()) * 100
    print(anova_table)

    # testing accuracy anova with fashion mnist and adam
    formula = 'Accuracy_Testing ~ C(Batch_Size) + C(Learning_Rate) + C(Architecture_ID)'
    model = ols(formula, data=df_clean[(df_clean["Optimizer_ID"] == 1) & (df_clean["Dataset_ID"] == 1)]).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table['sum_sq_perc'] = (anova_table['sum_sq'] / anova_table['sum_sq'].sum()) * 100
    print(anova_table)

    # runtime anova
    formula = 'Runtime ~ C(Batch_Size) + C(Dataset_ID) + C(Learning_Rate) + C(Optimizer_ID) + C(Architecture_ID)'
    model = ols(formula, data=df_clean).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table['sum_sq_perc'] = (anova_table['sum_sq'] / anova_table['sum_sq'].sum()) * 100
    print(anova_table)

    # gap anova
    formula = 'Gap ~ C(Q("Batch Size")) + C(Q("Dataset ID")) + C(Q("Learning Rate")) + ' \
              'C(Q("Optimizer ID")) + C(Q("Architecture ID"))'
    global_model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(global_model, typ=2)
    anova_table['eta_sq'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + anova_table.loc['Residual', 'sum_sq'])
    print(anova_table)

def trendAnalysis(df):
    # correlations and trends
    print(df[["Architecture ID", "Gap", "Runtime (s)", "Accuracy (Training)", "Accuracy (Testing)"]].corr())

    # tukey test for gap difference by architecture
    tukey = pairwise_tukeyhsd(endog=df["Accuracy (Training)"] - df["Accuracy (Testing)"],
                              groups=df["Architecture ID"],
                              alpha=0.01)
    print(tukey)

def visualizations(df):
    df_grouped = df.groupby([
        'Architecture ID', 'Epoch', 'Batch Size', 'Learning Rate', 'Optimizer ID'
    ])['Gap'].mean().reset_index()

    # gap trend visualization
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_grouped, x='Epoch', y='Gap', hue='Architecture ID', palette="Spectral")

    plt.title('Overfitting Gap over Time by Architecture Size')
    plt.ylabel('Generalization Gap (Train Acc - Test Acc)')
    plt.axhline(0, color='red', linestyle='--')
    plt.legend(title='Architecture ID')
    plt.show()

    # gap heatmaps
    pivot_map = df.pivot_table(index="Architecture ID",
                               columns="Optimizer ID",
                               values="Gap",
                               aggfunc="mean")
    plt.figure(figsize=(8, 10))
    sns.heatmap(pivot_map, annot=True, cmap="YlOrRd")
    plt.title("Overfitting Heatmap: Architecture vs Optimizer")

    pivot_map = df.pivot_table(index="Architecture ID",
                               columns="Optimizer ID",
                               values="Accuracy (Training)",
                               aggfunc="mean")
    plt.figure(figsize=(8, 10))
    sns.heatmap(pivot_map, annot=True, cmap="Greens")
    plt.title("Training Accuracy Heatmap: Architecture vs Optimizer")

    pivot_map = df.pivot_table(index="Architecture ID",
                               columns="Optimizer ID",
                               values="Accuracy (Testing)",
                               aggfunc="mean")
    plt.figure(figsize=(8, 10))
    sns.heatmap(pivot_map, annot=True, cmap="Greens")
    plt.title("Testing Accuracy Heatmap: Architecture vs Optimizer")
    plt.show()

    # graphs over time by architecture ID
    df.pivot_table(index="Epoch", columns="Architecture ID", values="Accuracy (Training)", aggfunc="mean").plot(
        colormap="gist_rainbow", title="Training Accuracy over Time by Architecture ID", ylabel="Training Accuracy")

    df.pivot_table(index="Epoch", columns="Architecture ID", values="Accuracy (Testing)", aggfunc="mean").plot(
        colormap="gist_rainbow", title="Testing Accuracy over Time by Architecture ID", ylabel="Testing Accuracy")

    df.pivot_table(index="Epoch", columns="Architecture ID", values="Loss (Training)", aggfunc="mean").plot(
        colormap="gist_rainbow", title="Training Loss over Time by Architecture ID", ylabel="Training Loss")

    df.pivot_table(index="Epoch", columns="Architecture ID", values="Loss (Testing)", aggfunc="mean").plot(
        colormap="gist_rainbow", title="Testing Loss over Time by Architecture ID", ylabel="Testing Loss")
    plt.show()

    # scatter plot
    accuracies = df[(df["Epoch"] % 5 == 0)]["Accuracy (Testing)"]
    runtimes = df.groupby(df.index // 5)["Runtime (s)"].sum()
    colors = df.groupby(df.index // 5)["Architecture ID"].agg("mean")

    scatter = plt.scatter(runtimes, accuracies, c=colors, cmap="rainbow")
    plt.colorbar(scatter).set_ticks(ticks=range(0, 20))
    plt.title("Testing Accuracy vs Runtime of Every 5 Epochs by Architecture ID")
    plt.xlabel("Runtime of 5 Epochs (s)")
    plt.ylabel("Average Testing Accuracy of 5 Epochs")
    plt.show()

    # pie chart of runtime
    total_runtime = df.sum()["Runtime (s)"] / 60 / 60
    total_loadtime = df.sum()["Data Load Time (s)"] / 50 / 60 / 60
    print(f"Total Runtime (h): {total_runtime}")
    print(f"Total Loadtime (h): {total_loadtime}")
    print(f"Total Time (h): {total_runtime + total_loadtime}")
    (df.groupby(["Batch Size"]).sum()["Runtime (s)"] / 60 / 60 / total_runtime).plot(kind="pie", colormap="Spectral", autopct="%1.1f%%")
    plt.title("Percentage of Total Runtime by Batch Size")
    plt.text(s=f"Total Runtime (h): {round(total_runtime, 2)}", x=0, y=0, horizontalalignment="center", verticalalignment="bottom")
    plt.legend(title="Batch Size", loc="upper left")
    plt.show()
    (df.groupby(["Architecture ID"]).sum()["Runtime (s)"] / 60 / 60 / total_runtime).plot(kind="pie", colormap="Spectral", autopct="%1.1f%%")
    plt.title("Percentage of Total Runtime by Architecture ID")
    plt.text(s=f"Total Runtime (h): {round(total_runtime, 2)}", x=0, y=0, horizontalalignment="center", verticalalignment="bottom")
    plt.legend(title="Architecture ID", loc="upper left")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("trials.csv", dtype=float)
    df['Gap'] = df['Accuracy (Training)'] - df['Accuracy (Testing)']

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    plt.style.use("fivethirtyeight")

    numericalData(df)
    hypothesisTesting(df)
    trendAnalysis(df)
    visualizations(df)