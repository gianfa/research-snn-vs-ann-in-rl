import matplotlib.pyplot as plt
from snntorch import spikeplot as splt  # noqa


# Data Generation #



# Visualization #

def plot_cur_mem_spk(
    cur,
    mem,
    spk,
    thr_line=False,
    vline=False, title=False,
    ylim_input=(0, 0.4),
    ylim_mempot=(0, 1.25),
    y_lim_raster=(0, 6.75),
    x_lim=(0, 200)
):
    """
    credits: Jason Eshraghian
    """
    # Generate Plots
    fig, ax = plt.subplots(
        3, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.4]}
    )

    # Plot input current
    ax[0].plot(cur, c="tab:orange")
    ax[0].set_ylim([ylim_input[0], ylim_input[1]])
    ax[0].set_xlim(x_lim)
    ax[0].set_ylabel("Input Current ($I_{in}$)")
    if title:
        ax[0].set_title(title)

    # Plot membrane potential
    ax[1].plot(mem)
    ax[1].set_ylim([ylim_mempot[0], ylim_mempot[1]])
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    if thr_line:
        ax[1].axhline(
            y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2
        )
    plt.xlabel("Time step")

    # Plot output spike using spikeplot
    splt.raster(spk, ax[2], s=400, c="black", marker="|")
    if vline:
        ax[2].axvline(
            x=vline,
            ymin=y_lim_raster[0],
            ymax=y_lim_raster[1],
            alpha=0.15,
            linestyle="dashed",
            c="black",
            linewidth=2,
            zorder=0,
            clip_on=False,
        )
    ax[2].set_ylabel("Output spikes")
    ax[2].set_yticks([])
    fig.tight_layout()

    plt.show()