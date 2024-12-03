import numpy as np
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from typing import Callable, Tuple
#

def add_button(fig: Figure, event: Callable, label: str = "Update", size: Tuple[float, float] = (None, None)):
    """
    Add a button to the bottom of a figure. Run the `event` when the button is clicked.
    The caller must maintain a reference to the returned widget object for the button to be active.

    :param fig: Figure to which the button is added
    :param event: The function to call when the button is clicked
    :param label: Label text displayed on the button
    :param size: Width and height of the button as fractions of the figure size
    :return: The axes of the newly created button and the button widget.
    """

    axes = fig.get_axes()

    # Initialize button metadata if not already present
    button_w, button_h = size
    if not hasattr(fig, "_button_positions"):
        fig.tight_layout()
        if button_h is None:
            button_h = 0.1 * 1 / len(axes)
        fig._button_positions = {"row_widths": [[]], "row_heights": [button_h], "n_plots": len(axes)}

    if button_w is None:
        button_w = len(label) * 0.02

    # use the height of the last button if button height is not provided
    button_h = fig._button_positions["row_heights"][-1] if button_h is None else button_h

    # find space for the new button and rearrange the old ones
    max_w = 0.97
    current_row_w = np.sum(fig._button_positions["row_widths"][-1])

    if current_row_w + button_w < max_w:
        fig._button_positions["row_widths"][-1].append(button_w)
        fig._button_positions["row_heights"][-1] = button_h
    else:
        fig._button_positions["row_widths"].append([button_w])
        fig._button_positions["row_heights"].append(button_h)

    # update the position of the previous buttons in the row
    last_row_widths = fig._button_positions["row_widths"][-1]
    x0 = (max_w - np.sum(last_row_widths))/2
    for i, w in enumerate(last_row_widths[:-1]):
        button_ax = axes[-len(last_row_widths) + i + 1]
        button_pos = (x0, button_ax.get_position().ymin, w, button_h)
        button_ax.set_position(button_pos)
        x0 += w

    # Adjust the layout to make space for the buttons when a new row is added
    if len(last_row_widths) == 1:
        ax_bottom = axes[fig._button_positions["n_plots"]-1].get_position().ymin
        fig.subplots_adjust(bottom = ax_bottom + button_h)

    # position for the new button
    y0 = 0.01 + np.sum(fig._button_positions["row_heights"]) - button_h
    button_pos = (x0, y0, button_w, button_h)

    # add the new button
    button_ax = fig.add_axes(button_pos)
    button = Button(button_ax, label)
    button.on_clicked(event)

    return button_ax, button


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.ion()


    def on_button_click():
        print(f"Button clicked!")

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(3,1, figsize=(5,8))  # 2x2 grid of subplots
    for ax in axs.flat:
        ax.plot([0, 1], [0, 1])  # Example plot
        ax.set_xlabel("X-axis Label")
        ax.set_ylabel("Y-axis Label")
    fig.tight_layout()

    # Add a button at the bottom of the figure
    button_ax, button = add_button(fig, label="Click Me", event=on_button_click) # todo: smarter button size
    button_ax1, button1 = add_button(fig, label="Click Me 2", event=on_button_click) # todo: smarter button size
    # button_ax2, button2 = add_button(fig, label="Click Me", event=func)

    plt.show()