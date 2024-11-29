from IPython.core.pylabtools import figsize
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from typing import Callable, Tuple


def add_button(fig: Figure, event: Callable, label: str = "Update", size: Tuple[float, float] = (0.2, None)):
    """
    Add a button to the bottom of the figure. Run the `event` when the button is clicked.
    The caller must maintain a reference to the returned widget object for the button to be alive.

    :param fig: Figure to which the button is added
    :param event: The function to call when the button is clicked
    :param label: Label text displayed on the button
    :param size: Width and height of the button as fractions of the figure size
    :return: The axes of the newly created button and the button widget.
    """

    # Adjust the layout to make space for the button
    button_width, button_height = size

    axes = fig.get_axes()
    n_axes = len(axes)
    ax_bottom = axes[-1].get_position().ymin
    button_height = 0.1 * 1/n_axes if button_height is None else button_height

    fig.subplots_adjust(bottom = ax_bottom + button_height)

    # Calculate position for the button relative to the figure
    button_pos = (
        0.5 - button_width / 2,  # Center horizontally
        0.01,  # Above the bottom edge of the figure
        button_width,  # Width of the button
        button_height  # Height of the button
    )

    button_ax = fig.add_axes(button_pos)
    button = Button(button_ax, label)
    button.on_clicked(event)

    return button_ax, button



if __name__ == "__main__":
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.ion()


    def on_button_click(word):
        print(f"Button clicked! {word}")

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(3,1, figsize=(5,8))  # 2x2 grid of subplots
    for ax in axs.flat:
        ax.plot([0, 1], [0, 1])  # Example plot
        ax.set_xlabel("X-axis Label")
        ax.set_ylabel("Y-axis Label")
    fig.tight_layout()

    # Add a button at the bottom of the figure
    func = lambda _: on_button_click("haha")
    button_ax, button = add_button(fig, label="Click Me", event=func) # todo: smarter button size

    plt.show()