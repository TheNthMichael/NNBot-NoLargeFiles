
# Standard Library
import tkinter as tk
from typing import Callable, Any


class Overlay:
    """
    Creates an overlay window using tkinter
    Uses the "-topmost" property to always stay on top of other Windows
    """
    def __init__(self,
                 close_callback: Callable[[Any], None],
                 get_new_text_callback: Callable[[], str],
                 update_frequency_ms: int = 5_000):
        self.close_callback = close_callback
        self.get_new_text_callback = get_new_text_callback
        self.update_frequency_ms = update_frequency_ms
        self.root = tk.Tk()

        # Set up Close Label
        self.close_label = tk.Label(
            self.root,
            text=' X |',
            font=('Consolas', '14'),
            fg='green3',
            bg='grey19'
        )
        self.close_label.bind("<Button-1>", close_callback)
        self.close_label.grid(row=0, column=0)

        # Set up Ping Label
        self.ping_text = tk.StringVar()
        self.ping_label = tk.Label(
            self.root,
            textvariable=self.ping_text,
            font=('Consolas', '14'),
            fg='green3',
            bg='grey19'
        )
        self.ping_label.grid(row=0, column=1)

        # Define Window Geometery
        self.root.overrideredirect(True)
        self.root.geometry("+5+5")
        self.root.lift()
        self.root.wm_attributes("-topmost", True)

    def update_label(self) -> None:
        self.root.update()
        self.ping_text.set(self.get_new_text_callback())
        self.root.after(self.update_frequency_ms, self.update_label)

    def run(self) -> None:
        self.root.after(0, self.update_label)
        self.root.mainloop()