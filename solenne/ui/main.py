from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext
from uuid import uuid4
import requests


API_URL = "http://127.0.0.1:6969"


def run() -> None:
    root = tk.Tk()
    root.title("Solenne")
    root.geometry("600x400")
    sid = str(uuid4())

    chat = scrolledtext.ScrolledText(root, width=80, height=20)
    chat.pack(fill=tk.BOTH, expand=True)
    entry = tk.Entry(root)
    entry.pack(fill=tk.X)

    def poll() -> None:
        try:
            r = requests.get(f"{API_URL}/state", params={"sid": sid}, timeout=5)
            if r.status_code >= 400:
                chat.insert(tk.END, "connection error\n")
            else:
                text = r.json().get("text", "")
                if text and text != "…":
                    chat.insert(tk.END, f"Solenne: {text}\n")
                    chat.see(tk.END)
        except Exception:
            pass
        root.after(1000, poll)

    def send(_ev: object | None = None) -> None:
        m = entry.get().strip()
        if not m:
            return
        chat.insert(tk.END, f"You: {m}\n")
        entry.delete(0, tk.END)
        try:
            resp = requests.post(
                f"{API_URL}/message", params={"sid": sid}, json={"text": m}, timeout=5
            )
            if resp.status_code >= 400:
                chat.insert(tk.END, "connection error\n")
        except Exception:
            pass

    entry.bind("<Return>", send)
    poll()
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual launch
    run()
