"""Desktop app for nanolm business-sales modeling (native Tkinter GUI).

    python desktop_app.py

Wraps the from-scratch ``nanolm`` transformer in a native desktop window:

  1. Load a public *business-sales* dataset (auto-download),
  2. Train the transformer from scratch on a modeling corpus built from it,
  3. Generate synthetic sales records (transformer inference / sampling),
  4. Show analytics + a transparent daily-revenue forecast + a chart.

Training / generation run in a background thread and stream back to the UI
through a thread-safe queue, so the window stays responsive.
"""

from __future__ import annotations

import contextlib
import os
import queue
import sys
import threading
import types

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tkinter as tk
from tkinter import messagebox, ttk

import nanolm.sales_data as sd
from nanolm.train import train as run_training
from nanolm.inference import ModelRunner, load_checkpoint

CKPT_DIR = os.path.join(HERE, "checkpoints", "sales")
CORPUS = os.path.join(HERE, "datasets", "sales_corpus.txt")
BEST = os.path.join(CKPT_DIR, "best.pt")

DEFAULTS = dict(steps=1200, batch=64, block=64, n_layer=4, n_head=4,
                n_embd=128, lr=6e-4, warmup=80)


class _QueueWriter:
    """A file-like object that forwards stdout to a thread-safe queue."""

    def __init__(self, q):
        self.q = q
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.q.put(("log", line))
        return len(s)

    def flush(self):
        if self._buf:
            self.q.put(("log", self._buf))
            self._buf = ""


class SalesDesktopApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("nanolm • Business Sales Modeling")
        self.geometry("1000x720")
        self.minsize(860, 600)
        self.q = queue.Queue()
        self.records = None
        self.runner = None
        self._build_ui()
# ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        ttk.Label(
            self,
            text="nanolm — a from-scratch transformer modeling real sales data",
            font=("Segoe UI", 13, "bold"),
            padding=8,
        ).pack(fill="x")

        bar = ttk.Frame(self, padding=(8, 0))
        bar.pack(fill="x")
        ttk.Button(bar, text="1. Load data", command=self.on_load).pack(side="left")
        ttk.Button(bar, text="2. Train model", command=self.on_train).pack(side="left", padx=6)
        ttk.Button(bar, text="Analytics", command=self.on_analytics).pack(side="left")
        ttk.Button(bar, text="Generate sales", command=self.on_generate).pack(side="left", padx=6)

        kpis = ttk.Frame(self, padding=(8, 6))
        kpis.pack(fill="x")
        self.kpi = {}
        for label in ("records", "revenue", "avg ticket", "forecast/day"):
            frame = ttk.LabelFrame(kpis, text=label)
            frame.pack(side="left", expand=True, fill="x", padx=4)
            value = ttk.Label(frame, text="—", font=("Consolas", 13, "bold"))
            value.pack()
            self.kpi[label] = value

        self.canvas = tk.Canvas(self, bg="#0f1220", height=190, highlightthickness=0)
        self.canvas.pack(fill="x", padx=8, pady=6)

        gen = ttk.Frame(self, padding=(8, 0))
        gen.pack(fill="x")
        ttk.Label(gen, text="seed:").pack(side="left")
        self.seed = ttk.Entry(gen, width=18)
        self.seed.insert(0, "13|")
        self.seed.pack(side="left", padx=4)
        ttk.Label(gen, text="temp:").pack(side="left")
        self.temp = ttk.Entry(gen, width=6)
        self.temp.insert(0, "0.9")
        self.temp.pack(side="left", padx=4)
        ttk.Label(gen, text="train steps:").pack(side="left", padx=(16, 0))
        self.steps_entry = ttk.Entry(gen, width=8)
        self.steps_entry.insert(0, str(DEFAULTS["steps"]))
        self.steps_entry.pack(side="left", padx=4)
        ttk.Label(gen, text="format: amount|day|hour|pay|dist", foreground="#7a8a99").pack(
            side="left", padx=10
        )

        self.log = tk.Text(
            self, wrap="word", bg="#0b0e14", fg="#d7e0ea", font=("Consolas", 11)
        )
        self.log.pack(fill="both", expand=True, padx=8, pady=8)
        self.status = ttk.Label(self, text="ready", anchor="w", relief="sunken")
        self.status.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------ #
    # Queue -> UI
    # ------------------------------------------------------------------ #
    def set_status(self, text):
        self.q.put(("status", text))

    def emit(self, text):
        self.q.put(("log", text))

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                elif kind == "status":
                    self.status.config(text=payload)
                elif kind == "chart":
                    self._draw_chart(payload)
                elif kind == "kpi":
                    label, text = payload
                    self.kpi[label].config(text=text)
        except queue.Empty:
            pass
        self.after(120, self._drain)

    # ------------------------------------------------------------------ #
    # Chart (plain canvas bars — no matplotlib dependency)
    # ------------------------------------------------------------------ #
    def _draw_chart(self, values):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width() or 960
        h = c.winfo_height() or 190
        n = len(values)
        if not n:
            return
        peak = max(values) or 1
        slot = w / n
        bw = max(1.0, slot * 0.7)
        for i, v in enumerate(values):
            bh = (v / peak) * (h - 24)
            x = 6 + i * slot
            c.create_rectangle(x, h - bh, x + bw, h - 8, fill="#4cc9f0", outline="")
        c.create_text(6, 6, anchor="nw", fill="#8aa0b5",
                      text=f"daily revenue  (peak ${peak:,.0f})")
# ------------------------------------------------------------------ #
    # Threading helpers
    # ------------------------------------------------------------------ #
    def _start(self, target, label):
        self.set_status(label)
        threading.Thread(target=self._worker, args=(target,), daemon=True).start()

    def _worker(self, target):
        try:
            target()
        except Exception as exc:  # surface errors into the log pane
            self.emit(f"[error] {type(exc).__name__}: {exc}")
        finally:
            self.set_status("ready")

    def _int_entry(self, entry, default):
        try:
            return int(entry.get())
        except (ValueError, AttributeError):
            return default

    def _float_entry(self, entry, default):
        try:
            return float(entry.get())
        except (ValueError, AttributeError):
            return default

    def _push_kpis(self, a):
        self.q.put(("kpi", ("records", f"{a['n']:,}")))
        self.q.put(("kpi", ("revenue", f"${a['total_revenue']:,.0f}")))
        self.q.put(("kpi", ("avg ticket", f"${a['avg_ticket']:,.2f}")))
        self.q.put(("kpi", ("forecast/day", f"${sd.next_period_forecast(self.records):,.0f}")))
        self.q.put(("chart", [v for _, v in a["daily"]]))

    def _ensure_records(self):
        if self.records is None:
            path = sd.ensure_sales_csv()
            self.records = sd.load_sales(path)
        return self.records

    # ------------------------------------------------------------------ #
    # Button handlers
    # ------------------------------------------------------------------ #
    def on_load(self):
        def work():
            path = sd.ensure_sales_csv()
            self.records = sd.load_sales(path)
            a = sd.analytics(self.records)
            self.emit(f"Loaded {a['n']} sales records from {path}")
            self._push_kpis(a)
            self.emit("Revenue by payment: "
                      + ", ".join(f"{k}=${v:,.0f}" for k, v in a["by_payment"]))
        self._start(work, "downloading + loading data…")

    def on_analytics(self):
        def work():
            self._ensure_records()
            a = sd.analytics(self.records)
            f = sd.next_period_forecast(self.records)
            self._push_kpis(a)
            self.emit(f"Total revenue ${a['total_revenue']:,.2f} | "
                      f"avg ticket ${a['avg_ticket']:.2f} | next-day estimate ${f:,.0f}")
            self.emit("Revenue by day (first 8): "
                      + ", ".join(f"{d}=${v:,.0f}" for d, v in a["daily"][:8]))
        self._start(work, "computing analytics…")

    def _train_args(self, steps):
        ns = types.SimpleNamespace()
        ns.text_file = None
        ns.data_size = 0
        ns.min_terms, ns.max_terms, ns.max_val = 2, 3, 50
        ns.n_layer = DEFAULTS["n_layer"]
        ns.n_head = DEFAULTS["n_head"]
        ns.n_embd = DEFAULTS["n_embd"]
        ns.block_size = DEFAULTS["block"]
        ns.dropout = 0.0
        ns.rope_theta = 10000.0
        ns.no_tie = False
        ns.steps = steps
        ns.batch_size = DEFAULTS["batch"]
        ns.lr = DEFAULTS["lr"]
        ns.min_lr = 1e-4
        ns.warmup = DEFAULTS["warmup"]
        ns.weight_decay = 0.1
        ns.beta1, ns.beta2 = 0.9, 0.95
        ns.grad_clip = 1.0
        ns.log_interval = 50
        ns.eval_interval = 150
        ns.eval_iters = 8
        ns.out = CKPT_DIR
        ns.seed = 1337
        ns.device = "cpu"
        ns.val_frac = 0.05
        return ns

    def on_train(self):
        def work():
            self._ensure_records()
            os.makedirs(CKPT_DIR, exist_ok=True)
            corpus = sd.records_to_corpus(self.records)
            with open(CORPUS, "w", encoding="utf-8") as fh:
                fh.write(corpus)
            self.emit(f"Modeling corpus: {len(corpus):,} chars from "
                      f"{len(self.records)} sales")
            steps = self._int_entry(self.steps_entry, DEFAULTS["steps"])
            args = self._train_args(steps)
            args.text_file = CORPUS
            self.emit("Training from-scratch transformer (see loss stream below)…")
            with contextlib.redirect_stdout(_QueueWriter(self.q)):
                run_training(args)
            model, tok, _ = load_checkpoint(BEST, device="cpu")
            self.runner = ModelRunner(model, tok, device="cpu")
            self.emit("Training complete — model ready. Click 'Generate sales'.")
        self._start(work, "training model…")

    def on_generate(self):
        if self.runner is None:
            try:
                self.runner = ModelRunner.from_checkpoint(BEST, device="cpu")
            except Exception:
                messagebox.showinfo(
                    "Train first", "No trained model found — click '2. Train model'."
                )
                return

        def work():
            seed = self.seed.get().strip() or "13|"
            temp = self._float_entry(self.temp, 0.9)
            self.emit(f"Synthetic sales (seed={seed!r}, temp={temp})  "
                      f"format amount|day|hour|pay|dist:")
            for _ in range(6):
                line = self.runner.complete(
                    seed, max_new_tokens=24, temperature=temp, top_k=40, top_p=0.95,
                    stop_at="\n", repetition_penalty=1.05,
                )
                line = line.split("\n", 1)[0] if "\n" in line else line
                self.emit("   " + (line or "(empty)"))
        self._start(work, "generating synthetic sales…")


def main():
    SalesDesktopApp().mainloop()


if __name__ == "__main__":
    main()