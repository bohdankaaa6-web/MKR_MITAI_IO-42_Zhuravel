import re
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==========================================================
# 1. БЕЗПЕЧНЕ ОБЧИСЛЕННЯ ФУНКЦІЇ КОРИСТУВАЧА
# ==========================================================
SAFE_MATH = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tg": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "log": np.log,
    "ln": np.log,
    "log10": np.log10,
    "pi": np.pi,
    "e": np.e,
}


def detect_num_vars(expr_str):
    cleaned = expr_str.replace("^", "**").strip()
    return 2 if re.search(r"\by\b", cleaned) else 1


def eval_user_function(expr_str, x_arr, y_arr=None):
    cleaned = expr_str.replace("^", "**").strip()
    env = dict(SAFE_MATH)
    env["x"] = x_arr
    if y_arr is not None:
        env["y"] = y_arr
    res = eval(cleaned, {"__builtins__": {}}, env)
    if np.ndim(res) == 0:
        res = np.full_like(x_arr, float(res), dtype=float)
    res = np.asarray(res, dtype=float)
    if not np.all(np.isfinite(res)):
        raise ValueError("Функція дає NaN або нескінченність на заданому відрізку.")
    return res


# ==========================================================
# 2. ДЕКОДУВАННЯ ПРЯМОГО КОДУВАННЯ ТА НАВЧАННЯ DAG-МЕРЕЖІ
# ==========================================================
def decode_and_prune(chrom, num_inputs, max_neurons):
    """
    Пряме кодування (Лекція 13, слайди 23-26):
    Вузли 0..L-1 — входи; L..B-2 — приховані нейрони; B-1 — вихідний нейрон.
    """
    B = num_inputs + max_neurons
    output_node = B - 1

    C = np.zeros((B, B), dtype=int)
    triu_r, triu_c = np.triu_indices(B, k=1)
    C[triu_r, triu_c] = chrom

    # Пряма досяжність від входів
    forward_reach = set(range(num_inputs))
    changed = True
    while changed:
        changed = False
        for u in list(forward_reach):
            for v in range(u + 1, B):
                if C[u, v] == 1 and v not in forward_reach:
                    forward_reach.add(v)
                    changed = True

    if output_node not in forward_reach:
        return None

    # Зворотна досяжність від виходу (видалення тупикових нейронів)
    backward_reach = {output_node}
    changed = True
    while changed:
        changed = False
        for v in list(backward_reach):
            for u in range(v):
                if C[u, v] == 1 and u not in backward_reach:
                    backward_reach.add(u)
                    changed = True

    active_nodes = sorted(forward_reach & backward_reach)
    active_set = set(active_nodes)

    active_inputs = [v for v in active_nodes if v < num_inputs]
    active_neurons = [v for v in active_nodes if v >= num_inputs]
    if not active_inputs or output_node not in active_set:
        return None

    parents = {}
    active_edges = []
    C_pruned = np.zeros((B, B), dtype=int)
    for v in active_neurons:
        p_list = [u for u in active_nodes if u < v and C[u, v] == 1]
        parents[v] = p_list
        for u in p_list:
            active_edges.append((u, v))
            C_pruned[u, v] = 1

    depth = {u: 0 for u in active_inputs}
    for v in active_neurons:
        depth[v] = 1 + max(depth[u] for u in parents[v])

    max_d = depth[output_node]
    layer_counts = [
        sum(1 for node, dep in depth.items() if dep == d)
        for d in range(max_d + 1)
    ]

    return {
        "B": B,
        "output_node": output_node,
        "active_nodes": active_nodes,
        "active_inputs": active_inputs,
        "active_neurons": active_neurons,
        "active_edges": active_edges,
        "parents": parents,
        "depth": depth,
        "layer_str": "-".join(map(str, layer_counts)),
        "C_pruned": C_pruned,
        "topology_key": tuple(active_edges),
    }


def forward_dag(X_norm, arch, weights, biases, y_mean, y_std):
    N = X_norm.shape[0]
    act = {u: X_norm[:, u] for u in arch["active_inputs"]}

    out_node = arch["output_node"]
    for v in arch["active_neurons"]:
        z = np.full(N, biases[v], dtype=float)
        for idx, u in enumerate(arch["parents"][v]):
            z += act[u] * weights[v][idx]
        act[v] = z if v == out_node else np.tanh(z)

    return act[out_node] * y_std + y_mean


def train_dag_adam(arch, X_norm, y_true, epochs=220, lr=0.035, seed=0):
    rng = np.random.default_rng(seed)
    N = X_norm.shape[0]
    y_mean = float(np.mean(y_true))
    y_std = float(np.std(y_true))
    if y_std < 1e-9:
        y_std = 1.0
    y_scaled = (y_true - y_mean) / y_std

    active_inputs = arch["active_inputs"]
    active_neurons = arch["active_neurons"]
    parents = arch["parents"]
    out_node = arch["output_node"]

    W, b = {}, {}
    mW, vW, mb, vb = {}, {}, {}, {}

    for v in active_neurons:
        k = len(parents[v])
        lim = 1.5 / np.sqrt(max(1, k))
        W[v] = rng.uniform(-lim, lim, size=k)
        b[v] = 0.0
        mW[v] = np.zeros(k)
        vW[v] = np.zeros(k)
        mb[v] = 0.0
        vb[v] = 0.0

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_loss = float("inf")
    best_W = {v: W[v].copy() for v in active_neurons}
    best_b = {v: b[v] for v in active_neurons}

    for step in range(1, epochs + 1):
        act = {u: X_norm[:, u] for u in active_inputs}
        for v in active_neurons:
            z = np.full(N, b[v], dtype=float)
            for idx, u in enumerate(parents[v]):
                z += act[u] * W[v][idx]
            act[v] = z if v == out_node else np.tanh(z)

        diff = act[out_node] - y_scaled
        loss = float(np.mean(diff ** 2))
        if loss < best_loss:
            best_loss = loss
            best_W = {v: W[v].copy() for v in active_neurons}
            best_b = {v: b[v] for v in active_neurons}

        if loss < 1e-7:
            break

        delta = {v: np.zeros(N, dtype=float) for v in active_neurons}
        delta[out_node] = (2.0 / N) * diff

        b1_corr = 1.0 - beta1 ** step
        b2_corr = 1.0 - beta2 ** step

        for v in reversed(active_neurons):
            d_pre = delta[v] if v == out_node else delta[v] * (1.0 - act[v] ** 2)
            gb = float(np.sum(d_pre))
            gW = np.empty_like(W[v])
            for idx, u in enumerate(parents[v]):
                gW[idx] = float(np.dot(d_pre, act[u]))
                if u in delta:
                    delta[u] += d_pre * W[v][idx]

            mW[v] = beta1 * mW[v] + (1.0 - beta1) * gW
            vW[v] = beta2 * vW[v] + (1.0 - beta2) * (gW ** 2)
            W[v] -= lr * (mW[v] / b1_corr) / (np.sqrt(vW[v] / b2_corr) + eps)

            mb[v] = beta1 * mb[v] + (1.0 - beta1) * gb
            vb[v] = beta2 * vb[v] + (1.0 - beta2) * (gb ** 2)
            b[v] -= lr * (mb[v] / b1_corr) / (np.sqrt(vb[v] / b2_corr) + eps)

    y_pred = forward_dag(X_norm, arch, best_W, best_b, y_mean, y_std)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {
        "weights": best_W,
        "biases": best_b,
        "y_mean": y_mean,
        "y_std": y_std,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


# ==========================================================
# 3. ОКРЕМЕ ВІКНО: ІНФОРМАЦІЯ ПО РОБОТІ ГА
# ==========================================================
class GAMonitorWindow:
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Інформація по роботі ГА (параметри та проміжні мережі)")
        self.win.geometry("940x660")
        self.win.protocol("WM_DELETE_WINDOW", self.hide)

        # --- Блок 1: Параметри ГА під час пошуку (фіксовані колонки без зміщень) ---
        param_frame = ttk.LabelFrame(self.win, text="Параметри генетичного алгоритму під час пошуку", padding=8)
        param_frame.pack(fill="x", padx=10, pady=6)

        self.param_vars = {
            "status": tk.StringVar(value="Очікування запуску..."),
            "gen": tk.StringVar(value="0 / 0"),
            "pop_size": tk.StringVar(value="-"),
            "chrom_len": tk.StringVar(value="-"),
            "crossover": tk.StringVar(value="-"),
            "mutation": tk.StringVar(value="-"),
            "elite": tk.StringVar(value="-"),
            "best_err": tk.StringVar(value="-"),
            "best_struct": tk.StringVar(value="-"),
            "unique_cnt": tk.StringVar(value="0"),
        }

        # Рядок 0: Статус на всю ширину
        ttk.Label(param_frame, text="Статус:", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=(6, 4), pady=2
        )
        ttk.Label(param_frame, textvariable=self.param_vars["status"], anchor="w").grid(
            row=0, column=1, columnspan=5, sticky="ew", padx=(0, 6), pady=2
        )

        # Рядки 1-3: Короткі числові параметри з фіксованою шириною
        compact_items = [
            ("Покоління:", "gen", 1, 0),
            ("Розмір популяції:", "pop_size", 1, 2),
            ("Довжина хромосоми K:", "chrom_len", 1, 4),
            ("Ймовірність кросоверу:", "crossover", 2, 0),
            ("Ймовірність мутації:", "mutation", 2, 2),
            ("Елітизм:", "elite", 2, 4),
            ("Унікальних топологій:", "unique_cnt", 3, 0),
            ("Найкраща похибка:", "best_err", 3, 2),
        ]
        for text, key, r, c in compact_items:
            ttk.Label(param_frame, text=text, font=("Segoe UI", 9, "bold")).grid(
                row=r, column=c, sticky="w", padx=(6, 4), pady=2
            )
            ttk.Label(param_frame, textvariable=self.param_vars[key], width=20, anchor="w").grid(
                row=r, column=c + 1, sticky="w", padx=(0, 10), pady=2
            )

        # Рядок 4: Найкраща структура на всю ширину
        ttk.Label(param_frame, text="Найкраща структура:", font=("Segoe UI", 9, "bold")).grid(
            row=4, column=0, sticky="w", padx=(6, 4), pady=2
        )
        ttk.Label(param_frame, textvariable=self.param_vars["best_struct"], anchor="w").grid(
            row=4, column=1, columnspan=5, sticky="ew", padx=(0, 6), pady=2
        )

        for col in (1, 3, 5):
            param_frame.grid_columnconfigure(col, weight=1, minsize=130)

        # --- Блок 2: Таблиця проміжних варіантів мереж ---
        table_frame = ttk.LabelFrame(self.win, text="Проміжні варіанти мережі та їх похибка", padding=6)
        table_frame.pack(fill="both", expand=True, padx=10, pady=4)

        cols = ("gen", "struct", "neurons", "edges", "rmse", "mse", "mae", "fitness")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=9)
        headers = {
            "gen": ("Покоління", 75),
            "struct": ("Структура (рівні)", 155),
            "neurons": ("Нейронів", 75),
            "edges": ("Зв'язків", 75),
            "rmse": ("RMSE", 100),
            "mse": ("MSE", 100),
            "mae": ("MAE", 100),
            "fitness": ("Фітнес-втрата", 110),
        }
        for col, (title, width) in headers.items():
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="center")

        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # --- Блок 3: Живий графік збіжності ГА ---
        plot_frame = ttk.LabelFrame(self.win, text="Динаміка пошуку ГА по поколіннях", padding=4)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        self.fig = Figure(figsize=(8.5, 2.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_title("Зміна похибки найкращої та середньої мережі в популяції")
        self.ax.set_xlabel("Покоління")
        self.ax.set_ylabel("Похибка")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def show(self):
        self.win.deiconify()
        self.win.lift()

    def hide(self):
        self.win.withdraw()

    def reset(self, params_info):
        self.param_vars["status"].set("Виконується пошук...")
        self.param_vars["gen"].set(f"0 / {params_info['generations']}")
        self.param_vars["pop_size"].set(str(params_info["pop_size"]))
        self.param_vars["chrom_len"].set(str(params_info["chrom_len"]))
        self.param_vars["crossover"].set(f"{params_info['cx_rate']:.2f}")
        self.param_vars["mutation"].set(f"{params_info['mut_rate']:.3f}")
        self.param_vars["elite"].set(str(params_info["elite"]))
        self.param_vars["best_err"].set("-")
        self.param_vars["best_struct"].set("-")
        self.param_vars["unique_cnt"].set("0")

        for row in self.tree.get_children():
            self.tree.delete(row)
        self._init_plot()

    def update_progress(self, data):
        self.param_vars["gen"].set(f"{data['gen']} / {data['max_gen']}")
        self.param_vars["best_err"].set(f"{data['metric_name']} = {data['best_metric']:.6f}")
        self.param_vars["best_struct"].set(
            f"{data['layer_str']} (нейронів: {data['neurons']}, зв'язків: {data['edges']})"
        )
        self.param_vars["unique_cnt"].set(str(data["unique_cnt"]))

        for item in data["intermediate_rows"]:
            iid = self.tree.insert(
                "",
                "end",
                values=(
                    item["gen"],
                    item["layer_str"],
                    item["neurons"],
                    item["edges"],
                    f"{item['rmse']:.6f}",
                    f"{item['mse']:.6f}",
                    f"{item['mae']:.6f}",
                    f"{item['fitness']:.6f}",
                ),
            )
            self.tree.see(iid)

        self.ax.clear()
        gens = list(range(1, len(data["best_hist"]) + 1))
        self.ax.plot(
            gens, data["best_hist"], marker="o", markersize=4,
            label="Найкраща особина", color="tab:blue", linewidth=1.8
        )
        if data["avg_hist"]:
            self.ax.plot(
                gens, data["avg_hist"], marker="s", markersize=3,
                label="Середнє (валідні)", color="tab:orange", linestyle="--", linewidth=1.3
            )
        self.ax.set_title(f"Динаміка пошуку ГА ({data['metric_name']})")
        self.ax.set_xlabel("Покоління")
        self.ax.set_ylabel(data["metric_name"])
        self.ax.set_yscale("log")
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.set_xlim(0.5, max(2.5, len(gens) + 0.5))
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def set_finished(self, status_text):
        self.param_vars["status"].set(status_text)


# ==========================================================
# 4. ГОЛОВНЕ ВІКНО ПРОГРАМИ
# ==========================================================
class StructuralSynthesisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Автоматичний структурний синтез ШНМ (Генетичний алгоритм)")
        self.root.geometry("1220x740")

        self.msg_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None

        self.ga_window = GAMonitorWindow(self.root)
        self.ga_window.hide()

        self._build_ui()
        self.root.after(80, self._poll_queue)

    def _build_ui(self):
        left = ttk.Frame(self.root, padding=10, width=360)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        task_box = ttk.LabelFrame(left, text="Вхідні параметри задачі", padding=8)
        task_box.pack(fill="x", pady=(0, 8))

        ttk.Label(task_box, text="Функція f(x) або f(x, y):").pack(anchor="w")
        self.entry_func = ttk.Entry(task_box)
        self.entry_func.insert(0, "7 * sin(5/2 * cos(x))")
        self.entry_func.pack(fill="x", pady=(2, 6))

        preset_frame = ttk.Frame(task_box)
        preset_frame.pack(fill="x", pady=(0, 6))
        ttk.Label(preset_frame, text="Приклад:").pack(side="left")
        self.preset_cb = ttk.Combobox(
            preset_frame,
            state="readonly",
            values=[
                "7 * sin(5/2 * cos(x))",
                "(x - y) * sin(x + y)",
                "x * cos(y) + sin(x)",
                "sin(2*x) + 0.5*cos(3*x)",
            ],
        )
        self.preset_cb.current(0)
        self.preset_cb.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.preset_cb.bind("<<ComboboxSelected>>", self._on_preset)

        var_frame = ttk.Frame(task_box)
        var_frame.pack(fill="x", pady=3)
        ttk.Label(var_frame, text="Кількість входів:").pack(side="left")
        self.var_mode = ttk.Combobox(
            var_frame,
            state="readonly",
            values=["Авто (1 або 2)", "1 змінна (x)", "2 змінні (x, y)"],
            width=18,
        )
        self.var_mode.current(0)
        self.var_mode.pack(side="right")

        range_frame = ttk.Frame(task_box)
        range_frame.pack(fill="x", pady=4)
        ttk.Label(range_frame, text="Відрізок [a; b]:").pack(side="left")
        self.entry_b = ttk.Entry(range_frame, width=8)
        self.entry_b.insert(0, "1.0")
        self.entry_b.pack(side="right")
        ttk.Label(range_frame, text="до").pack(side="right", padx=4)
        self.entry_a = ttk.Entry(range_frame, width=8)
        self.entry_a.insert(0, "0.0")
        self.entry_a.pack(side="right")

        neur_frame = ttk.Frame(task_box)
        neur_frame.pack(fill="x", pady=4)
        ttk.Label(neur_frame, text="Макс. число нейронів (A):").pack(side="left")
        self.entry_max_neurons = ttk.Entry(neur_frame, width=10)
        self.entry_max_neurons.insert(0, "8")
        self.entry_max_neurons.pack(side="right")

        err_frame = ttk.Frame(task_box)
        err_frame.pack(fill="x", pady=4)
        ttk.Label(err_frame, text="Макс. похибка:").pack(side="left")
        self.metric_cb = ttk.Combobox(err_frame, state="readonly", values=["RMSE", "MSE", "MAE"], width=6)
        self.metric_cb.current(0)
        self.metric_cb.pack(side="right", padx=(4, 0))
        self.entry_max_err = ttk.Entry(err_frame, width=10)
        self.entry_max_err.insert(0, "0.02")
        self.entry_max_err.pack(side="right")

        ga_box = ttk.LabelFrame(left, text="Налаштування генетичного алгоритму", padding=8)
        ga_box.pack(fill="x", pady=(0, 8))

        self.entry_pop = self._add_param_row(ga_box, "Розмір популяції:", "24")
        self.entry_gen = self._add_param_row(ga_box, "Макс. поколінь:", "35")
        self.entry_cx = self._add_param_row(ga_box, "Ймовірність кросоверу:", "0.85")
        self.entry_mut = self._add_param_row(ga_box, "Ймовірність мутації біта:", "0.08")
        self.entry_elite = self._add_param_row(ga_box, "Елітних хромосом:", "2")

        btn_box = ttk.Frame(left)
        btn_box.pack(fill="x", pady=4)

        self.btn_start = ttk.Button(btn_box, text="Запустити структурний синтез", command=self.start_synthesis)
        self.btn_start.pack(fill="x", pady=3)

        self.btn_stop = ttk.Button(btn_box, text="Зупинити пошук", command=self.stop_synthesis, state="disabled")
        self.btn_stop.pack(fill="x", pady=3)

        self.btn_ga_win = ttk.Button(btn_box, text="Відкрити вікно інформації ГА", command=self.ga_window.show)
        self.btn_ga_win.pack(fill="x", pady=3)

        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 2))
        self.lbl_status = ttk.Label(left, text="Готово до роботи", foreground="gray25")
        self.lbl_status.pack(anchor="w")

        right = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        right.pack(side="right", fill="both", expand=True)

        out_box = ttk.LabelFrame(right, text="Вивід програми: отримана структура мережі та її похибка", padding=8)
        out_box.pack(fill="x", pady=(0, 8))

        self.txt_result = tk.Text(out_box, height=8, font=("Consolas", 10), wrap="word")
        self.txt_result.pack(fill="x")
        self.txt_result.insert("1.0", "Натисніть «Запустити структурний синтез» для початку пошуку оптимальної архітектури.")
        self.txt_result.config(state="disabled")

        fig_box = ttk.LabelFrame(right, text="Графічне представлення синтезованої мережі та результату моделювання", padding=4)
        fig_box.pack(fill="both", expand=True)

        self.fig_main = Figure(figsize=(8.2, 4.8), dpi=100)
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=fig_box)
        self.canvas_main.get_tk_widget().pack(fill="both", expand=True)

    def _add_param_row(self, parent, label_text, default_val):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label_text).pack(side="left")
        entry = ttk.Entry(row, width=10)
        entry.insert(0, default_val)
        entry.pack(side="right")
        return entry

    def _on_preset(self, _event=None):
        self.entry_func.delete(0, tk.END)
        self.entry_func.insert(0, self.preset_cb.get())

    def _set_result_text(self, text):
        self.txt_result.config(state="normal")
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert("1.0", text)
        self.txt_result.config(state="disabled")

    def start_synthesis(self):
        try:
            expr = self.entry_func.get().strip()
            if not expr:
                raise ValueError("Введіть математичний вираз функції.")

            a = float(self.entry_a.get().replace(",", "."))
            b = float(self.entry_b.get().replace(",", "."))
            if b <= a:
                raise ValueError("Права межа відрізка має бути більшою за ліву (b > a).")

            max_neurons = int(self.entry_max_neurons.get())
            if max_neurons < 1 or max_neurons > 20:
                raise ValueError("Максимальне число нейронів має бути в межах від 1 до 20.")

            max_err = float(self.entry_max_err.get().replace(",", "."))
            if max_err <= 0:
                raise ValueError("Максимальна похибка має бути > 0.")

            pop_size = int(self.entry_pop.get())
            generations = int(self.entry_gen.get())
            cx_rate = float(self.entry_cx.get().replace(",", "."))
            mut_rate = float(self.entry_mut.get().replace(",", "."))
            elite = int(self.entry_elite.get())
            if pop_size < 4 or generations < 1 or elite < 0 or elite >= pop_size:
                raise ValueError("Перевірте коректність параметрів ГА.")

            mode = self.var_mode.get()
            if mode.startswith("1"):
                num_inputs = 1
            elif mode.startswith("2"):
                num_inputs = 2
            else:
                num_inputs = detect_num_vars(expr)

            if num_inputs == 1:
                x_tr = np.linspace(a, b, 35)
                X_raw_tr = x_tr.reshape(-1, 1)
                y_tr = eval_user_function(expr, x_tr)

                x_te = np.linspace(a, b, 120)
                X_raw_te = x_te.reshape(-1, 1)
                y_te = eval_user_function(expr, x_te)
                grid_info = (x_te,)
            else:
                g_tr = np.linspace(a, b, 11)
                X1_tr, X2_tr = np.meshgrid(g_tr, g_tr)
                X_raw_tr = np.column_stack([X1_tr.ravel(), X2_tr.ravel()])
                y_tr = eval_user_function(expr, X1_tr.ravel(), X2_tr.ravel())

                g_te = np.linspace(a, b, 25)
                X1_te, X2_te = np.meshgrid(g_te, g_te)
                X_raw_te = np.column_stack([X1_te.ravel(), X2_te.ravel()])
                y_te = eval_user_function(expr, X1_te.ravel(), X2_te.ravel())
                grid_info = (X1_te, X2_te)

            X_tr_norm = 2.0 * (X_raw_tr - a) / (b - a) - 1.0
            X_te_norm = 2.0 * (X_raw_te - a) / (b - a) - 1.0

        except Exception as exc:
            messagebox.showerror("Помилка вхідних даних", str(exc))
            return

        B = num_inputs + max_neurons
        chrom_len = B * (B - 1) // 2
        metric_name = self.metric_cb.get()

        params = {
            "expr": expr,
            "a": a,
            "b": b,
            "num_inputs": num_inputs,
            "max_neurons": max_neurons,
            "max_err": max_err,
            "metric_name": metric_name,
            "pop_size": pop_size,
            "generations": generations,
            "cx_rate": cx_rate,
            "mut_rate": mut_rate,
            "elite": elite,
            "chrom_len": chrom_len,
            "X_tr_norm": X_tr_norm,
            "y_tr": y_tr,
            "X_te_norm": X_te_norm,
            "y_te": y_te,
            "grid_info": grid_info,
        }

        self.stop_event.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.progress.config(maximum=generations, value=0)
        self.lbl_status.config(text="Виконується структурний синтез...")
        self._set_result_text("Триває еволюційний пошук структури нейронної мережі...")

        self.ga_window.reset(params)
        self.ga_window.show()

        self.worker_thread = threading.Thread(target=self._run_structural_ga, args=(params,), daemon=True)
        self.worker_thread.start()

    def stop_synthesis(self):
        self.stop_event.set()
        self.lbl_status.config(text="Зупинка після поточного покоління...")

    def _make_valid_chromosome(self, rng, num_inputs, max_neurons, chrom_len):
        B = num_inputs + max_neurons
        triu_r, triu_c = np.triu_indices(B, k=1)
        density = rng.uniform(0.25, 0.65)
        for _ in range(10):
            chrom = (rng.random(chrom_len) < density).astype(int)
            if decode_and_prune(chrom, num_inputs, max_neurons) is not None:
                return chrom
        chrom = (rng.random(chrom_len) < 0.3).astype(int)
        in_node = int(rng.integers(0, num_inputs))
        mid = num_inputs
        pair_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(triu_r, triu_c))}
        if max_neurons >= 2:
            chrom[pair_to_idx[(in_node, mid)]] = 1
            chrom[pair_to_idx[(mid, B - 1)]] = 1
        else:
            chrom[pair_to_idx[(in_node, B - 1)]] = 1
        return chrom

    def _run_structural_ga(self, p):
        t0 = time.time()
        rng = np.random.default_rng(42)

        num_inputs = p["num_inputs"]
        max_neurons = p["max_neurons"]
        chrom_len = p["chrom_len"]
        pop_size = p["pop_size"]
        generations = p["generations"]
        metric_key = p["metric_name"].lower()

        population = [
            self._make_valid_chromosome(rng, num_inputs, max_neurons, chrom_len)
            for _ in range(pop_size)
        ]

        cache = {}
        best_overall = None
        best_hist = []
        avg_hist = []

        for gen in range(generations):
            if self.stop_event.is_set():
                break

            eval_results = []
            intermediate_rows = []

            for chrom in population:
                arch = decode_and_prune(chrom, num_inputs, max_neurons)
                if arch is None:
                    eval_results.append({"valid": False, "fitness": 1e9, "metric": 1e9, "chrom": chrom})
                    continue

                tkey = arch["topology_key"]
                if tkey not in cache:
                    train_res = train_dag_adam(arch, p["X_tr_norm"], p["y_tr"], epochs=220, lr=0.035, seed=len(cache))
                    complexity = 1e-4 * p["max_err"] * (
                        len(arch["active_neurons"]) / max(1, max_neurons)
                        + len(arch["active_edges"]) / max(1, chrom_len)
                    )
                    fit_loss = train_res[metric_key] + complexity
                    cache[tkey] = {
                        "arch": arch,
                        "train": train_res,
                        "fitness": fit_loss,
                        "metric": train_res[metric_key],
                    }

                rec = cache[tkey]
                item = {
                    "valid": True,
                    "fitness": rec["fitness"],
                    "metric": rec["metric"],
                    "arch": rec["arch"],
                    "train": rec["train"],
                    "chrom": chrom,
                }
                eval_results.append(item)

                # Фіксуємо кожне покращення в таблиці проміжних варіантів
                if best_overall is None or item["fitness"] < best_overall["fitness"]:
                    best_overall = item
                    intermediate_rows.append({
                        "gen": gen + 1,
                        "layer_str": item["arch"]["layer_str"],
                        "neurons": len(item["arch"]["active_neurons"]),
                        "edges": len(item["arch"]["active_edges"]),
                        "rmse": item["train"]["rmse"],
                        "mse": item["train"]["mse"],
                        "mae": item["train"]["mae"],
                        "fitness": item["fitness"],
                    })

            valid_items = [r for r in eval_results if r["valid"]]
            order = np.argsort([r["fitness"] for r in eval_results])
            gen_best = eval_results[order[0]]

            if not intermediate_rows and gen_best["valid"]:
                intermediate_rows.append({
                    "gen": gen + 1,
                    "layer_str": gen_best["arch"]["layer_str"],
                    "neurons": len(gen_best["arch"]["active_neurons"]),
                    "edges": len(gen_best["arch"]["active_edges"]),
                    "rmse": gen_best["train"]["rmse"],
                    "mse": gen_best["train"]["mse"],
                    "mae": gen_best["train"]["mae"],
                    "fitness": gen_best["fitness"],
                })

            cur_best_m = best_overall["metric"] if best_overall and best_overall["valid"] else 1e6
            avg_m = float(np.mean([r["metric"] for r in valid_items])) if valid_items else 1e6
            best_hist.append(max(cur_best_m, 1e-9))
            avg_hist.append(max(avg_m, 1e-9))

            if best_overall and best_overall["valid"]:
                self.msg_queue.put({
                    "type": "progress",
                    "gen": gen + 1,
                    "max_gen": generations,
                    "metric_name": p["metric_name"],
                    "best_metric": best_overall["metric"],
                    "layer_str": best_overall["arch"]["layer_str"],
                    "neurons": len(best_overall["arch"]["active_neurons"]),
                    "edges": len(best_overall["arch"]["active_edges"]),
                    "unique_cnt": len(cache),
                    "intermediate_rows": intermediate_rows,
                    "best_hist": list(best_hist),
                    "avg_hist": list(avg_hist),
                })

            if best_overall and best_overall["valid"] and best_overall["metric"] <= p["max_err"]:
                break

            new_pop = [eval_results[i]["chrom"].copy() for i in order[:p["elite"]]]
            fits = np.array([r["fitness"] for r in eval_results])

            while len(new_pop) < pop_size:
                idx1 = rng.integers(0, pop_size, size=3)
                p1 = population[idx1[np.argmin(fits[idx1])]].copy()
                idx2 = rng.integers(0, pop_size, size=3)
                p2 = population[idx2[np.argmin(fits[idx2])]].copy()

                if rng.random() < p["cx_rate"] and chrom_len >= 3:
                    pt1, pt2 = sorted(rng.choice(chrom_len, size=2, replace=False))
                    c1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])
                    c2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]])
                else:
                    c1, c2 = p1, p2

                for child in (c1, c2):
                    mask = rng.random(chrom_len) < p["mut_rate"]
                    child[mask] = 1 - child[mask]
                    if len(new_pop) < pop_size:
                        new_pop.append(child)

            population = new_pop

        elapsed = time.time() - t0
        if best_overall is None or not best_overall["valid"]:
            self.msg_queue.put({"type": "error", "message": "Не вдалося знайти валідну топологію мережі."})
            return

        arch = best_overall["arch"]
        tr = best_overall["train"]
        y_pred_te = forward_dag(p["X_te_norm"], arch, tr["weights"], tr["biases"], tr["y_mean"], tr["y_std"])
        y_te = p["y_te"]
        mse_te = float(np.mean((y_te - y_pred_te) ** 2))
        rmse_te = float(np.sqrt(mse_te))
        mae_te = float(np.mean(np.abs(y_te - y_pred_te)))
        ss_res = float(np.sum((y_te - y_pred_te) ** 2))
        ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
        r2_te = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        self.msg_queue.put({
            "type": "done",
            "params": p,
            "best": best_overall,
            "y_pred_te": y_pred_te,
            "mse_te": mse_te,
            "rmse_te": rmse_te,
            "mae_te": mae_te,
            "r2_te": r2_te,
            "elapsed": elapsed,
            "stopped": self.stop_event.is_set(),
            "gens_done": len(best_hist),
        })

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg["type"] == "progress":
                    self.progress["value"] = msg["gen"]
                    self.lbl_status.config(
                        text=f"Покоління {msg['gen']}/{msg['max_gen']} | {msg['metric_name']} = {msg['best_metric']:.5f}"
                    )
                    self.ga_window.update_progress(msg)
                elif msg["type"] == "error":
                    self.btn_start.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self.ga_window.set_finished("Помилка пошуку")
                    messagebox.showerror("Помилка", msg["message"])
                elif msg["type"] == "done":
                    self._handle_done(msg)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    def _handle_done(self, msg):
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

        p = msg["params"]
        arch = msg["best"]["arch"]
        tr = msg["best"]["train"]
        chrom = msg["best"]["chrom"]

        reached = msg["best"]["metric"] <= p["max_err"]
        if msg["stopped"]:
            status = f"Зупинено користувачем (покоління {msg['gens_done']})"
        elif reached:
            status = f"Успіх: досягнуто похибки <= {p['max_err']} на поколінні {msg['gens_done']}"
        else:
            status = f"Завершено всі {msg['gens_done']} поколінь"

        self.lbl_status.config(text=status)
        self.ga_window.set_finished(status)

        edges_str = ", ".join(f"{u}->{v}" for u, v in arch["active_edges"])
        chrom_str = "".join(map(str, chrom.tolist()))
        report_txt = (
            f"Статус: {status} | Час пошуку: {msg['elapsed']:.2f} с\n"
            f"Структура отриманої мережі (по рівнях): {arch['layer_str']} "
            f"(активних нейронів: {len(arch['active_neurons'])}, зв'язків: {len(arch['active_edges'])})\n"
            f"Похибка (тест): RMSE = {msg['rmse_te']:.6f} | MSE = {msg['mse_te']:.6f} | "
            f"MAE = {msg['mae_te']:.6f} | R^2 = {msg['r2_te']:.6f}\n"
            f"Похибка (навч): RMSE = {tr['rmse']:.6f} | MSE = {tr['mse']:.6f}\n"
            f"Активні зв'язки: {edges_str}\n"
            f"Хромосома ({len(chrom)} біт): {chrom_str}"
        )
        self._set_result_text(report_txt)
        self._draw_main_plots(p, arch, msg["y_pred_te"])

    def _draw_main_plots(self, p, arch, y_pred_te):
        self.fig_main.clear()

        ax_g = self.fig_main.add_subplot(121)
        depth = arch["depth"]
        max_d = max(depth.values())

        levels = {}
        for node, d in depth.items():
            levels.setdefault(d, []).append(node)

        pos = {}
        for d, nodes in levels.items():
            nodes_sorted = sorted(nodes)
            n_lvl = len(nodes_sorted)
            for idx, node in enumerate(nodes_sorted):
                pos[node] = ((idx + 1) / (n_lvl + 1), d / max(1, max_d))

        for u, v in arch["active_edges"]:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax_g.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.3, shrinkA=13, shrinkB=13),
                zorder=1,
            )

        L = p["num_inputs"]
        out_node = arch["output_node"]
        for node, (px, py) in pos.items():
            if node < L:
                color, label = "#4CAF50", f"x{node + 1}"
            elif node == out_node:
                color, label = "#FF7043", "y"
            else:
                color, label = "#42A5F5", f"N{node}"

            ax_g.scatter([px], [py], s=480, c=color, edgecolors="black", zorder=3)
            ax_g.text(px, py, label, ha="center", va="center", color="white", fontweight="bold", fontsize=9, zorder=4)

        ax_g.set_title(f"Синтезована структура: {arch['layer_str']}")
        ax_g.set_xlim(-0.05, 1.05)
        ax_g.set_ylim(-0.12, 1.12)
        ax_g.axis("off")

        if L == 1:
            ax_f = self.fig_main.add_subplot(122)
            x_te = p["grid_info"][0]
            ax_f.plot(x_te, p["y_te"], label="Еталон", color="tab:blue", linewidth=2.2)
            ax_f.plot(x_te, y_pred_te, label="Вихід ШНМ", color="tab:red", linestyle="--", linewidth=2.0)
            ax_f.set_title("Порівняння функції та апроксимації")
            ax_f.set_xlabel("x")
            ax_f.set_ylabel("y")
            ax_f.grid(True, alpha=0.3)
            ax_f.legend()
        else:
            ax_f = self.fig_main.add_subplot(122, projection="3d")
            X1_te, X2_te = p["grid_info"]
            Z_true = p["y_te"].reshape(X1_te.shape)
            Z_pred = y_pred_te.reshape(X1_te.shape)
            ax_f.plot_surface(X1_te, X2_te, Z_true, color="tab:blue", alpha=0.45, edgecolor="none")
            ax_f.plot_wireframe(X1_te, X2_te, Z_pred, color="tab:red", rstride=2, cstride=2, linewidth=0.8)
            ax_f.set_title("Еталон (суцільна) та ШНМ (сітка)")
            ax_f.set_xlabel("x")
            ax_f.set_ylabel("y")
            ax_f.set_zlabel("z")

        self.fig_main.tight_layout()
        self.canvas_main.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = StructuralSynthesisApp(root)
    root.mainloop()