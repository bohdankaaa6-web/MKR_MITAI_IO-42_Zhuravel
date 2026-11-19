import time
import numpy as np
import matplotlib.pyplot as plt

# Фіксуємо seed для відтворюваності результатів (за бажанням можна прибрати)
np.random.seed(42)

# ==========================================================
# 1. ПАРАМЕТРИ ВАРІАНТУ № 63
# ==========================================================
LAYERS = [2, 4, 8, 10, 8, 8, 1]  # Архітектура: 2-4-8-10-8-8-1
INTERVAL = (0.0, 1.0)            # Можна, наприклад, змінити на (0.0, 5.0)
GRID_TRAIN = 16                  # 16x16 = 256 точок для навчання
GRID_TEST = 25                   # 25x25 = 625 точок для перевірки

# Гіперпараметри генетичного алгоритму
POP_SIZE = 100                   # Розмір популяції
GENERATIONS = 800                # Для інтервалу (0, 5) краще ставити 1500-2000
ELITE_COUNT = 5                  # Кількість елітних хромосом
TOURNAMENT_K = 3                 # Розмір турніру при селекції
CROSSOVER_RATE = 0.85            # Ймовірність схрещування
MUTATION_RATE = 0.05             # Ймовірність мутації кожного гена
TARGET_MSE = 1e-4                # Критерій зупинки за помилкою


def target_function(x1, x2):
    """Цільова функція варіанту 63: y = 7 * sin(5/2 * cos(x))."""
    return 7.0 * np.sin((5.0 / 2.0) * np.cos(x1))


# ==========================================================
# 2. НЕЙРОННА МЕРЕЖА ПРЯМОГО ПОШИРЕННЯ
# ==========================================================
class FeedforwardNN:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.chromosome_length = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )

    def decode(self, chromosome):
        """Декодування хромосоми в матриці ваг W та вектори зсувів b."""
        weights, biases = [], []
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]

            w_count = in_dim * out_dim
            W = chromosome[idx : idx + w_count].reshape((in_dim, out_dim))
            idx += w_count

            b = chromosome[idx : idx + out_dim]
            idx += out_dim

            weights.append(W)
            biases.append(b)
        return weights, biases

    def forward(self, X, chromosome):
        """Пряме поширення сигналу (tanh у прихованих шарах, лінійний вихід)."""
        weights, biases = self.decode(chromosome)
        a = X
        for i in range(len(weights) - 1):
            a = np.tanh(a @ weights[i] + biases[i])
        out = a @ weights[-1] + biases[-1]
        return out.ravel()


# ==========================================================
# 3. ПІДГОТОВКА НАВЧАЛЬНОЇ ТА ТЕСТОВОЇ ВИБІРОК
# ==========================================================
def make_grid(interval, grid_size):
    x1 = np.linspace(interval[0], interval[1], grid_size)
    x2 = np.linspace(interval[0], interval[1], grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    X_raw = np.column_stack([X1.ravel(), X2.ravel()])
    # Масштабування входів у [-1, 1] для стабільної роботи tanh
    X_norm = 2.0 * (X_raw - interval[0]) / (interval[1] - interval[0]) - 1.0
    Z = target_function(X1.ravel(), X2.ravel())
    return X1, X2, X_norm, Z


X1_tr, X2_tr, X_train, Z_train = make_grid(INTERVAL, GRID_TRAIN)
X1_te, X2_te, X_test, Z_test = make_grid(INTERVAL, GRID_TEST)

nn = FeedforwardNN(LAYERS)
print(f"Архітектура мережі: {'-'.join(map(str, LAYERS))}")
print(f"Розмір хромосоми (кількість ваг і зсувів): K = {nn.chromosome_length}")
print(f"Інтервал моделювання: {INTERVAL}")


# ==========================================================
# 4. ЕВОЛЮЦІЙНІ ОПЕРАТОРИ, за лекцією
# ==========================================================
def init_population(pop_size, layer_sizes):
    """Ініціалізація популяції з урахуванням масштабу шарів."""
    pop = []
    for _ in range(pop_size):
        parts = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            # Останній шар має ширший діапазон, бо функція сягає [-7, 7]
            limit = 3.0 if i == len(layer_sizes) - 2 else 1.5 / np.sqrt(in_dim)
            w = np.random.uniform(-limit, limit, size=in_dim * out_dim)
            b = np.random.uniform(-0.5 * limit, 0.5 * limit, size=out_dim)
            parts.extend([w, b])
        pop.append(np.concatenate(parts))
    return np.array(pop)


def calc_mse(chromosome):
    Z_pred = nn.forward(X_train, chromosome)
    return np.mean((Z_train - Z_pred) ** 2)


def tournament_selection(pop, errors, k=TOURNAMENT_K):
    idx = np.random.randint(0, len(pop), size=k)
    best = idx[np.argmin(errors[idx])]
    return pop[best].copy()


def blx_alpha_crossover(p1, p2, alpha=0.3):
    """Кросовер BLX-alpha (зберігає різноманіття хромосом)."""
    if np.random.rand() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    low = np.minimum(p1, p2)
    high = np.maximum(p1, p2)
    diff = high - low
    c1 = np.random.uniform(low - alpha * diff, high + alpha * diff)
    c2 = np.random.uniform(low - alpha * diff, high + alpha * diff)
    return c1, c2


def adaptive_mutation(chrom, gen, max_gen):
    """Точкова гаусівська мутація зі зменшенням кроку на пізніх ітераціях."""
    progress = gen / max_gen
    std = 0.25 * (1.0 - 0.92 * progress)
    mask = np.random.rand(len(chrom)) < MUTATION_RATE
    chrom[mask] += np.random.normal(0.0, std, size=np.sum(mask))
    return np.clip(chrom, -20.0, 20.0)


# ==========================================================
# 5. ОСНОВНИЙ ЦИКЛ ГЕНЕТИЧНОГО АЛГОРИТМУ
# ==========================================================
start_time = time.time()
population = init_population(POP_SIZE, LAYERS)
mse_history = []
best_chromosome = None
best_mse = float("inf")

for gen in range(GENERATIONS):
    # Крок 2: Оцінювання популяції
    errors = np.array([calc_mse(ind) for ind in population])
    sorted_idx = np.argsort(errors)

    if errors[sorted_idx[0]] < best_mse:
        best_mse = errors[sorted_idx[0]]
        best_chromosome = population[sorted_idx[0]].copy()

    mse_history.append(best_mse)

    if gen % 50 == 0 or gen == GENERATIONS - 1:
        print(f"Ітерація {gen:4d} | Найкраща MSE: {best_mse:.6f} | RMSE: {np.sqrt(best_mse):.6f}")

    # Крок 3: Критерій зупинки
    if best_mse <= TARGET_MSE:
        print(f"Зупинка на ітерації {gen}: досягнуто цільової точності MSE <= {TARGET_MSE}.")
        break

    # Крок 6: Елітизм
    new_population = [population[i].copy() for i in sorted_idx[:ELITE_COUNT]]

    # Кроки 4-5: Селекція, схрещування та мутація
    while len(new_population) < POP_SIZE:
        p1 = tournament_selection(population, errors)
        p2 = tournament_selection(population, errors)
        c1, c2 = blx_alpha_crossover(p1, p2)
        new_population.append(adaptive_mutation(c1, gen, GENERATIONS))
        if len(new_population) < POP_SIZE:
            new_population.append(adaptive_mutation(c2, gen, GENERATIONS))

    population = np.array(new_population)

elapsed = time.time() - start_time


# ==========================================================
# 6. ОЦІНКА ЯКОСТІ НА ТЕСТОВІЙ ВИБІРЦІ
# ==========================================================
Z_pred_test = nn.forward(X_test, best_chromosome)
abs_err = np.abs(Z_test - Z_pred_test)
mse_test = np.mean((Z_test - Z_pred_test) ** 2)
rmse_test = np.sqrt(mse_test)
mae_test = np.mean(abs_err)
max_err = np.max(abs_err)
ss_res = np.sum((Z_test - Z_pred_test) ** 2)
ss_tot = np.sum((Z_test - np.mean(Z_test)) ** 2)
r2 = 1.0 - ss_res / ss_tot

print("\n=== РЕЗУЛЬТАТИ НА ТЕСТОВІЙ СІТЦІ ===")
print(f"Час навчання:            {elapsed:.2f} с")
print(f"MSE (тест):              {mse_test:.6f}")
print(f"RMSE (тест):             {rmse_test:.6f}")
print(f"MAE (середня різниця):   {mae_test:.6f}")
print(f"Макс. абсолютна похибка: {max_err:.6f}")
print(f"Коефіцієнт R^2:          {r2:.6f}")


# ==========================================================
# 7. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
# ==========================================================
Z_true_grid = Z_test.reshape(X1_te.shape)
Z_pred_grid = Z_pred_test.reshape(X1_te.shape)
Err_grid = np.abs(Z_true_grid - Z_pred_grid)

# Графік 1: Збіжність ГА
fig1 = plt.figure(figsize=(7, 5))
ax1 = fig1.add_subplot(111)
ax1.plot(mse_history, color="tab:blue", linewidth=1.8)
ax1.set_title("Збіжність ГА (MSE)")
ax1.set_xlabel("Ітерація (покоління)")
ax1.set_ylabel("MSE")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3)
fig1.tight_layout()

# Графік 2: Еталонна поверхня
fig2 = plt.figure(figsize=(7, 5))
ax2 = fig2.add_subplot(111, projection="3d")
ax2.plot_surface(X1_te, X2_te, Z_true_grid, cmap="viridis", edgecolor="none")
ax2.set_title("Еталон: 7*sin(2.5*cos(x1))")
ax2.set_xlabel("x1")
ax2.set_ylabel("x2")
ax2.set_zlabel("y")
fig2.tight_layout()

# Графік 3: Апроксимація нейромережею
fig3 = plt.figure(figsize=(7, 5))
ax3 = fig3.add_subplot(111, projection="3d")
ax3.plot_surface(X1_te, X2_te, Z_pred_grid, cmap="plasma", edgecolor="none")
ax3.set_title("Вихід ШНМ")
ax3.set_xlabel("x1")
ax3.set_ylabel("x2")
ax3.set_zlabel("y_pred")
fig3.tight_layout()

# Графік 4: Карта абсолютних похибок
fig4 = plt.figure(figsize=(7, 5))
ax4 = fig4.add_subplot(111)
im = ax4.imshow(
    Err_grid,
    extent=[INTERVAL[0], INTERVAL[1], INTERVAL[0], INTERVAL[1]],
    origin="lower",
    cmap="inferno",
    aspect="auto",
)
fig4.colorbar(im, ax=ax4)
ax4.set_title("Абсолютна похибка |y - y_pred|")
ax4.set_xlabel("x1")
ax4.set_ylabel("x2")
fig4.tight_layout()

plt.show()