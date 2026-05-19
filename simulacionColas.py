import simpy
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
from collections import deque

# ─── Parámetros (datos reales Víveres Emmanuel) ───────────────────────────────
LAMBDA   = 20       # llegadas por hora
MU1      = 7      # tasa servicio despacho (por servidor)
MU2      = 20       # tasa servicio caja
S        = 3        # servidores de despacho
SIM_TIME = 60       # minutos a simular
SPEED    = 0.05    # segundos de pausa entre pasos (animación)
SEED     = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Estado compartido ────────────────────────────────────────────────────────
state = {
    "time": 0.0,
    "clients": [],          # lista de dicts con posición y estado
    "served": 0,
    "q_despacho": 0,        # clientes esperando turno en despacho
    "q_caja": 0,
    "dispatch_busy": [False] * S,
    "cashier_busy": False,
    "wait_times": [],
    "total_times": [],
    "arrivals_hist": [],
    "q_hist": [],
    "rho_hist": [],
    "time_hist": [],
}

CLIENT_STATES = {
    "arriving":   ("#4C9BE8", "Llegando"),
    "q_despacho": ("#82CFFF", "Fila despacho"),
    "dispatch":   ("#F5A623", "En despacho"),
    "q_caja":     ("#E8734C", "Fila caja"),
    "cashier":    ("#4CE87A", "Pagando"),
    "leaving":    ("#B07FE8", "Saliendo"),
}

# ─── SimPy ────────────────────────────────────────────────────────────────────
env         = simpy.Environment()
dispatch_res = simpy.Resource(env, capacity=S)
cashier_res  = simpy.Resource(env, capacity=1)

def client_process(env, cid):
    arrive = env.now
    c = {"id": cid, "state": "arriving", "x": 0.08, "y": 0.5,
         "tx": 0.08, "ty": 0.5, "color": CLIENT_STATES["arriving"][0]}
    state["clients"].append(c)

    # ── Etapa 1a: Fila de espera despacho ─────────────────────────────────
    c["state"] = "q_despacho"
    c["color"] = CLIENT_STATES["q_despacho"][0]
    # posición visual: cola horizontal antes del área de despacho
    c["tx"] = 0.16
    c["ty"] = 0.5
    state["q_despacho"] += 1

    with dispatch_res.request() as req:
        yield req
        state["q_despacho"] = max(0, state["q_despacho"] - 1)

        # ── Etapa 1b: Siendo atendido en servidor ─────────────────────────
        slot = next((i for i, b in enumerate(state["dispatch_busy"]) if not b), 0)
        state["dispatch_busy"][slot] = True
        c["state"] = "dispatch"
        c["color"] = CLIENT_STATES["dispatch"][0]
        # posición: centro del box del servidor asignado
        c["tx"] = 0.295
        c["ty"] = 0.18 + slot * (0.65 / S)
        svc1 = random.expovariate(MU1 / 60)
        yield env.timeout(svc1)
        state["dispatch_busy"][slot] = False

    # ── Etapa 2a: Fila de caja ────────────────────────────────────────────
    c["state"] = "q_caja"
    c["color"] = CLIENT_STATES["q_caja"][0]
    # posición visual: cola horizontal antes del cajero
    c["tx"] = 0.60
    c["ty"] = 0.5
    wait_start = env.now
    state["q_caja"] += 1

    with cashier_res.request() as req:
        yield req
        state["q_caja"] = max(0, state["q_caja"] - 1)
        wait = env.now - wait_start
        state["wait_times"].append(wait)

        # ── Etapa 2b: Siendo atendido en caja ────────────────────────────
        state["cashier_busy"] = True
        c["state"] = "cashier"
        c["color"] = CLIENT_STATES["cashier"][0]
        c["tx"] = 0.815
        c["ty"] = 0.5
        svc2 = random.expovariate(MU2 / 60)
        yield env.timeout(svc2)
        state["cashier_busy"] = False

    # ── Salida ─────────────────────────────────────────────────────────────
    total = env.now - arrive
    state["total_times"].append(total)
    state["served"] += 1
    c["state"] = "leaving"
    c["color"] = CLIENT_STATES["leaving"][0]
    c["tx"] = 0.97
    c["ty"] = 0.5 + random.uniform(-0.2, 0.2)
    yield env.timeout(0.3)
    if c in state["clients"]:
        state["clients"].remove(c)

def arrival_process(env):
    cid = 0
    while True:
        iat = random.expovariate(LAMBDA / 60)
        yield env.timeout(iat)
        cid += 1
        env.process(client_process(env, cid))

def stats_process(env):
    while True:
        yield env.timeout(0.5)
        state["time"] = env.now
        state["time_hist"].append(env.now)
        state["q_hist"].append(state["q_caja"])
        rho = min(1.0, (LAMBDA / 60) / (MU2 / 60))
        state["rho_hist"].append(rho * 100)
        state["arrivals_hist"].append(len(state["clients"]))

env.process(arrival_process(env))
env.process(stats_process(env))

# ─── Figura ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8), facecolor="#0D1117")
fig.suptitle("Simulación 2D — Modelo de Colas · Víveres Emmanuel, Cúcuta",
             color="white", fontsize=13, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 3, left=0.04, right=0.97,
                      top=0.93, bottom=0.07, hspace=0.42, wspace=0.35)

ax_sim   = fig.add_subplot(gs[0, :2])   # animación principal
ax_q     = fig.add_subplot(gs[0, 2])    # fila caja en el tiempo
ax_bar   = fig.add_subplot(gs[1, 0])    # distribución tiempos de espera
ax_occ   = fig.add_subplot(gs[1, 1])    # ocupación cajero
ax_stats = fig.add_subplot(gs[1, 2])    # métricas texto

for ax in [ax_sim, ax_q, ax_bar, ax_occ, ax_stats]:
    ax.set_facecolor("#161B22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")

# ─── Fondo escenario ─────────────────────────────────────────────────────────
def draw_scenario(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Sistema M/M/3 → M/M/1 en Serie", color="#8B949E",
                 fontsize=9, pad=4)

    # Zonas de fondo
    # x, y, w, h, facecolor, edgecolor, label
    zones = [
        (0.01,  0.05, 0.10, 0.9, "#1C2128", "#4C9BE8",  "ENTRADA"),
        (0.12,  0.05, 0.09, 0.9, "#1A1F27", "#82CFFF",  "FILA\nDESP."),
        (0.22,  0.05, 0.18, 0.9, "#1C2128", "#F5A623",  f"DESPACHO\n({S} serv.)"),
        (0.41,  0.05, 0.18, 0.9, "#1A1F27", "#E8734C",  "FILA\nCAJA"),
        (0.60,  0.05, 0.18, 0.9, "#1C2128", "#4CE87A",  "CAJERO"),
        (0.79,  0.05, 0.10, 0.9, "#1C2128", "#B07FE8",  "SALIDA"),
    ]
    for x, y, w, h, fc, ec, lbl in zones:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.01",
            facecolor=fc, edgecolor=ec, linewidth=0.8, alpha=0.55))
        ax.text(x + w/2, 0.945, lbl, ha="center", va="center",
                fontsize=6, color=ec, fontweight="bold")

    # Servidores de despacho (1 cliente max cada uno)
    slot_h = 0.60 / S
    for i in range(S):
        yp = 0.18 + i * (0.65 / S)
        col = "#F5A623" if state["dispatch_busy"][i] else "#2D333B"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.235, yp), 0.14, slot_h * 0.85,
            boxstyle="round,pad=0.01",
            facecolor=col, edgecolor="#F5A623", linewidth=0.8, alpha=0.75))
        ax.text(0.305, yp + slot_h * 0.42, f"S{i+1}",
                ha="center", va="center", fontsize=8, color="white")

    # Cajero (1 cliente max)
    col_c = "#4CE87A" if state["cashier_busy"] else "#2D333B"
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.625, 0.33), 0.14, 0.34, boxstyle="round,pad=0.01",
        facecolor=col_c, edgecolor="#4CE87A", linewidth=0.8, alpha=0.75))
    ax.text(0.695, 0.50, "💳\nCAJA", ha="center", va="center",
            fontsize=8, color="white")

    # Flecha flujo general
    ax.annotate("", xy=(0.80, 0.5), xytext=(0.02, 0.5),
                arrowprops=dict(arrowstyle="->", color="#30363D",
                                lw=1.0, connectionstyle="arc3,rad=0"))

draw_scenario(ax_sim)

# ─── Puntos de clientes ───────────────────────────────────────────────────────
scat = ax_sim.scatter([], [], s=80, zorder=5, edgecolors="white", linewidths=0.4)
ids_text = []

# ─── Gráfica fila caja ────────────────────────────────────────────────────────
ax_q.set_title("Fila en Caja (t)", color="#8B949E", fontsize=9)
ax_q.set_xlabel("Tiempo (min)", color="#8B949E", fontsize=7)
ax_q.set_ylabel("Clientes en fila", color="#8B949E", fontsize=7)
ax_q.tick_params(colors="#8B949E", labelsize=7)
line_q, = ax_q.plot([], [], color="#E8734C", lw=1.5)
ax_q.set_xlim(0, SIM_TIME); ax_q.set_ylim(0, 10)

# ─── Histograma espera ────────────────────────────────────────────────────────
ax_bar.set_title("Tiempos de espera caja", color="#8B949E", fontsize=9)
ax_bar.set_xlabel("Minutos", color="#8B949E", fontsize=7)
ax_bar.set_ylabel("Frecuencia", color="#8B949E", fontsize=7)
ax_bar.tick_params(colors="#8B949E", labelsize=7)

# ─── Ocupación cajero ─────────────────────────────────────────────────────────
ax_occ.set_title("Ocupación cajero ρ(t)", color="#8B949E", fontsize=9)
ax_occ.set_xlabel("Tiempo (min)", color="#8B949E", fontsize=7)
ax_occ.set_ylabel("Ocupación %", color="#8B949E", fontsize=7)
ax_occ.tick_params(colors="#8B949E", labelsize=7)
ax_occ.axhline(91, color="#E84C4C", lw=1, ls="--", alpha=0.6)
ax_occ.text(2, 92, "ρ real=91%", color="#E84C4C", fontsize=7)
line_rho, = ax_occ.plot([], [], color="#4C9BE8", lw=1.5)
ax_occ.set_xlim(0, SIM_TIME); ax_occ.set_ylim(0, 110)

# ─── Panel de métricas ────────────────────────────────────────────────────────
ax_stats.axis("off")
stats_title = ax_stats.text(0.5, 0.97, "Métricas en vivo", ha="center",
                             va="top", color="white", fontsize=9, fontweight="bold",
                             transform=ax_stats.transAxes)
stats_body = ax_stats.text(0.05, 0.82, "", ha="left", va="top",
                            color="#C9D1D9", fontsize=8.5,
                            transform=ax_stats.transAxes,
                            family="monospace")

# ─── Leyenda ─────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=v[0], label=v[1])
    for v in CLIENT_STATES.values()
]
ax_sim.legend(handles=legend_patches, loc="lower left",
              fontsize=7, facecolor="#161B22", edgecolor="#30363D",
              labelcolor="white", framealpha=0.9, ncol=2)

# ─── Loop de animación ────────────────────────────────────────────────────────
step = [0]

def animate(frame):
    if env.now < SIM_TIME:
        try:
            env.step()
        except simpy.core.EmptySchedule:
            pass

    # Suavizar posiciones
    for c in state["clients"]:
        c["x"] += (c["tx"] - c["x"]) * 0.15
        c["y"] += (c["ty"] - c["y"]) * 0.15

    # Redibujar escenario
    ax_sim.cla()
    draw_scenario(ax_sim)

    # Posiciones visuales de las filas
    # Fila despacho: columna vertical centrada en zona de fila despacho
    q_desp = sorted(
        [c for c in state["clients"] if c["state"] == "q_despacho"],
        key=lambda x: x["id"])
    for i, c in enumerate(q_desp):
        c["tx"] = 0.165
        c["ty"] = 0.75 - i * 0.10   # de arriba hacia abajo

    # En servidor: ya tienen tx/ty asignados al slot
    # Fila caja: columna vertical centrada en zona de fila caja
    q_caj = sorted(
        [c for c in state["clients"] if c["state"] == "q_caja"],
        key=lambda x: x["id"])
    for i, c in enumerate(q_caj):
        c["tx"] = 0.500
        c["ty"] = 0.75 - i * 0.10

    # Clientes
    if state["clients"]:
        xs = [c["x"] for c in state["clients"]]
        ys = [c["y"] for c in state["clients"]]
        cs = [c["color"] for c in state["clients"]]
        ax_sim.scatter(xs, ys, c=cs, s=90, zorder=5,
                       edgecolors="white", linewidths=0.4)
        for c in state["clients"]:
            ax_sim.text(c["x"], c["y"], str(c["id"]),
                        ha="center", va="center", fontsize=5.5,
                        color="white", zorder=6)

    # Gráfica fila
    if state["time_hist"]:
        line_q.set_data(state["time_hist"], state["q_hist"])
        ax_q.set_ylim(0, max(max(state["q_hist"]) + 1, 5))

    # Histograma espera
    if len(state["wait_times"]) > 2:
        ax_bar.cla()
        ax_bar.set_facecolor("#161B22")
        for sp in ax_bar.spines.values():
            sp.set_edgecolor("#30363D")
        ax_bar.hist(state["wait_times"], bins=12, color="#E8734C",
                    edgecolor="#0D1117", alpha=0.85)
        ax_bar.axvline(np.mean(state["wait_times"]), color="#F5A623",
                       lw=1.5, ls="--")
        ax_bar.set_title("Tiempos de espera caja", color="#8B949E", fontsize=9)
        ax_bar.set_xlabel("Minutos", color="#8B949E", fontsize=7)
        ax_bar.tick_params(colors="#8B949E", labelsize=7)

    # Ocupación
    if state["time_hist"]:
        line_rho.set_data(state["time_hist"], state["rho_hist"])

    # Métricas
    w_avg = np.mean(state["wait_times"]) if state["wait_times"] else 0
    t_avg = np.mean(state["total_times"]) if state["total_times"] else 0
    rho_c = min(100, round(LAMBDA / MU2 * 100))
    rho_d = min(100, round(LAMBDA / (MU1 * S) * 100))
    txt = (
        f"  Tiempo sim   : {env.now:5.1f} min\n"
        f"  Atendidos    : {state['served']:5d}\n"
        f"  En sistema   : {len(state['clients']):5d}\n"
        f"  Fila desp.   : {state['q_despacho']:5d}\n"
        f"  Fila caja    : {state['q_caja']:5d}\n"
        f"───────────────────────\n"
        f"  Espera prom  : {w_avg:5.1f} min\n"
        f"  Tiempo total : {t_avg:5.1f} min\n"
        f"───────────────────────\n"
        f"  ρ despacho   : {rho_d:4d}%\n"
        f"  ρ cajero     : {rho_c:4d}%  {'⚠ SAT.' if rho_c>=90 else '✓ OK'}\n"
        f"  λ entrada    : {LAMBDA:4d} /hr\n"
        f"  μ caja       : {MU2:4d} /hr"
    )
    stats_body.set_text(txt)

    step[0] += 1
    return []

ani = animation.FuncAnimation(
    fig, animate,
    frames=int(SIM_TIME / 0.03),
    interval=int(SPEED * 1000),
    blit=False, repeat=False
)

plt.show()