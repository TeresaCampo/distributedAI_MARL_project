import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Connettiti all'API
api = wandb.Api()

# Sostituisci con il tuo percorso: "tuo_username/MARL_Project_GridWorld"
# Lo trovi nell'URL di W&B dopo wandb.ai/....
ENTITY_PROJECT = "299011-unimore/MARL_IQ_sparse" 

print("Scaricamento dati in corso... (può richiedere qualche secondo)")
runs = api.runs(ENTITY_PROJECT)

summary_list = [] 
config_list = []

for run in runs: 
    # Filtriamo solo i run che sono finiti o in corso (evitiamo i crashati se vuoi)
    # Esempio: prendiamo solo quelli del gruppo Independent_IQ
    if "Independent_IQ" in run.group:
        
        # Scarica la history (la curva di apprendimento)
        # keys=['episode', 'test/avg_reward'] sono le metriche che ti interessano
        history = run.history(keys=["episode", "test/avg_reward","test/avg_steps", "test/success_rate"])
        
        # Aggiungiamo le colonne di configurazione per poter separare i grafici dopo
        # run.config contiene i parametri salvati (grid_size, reward_type, ecc.)
        history["grid_size"] = run.config.get("grid_size")
        history["reward_type"] = run.config.get("reward_type")
        history["seed"] = run.config.get("seed")
        history["algorithm"] = run.config.get("algorithm")
        
        summary_list.append(history)

# Creiamo un unico DataFrame gigante con tutti i dati
df = pd.concat(summary_list)

# Pulizia: Assicuriamoci che i dati siano numerici
df["test/avg_reward"] = pd.to_numeric(df["test/avg_reward"])
df["episode"] = pd.to_numeric(df["episode"])

print(f"Scaricamento completato. Righe totali: {len(df)}")

sns.set_theme(style="whitegrid")

# CREAZIONE DELLA GRIGLIA DI GRAFICI
g = sns.relplot(
    data=df,
    x="episode", 
    y="test/avg_reward",
    kind="line",            # Linea continua
    hue="algorithm",        # Colore diverso per algoritmo (se ne hai più di uno)
    col="grid_size",        # <--- COLONNE diverse per dimensione griglia
    row="reward_type",      # <--- RIGHE diverse per tipo reward
    height=4, 
    aspect=1.5,
    errorbar="sd",          # Mostra l'ombra della Deviazione Standard (sui 3 seed)
    linewidth=2.5
)

# Abbellimenti
g.fig.suptitle("Analisi Performance MARL: Grid Size vs Reward Type", y=1.02, fontsize=16)
g.set_titles("Grid: {col_name} | Reward: {row_name}") # Titoli dei singoli riquadri
g.set_axis_labels("Episodi", "Reward Media (Test)")

# Aggiungi una linea di riferimento (es. successo = 100)
for ax in g.axes.flat:
    ax.axhline(100, ls='--', c='red', alpha=0.5, label='Target')

plt.show()