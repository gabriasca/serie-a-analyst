# AGENTS.md

## Descrizione progetto

Serie A Analyst e una web app locale Streamlit per analisi dati della Serie A.
Il focus e su import CSV, persistenza SQLite, dashboard semplici e predictor spiegabile.

## Struttura cartelle

```text
app.py
requirements.txt
README.md
AGENTS.md
data/
  serie_a.db
  raw/
    serie_a_matches.csv
src/
  __init__.py
  config.py
  db.py
  data_import.py
  demo_data.py
  analytics.py
  predictor.py
  explain.py
pages/
  1_Import_Dati.py
  2_Dashboard_Serie_A.py
  3_Analisi_Squadra.py
  4_Confronto_Squadre.py
  5_Predictor_Partita.py
```

## Comandi utili

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
python -m compileall app.py src pages
python -c "import app; import src.analytics; import src.predictor"
```

## Regole per future modifiche

- mantenere il progetto nella cartella corrente, senza creare sottoprogetti inutili
- non introdurre API esterne o scraping in questo MVP
- non rimuovere file senza conferma esplicita
- privilegiare moduli piccoli e leggibili
- tenere la logica analitica fuori dalle pagine Streamlit
- non presentare mai le previsioni come certezze

## Convenzioni codice

- Python semplice e modulare
- SQLite solo tramite `sqlite3`
- Pandas per trasformazioni dati
- date sempre normalizzate in formato ISO `YYYY-MM-DD`
- commenti brevi solo quando chiariscono logica non ovvia
- evitare dipendenze aggiuntive non strettamente necessarie

## Cosa significa completare bene una task

- il codice e coerente con la struttura attuale
- le pagine Streamlit restano leggere
- l'import CSV continua a funzionare con alias comuni
- database e predictor non si rompono con dati mancanti o scarsi
- la documentazione resta aggiornata se cambia il comportamento dell'app
