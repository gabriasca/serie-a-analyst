# Serie A Analyst

Serie A Analyst e un MVP locale costruito con Python, Streamlit, Pandas e SQLite per analizzare dati calcistici della Serie A.

L'app consente di:

- importare dati da CSV
- salvare e deduplicare partite in SQLite
- generare dashboard di campionato
- analizzare una singola squadra
- confrontare due squadre
- stimare una partita con un modello semplice e spiegabile

## Requisiti

- Windows con PowerShell
- Python 3.10 o superiore consigliato

## Installazione su Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Dati CSV supportati

L'app si aspetta internamente queste colonne canoniche:

- `season`
- `match_date`
- `home_team`
- `away_team`
- `home_goals`
- `away_goals`
- `full_time_result`
- `home_shots`
- `away_shots`
- `home_shots_on_target`
- `away_shots_on_target`
- `home_corners`
- `away_corners`
- `home_cards`
- `away_cards`

Sono supportati anche alias comuni dei CSV calcistici, ad esempio:

- `Date` -> `match_date`
- `HomeTeam` -> `home_team`
- `AwayTeam` -> `away_team`
- `FTHG` -> `home_goals`
- `FTAG` -> `away_goals`
- `FTR` -> `full_time_result`
- `HS` -> `home_shots`
- `AS` -> `away_shots`
- `HST` -> `home_shots_on_target`
- `AST` -> `away_shots_on_target`
- `HC` -> `home_corners`
- `AC` -> `away_corners`
- `HY` -> `home_cards`
- `AY` -> `away_cards`

Se la colonna `season` non e presente nel CSV, puo essere inserita manualmente nella pagina di import.

## Predictor

Il predictor usa un modello semplice e leggibile:

- media gol del campionato
- forza offensiva di ogni squadra
- forza difensiva di ogni squadra
- vantaggio casa
- forma recente sulle ultime 5 partite
- distribuzione di Poisson per stimare i punteggi possibili

Output principali:

- expected goals casa e trasferta
- probabilita `1 / X / 2`
- risultato piu probabile
- spiegazione testuale dei fattori usati

## Limiti dell'MVP

- non usa API esterne
- non usa scraping
- non include quote scommesse
- il modello predittivo non e una black box, ma resta semplice
- non gestisce ancora metriche avanzate come xG reali, Elo o infortuni

## Prossimi step consigliati

- aggiungere piu stagioni e dataset piu completi
- introdurre pulizia avanzata dei nomi squadra
- salvare metadati di import e storico caricamenti
- arricchire il predictor con metriche aggiuntive
- costruire la futura sezione "chiedi all'analista" sopra i moduli attuali

## Deploy su Streamlit Community Cloud

Questa repo e pronta per essere pubblicata su Streamlit Community Cloud con `app.py` come entrypoint.

### Scelta consigliata per i dati

La soluzione piu sicura per il deploy e usare un CSV seed versionato nella repo, non il file SQLite locale.

In questa repo:

- `data/raw/serie_a_seed.csv` e il dataset seed da includere su GitHub
- `data/serie_a.db` e un database locale di lavoro da non versionare

Motivo:

- il CSV seed e leggibile, riproducibile e facile da aggiornare
- il database SQLite e un file binario, meno trasparente e meno comodo da mantenere su Git
- su Streamlit Community Cloud il filesystem locale non va considerato persistente: eventuali modifiche online al database possono andare perse dopo riavvio o redeploy

L'app, su un ambiente nuovo dove `data/serie_a.db` non esiste, crea automaticamente il database e lo inizializza da `data/raw/serie_a_seed.csv`.

### File da includere su GitHub

- `app.py`
- `pages/`
- `src/`
- `requirements.txt`
- `README.md`
- `AGENTS.md`
- `data/raw/serie_a_seed.csv`
- `data/raw/serie_a_matches.csv` se vuoi mantenere anche il demo locale
- `.gitignore`

### File da non includere su GitHub

- `.venv/`
- `__pycache__/`
- `*.pyc`
- `data/serie_a.db`

### Passi per pubblicarla

1. Crea una repository GitHub e carica il progetto.
2. Verifica che `app.py` sia nella root della repo.
3. Verifica che `requirements.txt` sia nella root della repo.
4. Controlla che `data/raw/serie_a_seed.csv` sia presente nella repo.
5. Vai su [Streamlit Community Cloud](https://share.streamlit.io/) e accedi con GitHub.
6. Clicca su `New app`.
7. Seleziona la repository GitHub, il branch e imposta come main file path `app.py`.
8. In `Advanced settings`, lascia o seleziona Python `3.12` se disponibile.
9. Avvia il deploy.
10. Alla prima esecuzione online, l'app creera `data/serie_a.db` nel filesystem del container e lo popolera dal CSV seed.
11. Condividi il link pubblico generato da Streamlit.

### Note operative importanti

- La versione online e pensata per consultazione e test.
- Gli import o le cancellazioni fatti online sul database locale non devono essere considerati permanenti.
- Se vuoi aggiornare i dati pubblicati, aggiorna `data/raw/serie_a_seed.csv` nella repo e fai un nuovo deploy.
- Se in futuro vorrai persistenza reale online, converra spostare i dati su uno storage esterno o un database gestito.
