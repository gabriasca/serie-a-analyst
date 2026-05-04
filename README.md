# Serie A Analyst

Serie A Analyst e un MVP locale costruito con Python, Streamlit, Pandas e SQLite per analizzare dati calcistici della Serie A.

L'app consente di:

- importare dati da CSV
- salvare e deduplicare partite in SQLite
- arricchire le squadre con team ratings Elo seedati da fonte pubblica
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

## Team Ratings

L'app supporta un primo layer informativo di team ratings basato su Elo.

Caratteristiche:

- il seed pubblico e `data/raw/team_ratings_seed.csv`
- il tipo di rating usato in questo step e `elo`
- la fonte pubblica di riferimento e ClubElo
- i rating vengono salvati nella tabella SQLite `team_ratings`
- in questo step i rating non modificano predictor o proiezioni

Uso attuale nell'app:

- `Profilo Squadra` mostra rating Elo attuale e fascia forza relativa
- `Report Partita` mostra il rating Elo delle due squadre, se disponibile
- il blocco forte/medio/debole in `Profilo Squadra` usa il ranking Elo solo quando il seed copre tutte le squadre della stagione; altrimenti torna automaticamente alla classifica corrente

I rating Elo non vanno confusi con le statistiche partita:

- il rating Elo riassume forza storica/recente in un numero sintetico
- le statistiche partita descrivono cio che la squadra sta producendo nella stagione corrente
- i due livelli sono complementari ma non equivalenti

## Metriche avanzate interne

L'app include anche un primo layer `Metriche Avanzate v1`, costruito solo con i dati gia presenti nel database.

Indicatori principali:

- pericolosita offensiva
- solidita difensiva
- volume offensivo
- rischio difensivo
- efficienza realizzativa
- dipendenza casa
- momento recente
- forza calendario

Caratteristiche:

- ogni indice e espresso su scala `0-100`
- `50` rappresenta circa la media del campionato nella stagione letta
- gli indici sono interni e non sono `xG` reali
- la forza calendario usa Elo quando il seed copre la stagione; altrimenti usa la classifica corrente

Uso attuale nell'app:

- `Metriche Avanzate` mostra tabella squadre, dettaglio per team e classifiche rapide
- `Profilo Squadra` integra i principali indici avanzati nel profilo e nei punti forti/deboli
- `Report Partita` confronta le due squadre su pericolosita offensiva, solidita difensiva e momento recente

Limiti:

- non sostituiscono un modello evento-per-evento
- soffrono dataset incompleti o colonne mancanti su tiri, tiri in porta o corner
- con campioni piccoli descrivono tendenze utili, ma non stabili
- non entrano ancora nel predictor e non modificano le proiezioni

## Studio Squadra

La pagina `Studio Squadra` approfondisce l'identita stagionale di un club usando solo dati gia presenti nel database.

Ogni blocco separa:

- dati osservati, come risultati, gol, casa/fuori e rendimento per fascia avversaria
- indicatori interni, come pericolosita offensiva, solidita difensiva, momento recente e volatilita
- ipotesi prudenti, cioe letture condizionali basate su regole semplici e non presentate come certezze

Cosa analizza:

- come la squadra tende a vincere, perdere o pareggiare
- contro quali fasce di avversarie rende meglio o peggio
- differenza tra identita casa e trasferta
- trend recente rispetto alla media stagionale
- stabilita o volatilita del rendimento
- cosa manca per una vera analisi tattica avanzata

Limiti:

- non usa API esterne o scraping
- non inventa dati tattici non presenti
- non puo affermare pressing, costruzione dal basso, possesso o lineups senza dati evento e giocatori
- non modifica predictor o Proiezione Classifica

## Matchup Analysis

La pagina `Matchup Analysis` serve a leggere perche una squadra puo trovarsi meglio o peggio contro un'avversaria specifica.

Usa:

- classifica e punti
- forma recente
- rendimento casa/fuori
- rating Elo informativo, se disponibile
- Profilo Squadra / DNA Squadra
- Metriche Avanzate interne
- predictor esistente, senza modificarlo

Cosa produce:

- riepilogo statistico della sfida
- predictor sintetico con probabilita e score piu probabili
- confronto diretto delle metriche interne
- mismatch principali tra attacco, difesa, forma e contesto
- motore di peso del dato e contesto con edge corretto, rischio pareggio, rischio upset e confidenza
- rischi specifici per squadra casa e squadra trasferta
- sintesi finale del matchup

Differenza rispetto al predictor:

- il predictor stima probabilita e punteggi
- Matchup Analysis spiega il perche del confronto con regole semplici e leggibili
- le due letture sono complementari, non equivalenti

Limiti:

- non usa dati evento-per-evento o fonti esterne aggiuntive
- resta sensibile a campioni piccoli o a stagioni con dati incompleti
- il rating Elo e solo informativo
- non trasforma mai il dato statistico in una certezza

## Analisi Giornata

La pagina `Analisi Giornata` genera una lettura partita per partita della prossima giornata o delle prossime partite disponibili.

Usa:

- predictor base Poisson come baseline numerica
- Predictor contestuale v2 come correzione prudente e spiegabile
- Matchup Analysis, context_engine, metriche avanzate, Elo e calendario/riposo
- classifica corrente, rendimento casa/fuori e forma recente

Fonte partite:

- se esiste `data/raw/serie_a_fixtures_seed.csv`, la pagina usa le partite future non ancora giocate presenti nel seed
- se il fixture seed non esiste, usa le partite mancanti inferite dal calendario home/away, con warning esplicito
- la UI non scarica internet, non chiama API e non usa calendario ufficiale se non e stato seedato

Aggiornamento fixture seed:

- lo script `scripts/update_serie_a_fixtures_seed.py` aggiorna `data/raw/serie_a_fixtures_seed.csv`
- la URL si configura con `FOOTBALL_DATA_SERIE_A_FIXTURES_URL`
- se la variabile non e impostata, usa la stessa URL football-data gia configurata per il seed Serie A
- il workflow `.github/workflows/update-fixtures.yml` puo girare manualmente o ogni giorno
- l'aggiornamento e best-effort: se la fonte non e raggiungibile e il fixture seed esiste gia, il file esistente viene mantenuto
- lo script non modifica `data/raw/serie_a_seed.csv` e non scrive nel database

Cosa produce:

- tabella riepilogo con probabilita base e contestuali `1 / X / 2`
- risultato piu probabile del modello base
- confidence, draw risk, upset risk, volatilita e interesse del match
- dettaglio con narrativa prudente, fattori chiave, possibili imprevisti e dati mancanti
- sintesi della giornata con partita piu equilibrata, maggiore rischio pareggio, maggiore rischio upset e maggiore volatilita

Limiti:

- non usa quote o scommesse
- non inventa lineup, assenze, infortuni o squalifiche
- non usa xG reali shot-by-shot
- se mancano fixture seed o competizioni extra, la lettura del calendario e parziale
- non modifica predictor base e non modifica la `Proiezione Classifica`

## Schedule & Competition Context

Il layer `Schedule & Competition Context v1` misura riposo, carico partite e differenza tra forma campionato e forma su tutte le competizioni disponibili.

Misura:

- giorni di riposo dall'ultima partita disponibile
- partite giocate negli ultimi 7, 14 e 30 giorni
- numero di competizioni recenti presenti nel database
- forma ultime 5 solo campionato
- forma ultime 5 su tutte le competizioni disponibili
- possibile vantaggio riposo rispetto all'avversaria

Uso attuale nell'app:

- `Matchup Analysis` mostra calendario, riposo, carico e forma all competitions
- `Report Partita` aggiunge una lettura sintetica di calendario/riposo
- `Profilo Squadra` indica se la forma recente e solo campionato o multi-competizione
- `Studio Squadra` mostra carico recente 7/14/30 giorni e confronto Serie A vs tutte le competizioni disponibili
- `context_engine` usa il fattore calendario con peso prudente e non dominante
- `Model Review` lo valuta tra i fattori, segnalando quando il campione e solo Serie A

Limiti:

- usa solo le partite gia presenti nella tabella `matches`
- se mancano Coppa Italia o coppe europee, la lettura del carico e parziale
- diventera piu utile quando verranno importate competizioni extra Serie A
- non modifica predictor o Proiezione Classifica

## Model Review e calibrazione context engine

La pagina `Model Review` esegue un backtest storico partita per partita usando solo i dati disponibili prima di ogni match.

Serve a verificare se:

- `adjusted_edge` migliora davvero la lettura del `base_edge`
- `draw_risk` e `upset_risk` separano partite utili da partite piu aperte
- `confidence` distingue davvero i contesti piu coerenti da quelli piu fragili
- i fattori del `context_engine` stanno pesando nel modo giusto

Nota importante sul rating Elo:

- se il seed Elo disponibile e solo uno snapshot recente e non uno storico pre-match, nel `Model Review` viene trattato come layer informativo
- in quel caso il factor review lo segnala come `informativo / non usato per calibrazione storica`
- questo evita di scambiare per "fattore neutro" un dato che in realta non era disponibile storicamente nel backtest

## Review Predictor contestuale v2

La `Model Review` include anche un confronto tra predictor base Poisson e Predictor contestuale v2.

- confronta accuracy 1/X/2, favorito che non perde e Brier score
- mostra quando il v2 cambia pick, migliora o peggiora rispetto alla baseline
- analizza bucket di `confidence`, `draw_risk` e `upset_risk`
- permette di scegliere campione, warmup minimo per squadra e numero massimo di partite analizzate
- mostra affidabilita del campione, copertura predictor/v2 e avvisi su bucket piccoli o v2 troppo neutro
- non modifica `src/predictor.py` e non modifica la `Proiezione Classifica`

## Limiti dell'MVP

- non usa API esterne
- non usa scraping
- non include quote scommesse
- il modello predittivo non e una black box, ma resta semplice
- non gestisce ancora metriche evento-per-evento o infortuni
- i rating Elo sono informativi e non guidano ancora il predictor

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

## Modalita pubblica

L'app usa una modalita read-only per default.

Logica:

- se `SERIE_A_ANALYST_MODE` non e impostata, l'app parte in modalita `public`
- se `SERIE_A_ANALYST_MODE` e diversa da `local`, l'app resta in modalita pubblica

In modalita pubblica:

- dashboard, analisi squadra, confronto, predictor e proiezioni restano disponibili
- la pagina `Import Dati` diventa informativa
- non sono disponibili upload CSV, svuotamento database, eliminazione stagioni o caricamento demo
- il database online va considerato temporaneo e non persistente

## Modalita locale/admin

Per usare gli strumenti di import e gestione dati in locale:

```powershell
$env:SERIE_A_ANALYST_MODE="local"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In modalita locale/admin:

- puoi importare CSV reali
- puoi caricare il dataset demo
- puoi eliminare una stagione
- puoi svuotare il database locale

## Come aggiornare i dati pubblici

Flusso consigliato:

1. Avvia l'app in modalita locale/admin.
2. Importa il nuovo CSV Serie A.
3. Verifica dashboard, predictor e proiezioni.
4. Esporta il nuovo seed pubblico.
5. Fai commit e push su GitHub.
6. Lascia che Streamlit Community Cloud faccia il redeploy della snapshot aggiornata.

Comando PowerShell per esportare il seed:

```powershell
.\.venv\Scripts\python.exe scripts\export_seed.py
```

Lo script legge `data/serie_a.db` ed esporta `data/raw/serie_a_seed.csv`, che e la snapshot usata dalla versione pubblica.

## Come aggiornare i team ratings pubblici

La repo include uno script dedicato per aggiornare il seed Elo:

- `scripts/update_clubelo_seed.py`
- output: `data/raw/team_ratings_seed.csv`

Comportamento:

- legge la URL da `CLUBELO_RATINGS_URL` se presente
- altrimenti usa la pagina Italy ufficiale di ClubElo come default
- normalizza i nomi squadra con mapping prudente
- l'aggiornamento ClubElo e best-effort: se la fonte esterna non e raggiungibile, viene mantenuto il seed rating esistente
- salva solo il seed CSV
- non modifica direttamente il database locale

Esecuzione locale:

```powershell
.\.venv\Scripts\python.exe scripts\update_clubelo_seed.py
```

Il bootstrap dell'app carica automaticamente i rating nel database se `data/raw/team_ratings_seed.csv` esiste.

## Aggiornamento automatico dei dati

La repo include uno script e un workflow GitHub Actions per aggiornare automaticamente `data/raw/serie_a_seed.csv`.

Componenti:

- `scripts/update_football_data_seed.py`
- `.github/workflows/update-data.yml`
- `scripts/update_clubelo_seed.py`
- `.github/workflows/update-ratings.yml`

Come funziona:

- GitHub Actions scarica il CSV Serie A dalla fonte configurata
- normalizza e valida i dati con la stessa logica usata dall'app
- aggiorna `data/raw/serie_a_seed.csv`
- se il seed e cambiato, fa commit e push automatico
- se non ci sono cambiamenti, non crea commit inutili

Il workflow parte:

- manualmente con `workflow_dispatch`
- automaticamente una volta al giorno

Per i team ratings il workflow dedicato:

- aggiorna `data/raw/team_ratings_seed.csv`
- crea commit solo se il file cambia
- esegue `fetch` e `pull --rebase` prima del push per ridurre errori `fetch first`

## Come lanciarlo manualmente da GitHub Actions

1. Vai nella repo su GitHub.
2. Apri la tab `Actions`.
3. Seleziona il workflow `Update Serie A Seed Data`.
4. Clicca `Run workflow`.
5. Attendi il completamento.
6. Se il seed cambia, GitHub fara un commit automatico e Streamlit Community Cloud vedra l'aggiornamento dopo il push.

## URL dati configurabile

L'app e lo script supportano la variabile ambiente opzionale:

- `FOOTBALL_DATA_SERIE_A_URL`
- `CLUBELO_RATINGS_URL`

Se non e impostata, viene usata questa URL di default:

- [Football-Data Serie A CSV](https://www.football-data.co.uk/mmz4281/2526/I1.csv)

Per cambiare URL nel workflow GitHub Actions:

1. Vai in `Settings` della repository.
2. Apri `Secrets and variables` -> `Actions`.
3. Crea o modifica una repository variable chiamata `FOOTBALL_DATA_SERIE_A_URL` oppure `CLUBELO_RATINGS_URL`.
4. Inserisci la nuova URL della fonte corrispondente.

## Limiti della fonte dati

- La fonte e gratuita e pubblica, ma non e controllata dall'app.
- La struttura del CSV potrebbe cambiare nel tempo.
- La URL di default e legata alla stagione corrente pubblicata in repo.
- Quando cambia stagione, potrebbe essere necessario aggiornare la URL di default o la variabile `FOOTBALL_DATA_SERIE_A_URL`.
- Il workflow aggiornera il seed solo se la fonte esterna e raggiungibile e contiene dati validi.

## Limiti dei team ratings

- il rating Elo e un indicatore sintetico e non sostituisce le statistiche delle partite correnti
- la disponibilita dipende da una fonte pubblica esterna
- se una squadra non viene mappata correttamente, il rating resta vuoto ma l'app continua a funzionare
- il seed rating puo essere temporaneamente meno aggiornato del seed partite
- in questo step il predictor non usa ancora i rating Elo
