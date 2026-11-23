# Ritocco ricette e ricette personali (web UI)

## Perché
Dopo aver convertito una foto con una ricetta, l'utente vuole ritoccare qualche
parametro (es. più grana), vedere subito il risultato e, se gli piace, salvarlo
come ricetta sua da riusare su altre foto.

## Obiettivi
- Nel pannello ricetta, rendere modificabili tutti i parametri che la
  conversione applica.
- "Converti modifiche": render sulla fotocamera con le impostazioni ritoccate,
  mostrato nella striscia accanto alle altre versioni per confrontarle.
- "Salva come ricetta…": salva le impostazioni correnti con un nome; la ricetta
  compare subito nell'elenco (badge "Mie") ed è usabile anche da CLI
  (`fxraw convert -r <slug>`, `fxraw recipes`).

## Non obiettivi
- Caricare le ricette negli slot C1–C7 della fotocamera.
- Anteprima istantanea senza fotocamera: ogni ritocco è un render reale (~5 s).
- Modificare le ricette incluse nel programma (restano di sola lettura; il
  ritocco crea sempre una versione/ricetta nuova).

## Approccio scelto (A)
Ricette personali in un file JSON dell'utente, letto a ogni avvio insieme a
quelle incluse.

Scartati:
- **B – solo nel browser (localStorage):** niente CLI, si perdono cancellando i
  dati del sito.
- **In `data/extra_recipes.json`:** richiede di ricompilare per ogni ricetta.

## Design

### Archivio
- File: `~/.config/fxraw/my_recipes.json`, stesso formato di `recipes.json`
  (array di `Recipe`), `source: { "name": "Mine" }` più `author` = ricetta di
  partenza (es. "based on Essenza").
- Sotto `sudo` si usa la home di `SUDO_USER` e il file viene restituito
  all'utente (`chown_to_sudo_user`), come già per i JPEG salvati.
- `recipes::all()` restituisce incluse + personali. Le personali sono
  modificabili a runtime dal server (lock interno), quindi `all()` smette di
  essere `&'static [Recipe]`; i 6 chiamanti vanno adattati.
- File assente → nessuna ricetta personale; file illeggibile → avviso a
  console, si prosegue con le sole incluse.

### Server (`src/web.rs`)
- `POST /api/images/{id}/convert-custom` con corpo JSON = impostazioni ricetta.
  Il render è salvato come `renders/{id}/custom-{hash}.jpg`, dove l'hash deriva
  dalle impostazioni: stesso ritocco = cache riusata.
- `POST /api/my-recipes` (nome + impostazioni) → crea o sovrascrive una ricetta
  personale; risponde con lo slug. Il render già fatto per quelle impostazioni
  viene collegato al nuovo slug, così non va rifatto.
- `DELETE /api/my-recipes/{slug}` → elimina una ricetta personale.
- Slug derivato dal nome; se coincide con una ricetta inclusa → errore "nome già
  usato da una ricetta inclusa". Se coincide con una personale → l'interfaccia
  chiede conferma e sovrascrive.

### Interfaccia (`src/web/index.html`)
- Tabella del pannello ricetta modificabile:
  - menu: simulazione, DR (auto/100/200/400), grana (off/weak/strong) +
    dimensione (small/large), Color Chrome, FX Blue, WB (modalità + Kelvin
    2500–10000 quando "temperature");
  - pulsanti − / +: Highlight e Shadow −2…+4 a passi di 0.5, Color/Sharpness/
    High ISO NR −4…+4, Clarity −5…+5, WB shift R/B −9…+9.
- I valori cambiati sono evidenziati; compaiono "Converti modifiche ↵",
  "Salva come ricetta…" e "Annulla modifiche".
- Nella striscia la versione ritoccata appare come "<Ricetta> (modificata)".
- Le ricette personali hanno badge "Mie", un filtro dedicato e un pulsante
  "Elimina" nel pannello.

## Lasciato a chi implementa
- Layout esatto dei controlli nel pannello.
- Formato dell'hash e pulizia dei render ritoccati non salvati (la cache è
  già svuotata a ogni avvio).
