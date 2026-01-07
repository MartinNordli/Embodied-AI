# plan.md — Synsbasert robotlæring med demonstrasjoner + Deep RL (ManiSkill2)

## Prosjektidé og mål

Målet er å bygge et moderne “embodied AI”-prosjekt der en manipulasjonsagent lærer en oppgave i simulator fra kamerabilder. Du starter med demonstrasjoner (behavior cloning / imitation learning) for å få en fungerende policy raskt, og fin-tuner deretter med deep reinforcement learning (typisk actor–critic) for å øke suksessrate og robusthet. Prosjektet skal ende i en reproduserbar kodebase med tydelige eksperimenter, logging, videoer, og en rapport som forklarer valg, ablations og resultater.

For å holde prosjektet gjennomførbart og samtidig imponerende, bør du velge én “hovedoppgave” (som blir best) og én “generaliseringsvariant” (samme oppgave men med domain randomization / nye objekter), slik at du kan demonstrere både læring og robusthet uten å spre deg over for mange miljøer.

## Valg av simulator og “datasett” (gratis og godt egnet)

Anbefalt plattform er ManiSkill2, som er laget for robotmanipulasjon med både visuelt input og demonstrasjoner. I dette prosjektet er “datasettet” primært demonstrasjonstrajektorier (state, action, reward, done) og eventuelt ekstra synsdata (RGB/RGB-D) som du selv sampler fra simulatoren ved å kjøre demonstrationspolicyene eller laste ferdige trajectories dersom de følger med oppsettet du bruker.

Den praktiske fordelen er at du får gratis, ubegrenset syntetisk data og full kontroll på variasjoner (lys, teksturer, kameravinkel, objekter). Det betyr at du kan bygge en ekte ML-pipeline med datasamling, preprosessering, trening og evaluering, uten å være avhengig av manuelle labels.

Konkret oppgavevalg bør ha disse egenskapene: kontinuerlige handlinger, tett nok reward til at RL faktisk lærer innen rimelig tid, og en klar suksessmetrik. Pick-and-place-varianter er ofte ideelle for en første semesterleveranse, mens mer komplekse assembly-oppgaver kan brukes som “stretch” dersom du får stabil læring tidlig.

## Hardware: hva du trenger og hva som er “nice to have”

Prosjektet kan kjøres på CPU for debugging og små tester, men du vil i praksis trenge en NVIDIA-GPU for effektiv trening fra piksler. Det viktigste er VRAM, fordi visuelle encodere og store batcher i RL fort fyller minne. Under er et realistisk hardware-bilde som fungerer i praksis; du kan jobbe på en “ok”-maskin og fortsatt få et CV-sterkt resultat, men raskere GPU gir flere eksperimenter og penere ablations.

| Tier | GPU (VRAM) | CPU / RAM | Når det holder | Kommentar |
|---|---:|---:|---|---|
| Minimum for seriøs trening | 8–12 GB | 8+ kjerner, 32 GB RAM | Én oppgave, moderate bildeoppløsninger | Krever stram logikk for batch/parallelle envs |
| Komfortabelt | 16–24 GB | 12+ kjerner, 64 GB RAM | Flere seeds, mer randomization | Gir rom for pretrente backbones og raskere iterasjon |
| “Research-lignende” | 24–48+ GB | 16+ kjerner, 64–128 GB | Mange ablations / mer kompleks policy | Særlig nyttig om du fryser/fin-tuner store ViT-backbones |

Lagring: sett av minst 50–100 GB for checkpoints, videoer og loggfiler. Hvis du bruker W&B for logging, kan du lagre store artefakter lokalt og bare synk’e nøkkelfiler.

## Programvarestakk og biblioteker

Språk og runtime bør være Python 3.10 eller 3.11 (velg én og lås den). Bruk Conda eller uv/poetry for reproducibility. Målet er at en arbeidsgiver skal kunne klone repoet, opprette miljøet og reprodusere en baseline på én kommando.

Kjernen i stakken bør bestå av PyTorch (GPU-trening), ManiSkill2 (simulator + oppgaver + demonstrasjoner), og et RL-rammeverk som gir stabil implementasjon av SAC/PPO og enkel logging. Stable-Baselines3 er et godt valg for rask progresjon, særlig fordi det støtter CNN-baserte policies og gjør det enkelt å bytte ut feature extractor med en egendefinert PyTorch-modul. For imitation learning kan du bruke “imitation”-biblioteket (behavior cloning, GAIL osv.) eller implementere en enkel BC-loop selv for full kontroll.

For prosjektlimet bør du legge til: Gymnasium-wrappers for observasjoner og action spaces, Hydra eller tilsvarende for konfigurasjoner, Weights & Biases (eller TensorBoard) for tracking, og et bibliotek for videoeksport. Hvis du ønsker mer “research”-følelse, kan du i tillegg bruke PyTorch Lightning for struktur, men for RL kan det være enklere å holde treningsloopen eksplisitt.

En typisk miljøfil (requirements/conda) bør inneholde: torch, torchvision, mani_skill2 (+ avhengigheter), gymnasium, stable-baselines3, imitation (valgfritt), hydra-core, wandb, numpy, opencv-python, imageio, moviepy, tqdm, rich, pandas (for resultattabeller), og einops (nyttig for ViT/attention). Hvis du vil bruke en pretrenet ViT-backbone som DINOv2, legg til en enkel måte å hente modellen på (torch.hub eller en definert dependency) og lås versjon.

## Overordnet ML-pipeline (end-to-end)

Systemet ditt består av fem tydelige deler som må “limes” sammen: simulatoren som genererer data, et observasjons- og datasamlingslag, en visuell representasjonsmodell (encoder), en policy/verdimodell (RL), og et evaluerings- og logginglag. Når du designer dette som moduler, blir det lett å bytte ut komponenter og gjøre ablations.

I dataflyten kjører du simulatoren i parallelle instanser, samler (obs, action, reward, done, info) og lagrer enten i en replay buffer (for off-policy som SAC) eller i on-policy batches (for PPO). Observasjonene er primært RGB (eventuelt RGB-D). Du normaliserer og resizer bildet tidlig, og bruker en “frame stack” dersom oppgaven krever mer temporal info.

Encoder → policy-koblingen bør være eksplisitt: en FeatureExtractor tar inn bilde og produserer en latent vektor z. Policy-nettverket tar z og produserer handlinger, mens critic tar (z, a). Ved å gjøre dette modulært kan du starte med en enkel CNN, og senere bytte til en DINOv2-initialisert encoder, eventuelt med freezing i starten.

Demonstrasjoner går i en egen vei: du samler trajectories med en demonstrationspolicy eller laster ned trajectories. Deretter trener du en behavior cloning policy ved supervisert læring på (obs → action). Denne policyen brukes som initialisering for RL-fasen, enten ved å laste weights direkte inn i actor-nettet, eller ved å bruke demonstrasjoner i replay buffer i starten (enkel “warm start”). Dette alene gjør prosjektet merkbart mer moderne enn “RL from scratch”.

## Modellvalg som passer både CV og Deep RL

For RL-algoritme er SAC ofte et godt valg når du har kontinuerlige handlinger, fordi den er sample-efficient og robust med riktig tuning. PPO er også et bra baselinevalg, særlig hvis du vil ha en stabil on-policy referanse. Planen under antar SAC som hovedløp og PPO som kontrollbaseline dersom du har tid.

For den visuelle delen bør du ha en trapp av modeller:

Første baseline er en lett CNN-encoder (3–4 conv-lag + MLP) som trener end-to-end sammen med policyen. Denne baseline gir deg et referansepunkt og lar deg verifisere hele systemet.

Andre steg er “pretrain-init”: bytt ut CNN med en pretrenet backbone. Et praktisk valg er en ViT-lignende encoder (for eksempel DINOv2-init) der du først fryser backbone og bare trener et lite “adapter head” inn i policyen. Når RL er stabil, kan du delvis unfreeze med lav læringsrate for å få bedre task-spesifikk representasjon.

Tredje steg er robusthet og generalisering: legg til domain randomization i simulatoren (lys, tekstur, posisjon, kamerastøy). Hvis agenten faller sammen, kan du øke data-augmentations på bildene (random crop, color jitter, blur) som en “DrQ-lignende” regularisering uten å måtte implementere en helt ny algoritme. Dette er ekstremt relevant for jobbmarkedet fordi det viser at du forstår distribusjonsskift og hvordan man gjør visuelle policies robuste.

Hvis du ønsker en ekstra CV-komponent fra “object detection / segmentation”, kan du inkludere et mellomledd der du bruker segmenteringsmasker som ekstra kanaler til policyen, eller du bruker en maske for å fokusere på relevant objekt. Dette må ikke være perfekt; poenget er å vise at du kan bygge og evaluere en representasjon som gir bedre læring eller bedre generalisering.

## Hvordan treningen bør utføres (stabilt, reproduserbart, og målbart)

Kjør alltid med en streng eksperimentprotokoll: faste seeds, tydelige “train/eval”-modus, og eksplisitte eval-episoder uten exploration. Du bør logge suksessrate og gjennomsnittlig retur over tid, men suksessrate er ofte den beste “forretningsmetrikken” i manipulasjon. Du bør i tillegg logge episode-lengde, actor/critic loss, entropy/alpha (for SAC), Q-verdier, og eventuelle representasjonsmål som feature-normer.

Parallelle miljøer er viktig. For off-policy SAC kan du samle data med flere envs for å øke throughput, men vær obs på at simulatorens rendering kan være flaskehals. Start med lav oppløsning (for eksempel 84×84 eller 128×128) og øk etter at alt fungerer. En god praksis er å kjøre hyppige, korte evalueringsrunder og sjeldnere lange evalueringsrunder som lagrer video.

Checkpointing bør skje på to kriterier: “beste suksessrate” og “siste checkpoint”. På den måten kan du alltid reprodusere en demonstrasjon selv om treningen senere degraderer.

For tracking er Weights & Biases svært effektivt i praksis. Det viktige er at du logger configs og commit-hash sammen med run. Hvis du ikke vil bruke W&B, kan du få mye av samme verdi med TensorBoard + en egen results.csv per run, men W&B gir bedre sammenligning og artefakthåndtering.

## Hvordan prosjektet “limer ting sammen” i kode (arkitektur og repo)

Repoet bør bygges slik at simulator, observasjonswrappers, modeller og trening er adskilt. En god tommelfingerregel er at alt som har med miljøet å gjøre ikke skal vite noe om RL-algoritmen, og at RL-trening kun ser “gymnasium”-grensesnittet.

En praktisk struktur er at du har en env-modul som bygger ManiSkill2-oppgaven med riktig kameraoppsett og wrapper observasjoner til et konsistent format. Deretter har du en models-modul som definerer FeatureExtractor (CNN/DINOv2) og policy-nettverk. Til slutt har du training-scripts som kjører BC og RL.

Hvis du bruker Stable-Baselines3, implementerer du en custom feature extractor-klasse som tar imot bilder og returnerer en latent vektor. Det er her du kan plugge inn en pretrenet backbone, kontrollere freezing, og gjøre augmentations. BC-fasen kan enten bruke “imitation”-biblioteket, eller du lager en ren PyTorch-loop som trener actor til å minimere MSE mellom predikert action og demonstrert action.

## Komplett “map structure” for repoet (anbefalt prosjektstruktur)

Målet med denne strukturen er: (1) reproduserbare eksperimenter, (2) klar separasjon mellom env/obs, modeller og trening, (3) enkel logging + video, (4) lett å gjøre ablations (CNN vs DINOv2, BC-init vs scratch, randomization on/off).

**Top-level (repo root)**

```
embodied-ai/
  README.md
  plan.md
  pyproject.toml              # (anbefalt) dependencies + tooling (ruff/black/mypy) + package metadata
  uv.lock / poetry.lock       # lock-fil (velg én løsning)
  .python-version             # pin Python (f.eks. 3.11.x)
  .gitignore
  Makefile                    # valgfritt: short-hands (install, lint, train, eval)
  Dockerfile                  # valgfritt men bra for “repro recipe”

  configs/
    default.yaml              # base config (seed, device, env, algo, logging)
    env/
      maniskill_pickplace.yaml
      maniskill_stack.yaml
    obs/
      rgb_84.yaml
      rgb_128.yaml
      rgbd_128.yaml
    algo/
      sac_pixels.yaml
      ppo_pixels.yaml
      bc.yaml
    encoder/
      cnn_small.yaml
      dinov2_frozen.yaml
      dinov2_unfreeze.yaml
    augmentation/
      none.yaml
      drq_light.yaml
    domain_randomization/
      off.yaml
      light.yaml
      heavy.yaml

  src/
    embodied_ai/
      __init__.py

      cli/
        __init__.py
        train_bc.py            # entrypoint: train BC on demos → saves policy
        train_rl.py            # entrypoint: train RL (SAC/PPO) from pixels
        finetune_bc_to_rl.py   # entrypoint: load BC actor → continue RL
        eval_policy.py         # eval episodes + metrics + video export
        collect_demos.py       # collect trajectories from scripted/demo policy

      envs/
        __init__.py
        make_env.py            # single source of truth: build ManiSkill2 env + wrappers
        tasks.py               # task registry (task_id → ManiSkill env spec)
        cameras.py             # camera setup (names, resolutions, intrinsics if needed)
        wrappers/
          __init__.py
          obs_wrappers.py      # RGB/RGB-D formatting, resize, normalize, frame-stack
          success_wrappers.py  # unify success metric extraction into info["is_success"]
          record_wrappers.py   # video recorder wrapper (train/eval)
          time_limit.py        # consistent episode length (if needed)

      data/
        __init__.py
        demos_dataset.py       # loads trajectories → (image, action, done, etc.)
        replay_export.py       # (valgfritt) export RL replay segments for analysis
        storage.py             # paths + naming conventions for datasets/artifacts

      models/
        __init__.py
        encoders/
          __init__.py
          cnn.py               # small CNN baseline
          dinov2.py            # DINOv2 feature extractor + freeze/unfreeze knobs
          common.py            # utilities: preprocess, flatten, feature dims
        sb3/
          __init__.py
          feature_extractor.py # SB3 BaseFeaturesExtractor wiring for image encoders
          policy_kwargs.py     # helper to build consistent SB3 policy configs
        bc/
          __init__.py
          actor.py             # simple actor head for BC (latent → action)

      algorithms/
        __init__.py
        bc_trainer.py          # pure PyTorch BC loop (loss, eval in env, checkpoints)
        rl_sb3.py              # SB3 training harness (SAC/PPO) + callbacks
        callbacks.py           # eval callback, checkpoint callback, video callback, metrics

      logging/
        __init__.py
        wandb_logger.py        # W&B init + config logging (optional)
        tb_logger.py           # TensorBoard fallback
        video.py               # write mp4/gif, frame utilities
        metrics.py             # success-rate, return, episode length aggregations

      utils/
        __init__.py
        seed.py                # full seeding (python/numpy/torch/env)
        device.py              # cpu/cuda selection + determinism flags
        config.py              # Hydra/OmegaConf helpers (if using Hydra)
        serialization.py       # save/load (torch, npz), model versioning
        timers.py              # throughput profiling (fps, env step time)

  scripts/
    install.sh                # optional: environment bootstrap
    download_assets.sh        # optional: extra assets/models (if any)
    run_sweep.sh              # optional: launch multiple seeds / ablations

  tests/
    test_make_env.py          # smoke test: env builds + step works + obs shapes
    test_demos_dataset.py     # smoke test: demo loading + batching works

  docs/
    report.md                 # paper-lignende rapport (evt. senere flyttet til PDF)
    experiments.md            # “what we ran” + key plots + takeaways
    troubleshooting.md        # install + common sim issues + GPU/render tips

  assets/
    videos/                   # small, checked-in gifs (keep light)
    figures/                  # plots used in README/report

  data_local/                 # NOT committed (raw demos, caches, etc.)
    demos/
      <task_name>/
        trajectories.h5 / .npz / .pkl
        metadata.json
    pretrained/
      dinov2/                 # cached weights (if you don’t want to redownload)

  runs/                       # NOT committed (or only partial)
    <exp_name>/
      config.yaml
      checkpoints/
      videos/
      tb/ or wandb/
      metrics.csv
```

**Navnekonvensjoner (anbefalt)**
- `configs/`: alt som påvirker eksperimentet (seed, env, obs, encoder, algo, logging) slik at hver run kan reproduseres fra config alene.
- `src/embodied_ai/envs/make_env.py`: eneste sted som “vet” hvordan ManiSkill2 skrus sammen (task, camera, wrappers).
- `src/embodied_ai/algorithms/`: trening-lag (BC loop og SB3-harness) uten miljø-spesifikke hacks.
- `data_local/` og `runs/`: holdes utenfor git (store artefakter), men hver run lagrer sin config + checkpoints + eval-video.

## Robust plan for gjennomføring (faser og milepæler)

### Fase 0: Miljø og sanity-check (1–3 dager)

Før du gjør noe visuelt, verifiser at du kan installere ManiSkill2 og kjøre en enkel rollout, og at du kan rendere og lagre en video. Deretter verifiserer du at du kan lese ut suksessmetrikken fra info-dict eller definere den selv. På dette tidspunktet bør du også kjøre en “privileged state”-baseline: bruk simulatorens tilstandsvariabler som observasjon og lær en policy med SAC/PPO. Dette er gull fordi det avdekker reward/action-space-problemer før du legger på visjon.

Leveransen fra denne fasen er at du har et fungerende treningsscript, en enkel læringskurve, og at du kan reprodusere det på nytt med samme seed.

### Fase 1: Pixel-baseline med enkel CNN (1–2 uker)

Bytt observasjon til RGB (evt. RGB-D). Tren SAC med en enkel CNN-encoder. Bruk lav oppløsning og få en tydelig forbedring i suksessrate over tid. Når du får en agent som “noen ganger” lykkes, har du bevist at limingen fungerer.

Leveransen er en baseline som kan trenes på nytt, og en eval-video som viser agenten løse oppgaven minst av og til.

### Fase 2: Demonstrasjoner og behavior cloning (1 uke)

Samle eller last demonstrasjoner, og tren en BC-policy på samme observasjoner som RL-policyen bruker. Du må måle BC-suksessrate i simulator, ikke bare treningsloss. BC vil ofte gi deg en policy som er “nesten der”, men feiler på små ting. Det er forventet og en perfekt inngang til RL fin-tuning.

Leveransen er en BC-policy som løser oppgaven med en tydelig suksessrate og som kan brukes som init for RL.

### Fase 3: BC → RL fin-tuning (1–2 uker)

Initialiser SAC-actor med BC-weights, og fin-tun med RL. Her skal du se betydelig løft i suksessrate og robusthet. Dette er kjernen i prosjektet, og i rapporten bør du vise kurver som sammenligner RL-from-scratch mot BC-init.

Leveransen er din “beste agent” med høy suksessrate, video, og tydelige eksperimentlogger.

### Fase 4: Moderne representasjoner og generalisering (1–2 uker)

Bytt encoderen til en pretrenet backbone (for eksempel DINOv2-init) med freezing i starten, og mål sample-efficiency. Deretter legger du inn domain randomization og augmentations og måler generalisering. Denne fasen er det som gjør prosjektet “moderne” og jobbrelevant, fordi du viser at du kan tenke utover en enkel benchmark-run.

Leveransen er ablations som viser at pretrain-init og/eller augmentations gir bedre læring og bedre generalisering.

### Fase 5: Polering og “CV-leveranse” (1 uke)

Skriv en kort, paper-lignende rapport: problem, metode, eksperimenter, resultater, og begrensninger. Lag en enkel demo-side i README med GIF/video, kommandoer for å trene og evaluere, og en “repro recipe” med eksakte versjoner. Hvis du har tid, legg til en liten inference-demo som kjører en ferdig policy og lagrer video på én kommando.

Leveransen er en portefølje-klar repo.

## Treningstips som vanligvis avgjør om dette lykkes

Stabilitet i RL fra piksler handler ofte om tre ting: datastrøm, normalisering og regularisering. Pass på at du ikke gjør unødvendig tung rendering i train-loop. Sørg for at bilde-input er konsekvent normalisert og at du ikke blander BGR/RGB. Bruk augmentations forsiktig og mål effekten. Hvis du ser policy-kollaps, reduser læringsrate, øk replay buffer, og vurder å fryse encoderen i starten.

For generalisering er domain randomization bedre enn å “jage hyperparametre”. Hvis agenten bare funker på én tekstur eller én kameravinkel, er den ikke særlig imponerende. Små variasjoner i simulatoren som du kontrollerer gir stor verdi i rapporten.

## Tracking og rapportering: hva som bør være i loggene

For hver run bør du ha config (algoritme, lr, batch size, image size, seeds, randomization-nivå), suksessrate over tid, return over tid, eval-videoer per N steps, og checkpoint-artefakter. I rapporten bør du kunne peke på minst tre sammenligninger: RL-from-scratch vs BC-init, CNN vs pretrain-init, og uten vs med domain randomization/augmentations.

## Risiki og hvordan du håndterer dem

Det vanligste problemet er at visuell RL tar lengre tid enn du tror. Mottiltaket er å låse scope tidlig: én oppgave først, én algoritme først, én encoder først. Deretter moderniserer du gradvis. Et annet problem er at installasjon og simulatoravhengigheter kan spise tid. Mottiltaket er å containerize (Docker) eller i det minste låse miljøet i conda/poetry og dokumentere det tidlig i fase 0.

Til slutt: hvis du merker at manipulasjonsoppgaven du valgte har ekstremt spars reward og du ikke får læring i det hele tatt, bytt oppgave tidlig. Prosjektet måles på leveransen, ikke på hvor “hard” oppgaven var.

## Konkret “default stack” jeg anbefaler å starte med

Start med ManiSkill2 + PyTorch + Stable-Baselines3 (SAC) + en custom CNN feature extractor. Legg til W&B for logging og Hydra for configs. Når pipeline fungerer, legg inn BC med en enkel PyTorch-loop. Deretter, når alt er stabilt, bytt encoderen til en pretrenet backbone og evaluer. Denne rekkefølgen maksimerer sjansen for at du ender med et imponerende sluttprodukt innen et semester.

