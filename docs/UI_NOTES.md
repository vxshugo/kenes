# Kenes UI and Claude layer: notes

What lives in `app/src` (everything in `app/` except `app/src-tauri/`), how the Claude calls are
built, and what the UI expects from the Rust side. The interface itself is `docs/CONTRACT.md`.

## Running it

```bash
cd app
pnpm install
pnpm dev      # http://localhost:1420, runs against the scripted mock in a normal browser
pnpm test     # vitest, no network
pnpm build    # tsc + vite build
```

In a plain browser the UI uses the mock backend automatically. It replays a scripted 10-person
project sync (mostly Russian, some Kazakh) with speaker labels for the chosen meeting format, and
keeps settings, meetings, notes, speaker names, the voiceprint and the API key in `localStorage`.
The key can also come from `VITE_ANTHROPIC_API_KEY` in `app/.env.local`. Add `?mockSpeed=3` to the
URL to replay faster; the silence wait and debounces stay wall-clock, so some auto hints only show
up at speed 1. The mock's default settings (and only those) set `myNames: ["Хуго"]`, the name the
script addresses the user by. A running mock meeting survives a page reload like a real one survives a
webview reload: its state is in `localStorage` (`kenes.mock.live`, with a 1 s heartbeat), and the next page
reattaches and continues the script. A record without a heartbeat for 30 s belongs to a closed tab; that
meeting is ended instead.

## File map

```
app/src/
  main.tsx, App.tsx, App.css     shell: header, tabs, global shortcuts, toasts; all styling (light + dark)
  types.ts                       contract types (Segment, PipelineEvent, Settings, Speaker, ...) + defaults/normalizer/migration
  backend/
    types.ts                     Backend interface: every command + onEvent()
    tauri.ts                     invoke()/listen("kenes://event", "kenes://hotkey"…); tolerates a get_meeting without `speakers`
    mock.ts, mockScript.ts       browser stand-in: 10-person scripted meeting, labels per format (MockDiarizer),
                                 relabel on stop, rename_speaker, enroll_voice (20 s, stored in localStorage),
                                 a running meeting that survives reloads (session_status), simulateHotkey()
    index.ts                     getBackend(): Tauri if isTauri() / __TAURI_INTERNALS__, else mock
  llm/                           framework-free, unit-tested
    client.ts                    SDK client factory, model capabilities, request params (+ structured output), streaming, Russian errors
    prompts.ts                   frozen system prompt, context block (format, user's names), transcript formatting, rename notes
    speakers.ts                  labels → default names, SpeakerDirectory (custom names), colors, talk-time stats, relabel
    conversation.ts              per-meeting append-only message log with cache breakpoints and rename notes (+ fork, final request,
                                 rollover to a new log seeded with the summary when the context fills up)
    budget.ts                    context windows per model, rollover thresholds, request size estimates
    usage.ts                     prices per model, per-meeting token/cache/cost totals
    tasks.ts                     task texts (hint with SKIP, ask, explain, translate, recap, rolling, final), names map, budgets
    triggers.ts                  question detector (ru/kk), name matching with case endings, auto-hint modes, silence watch,
                                 rolling-summary and suggestion timers
    suggestions.ts               speaker-name suggestions: side request, JSON schema, validation
    copilot.ts                   runs tasks against a Conversation: serialized live queue, fork, final, side requests;
                                 reports usage, asks for a fresh summary near the limit, rolls over past it
    *.test.ts                    vitest
  session/
    controller.ts                SessionController: backend events, transcript, speakers, triggers, cards, summaries, notes,
                                 reattach after a reload, usage totals, global-shortcut actions
    restore.ts                   pure helpers for reattach: notes → cards, phase, mic mode, usage in localStorage
    controller.test.ts           end-to-end with the mock backend and a fake Claude client
    useController.ts             singleton + useSyncExternalStore hook
  components/
    Header.tsx                   title, Start/Stop/New, status + model-load progress, elapsed timer, level meters
    LevelMeter.tsx, Elapsed.tsx
    PreStart.tsx                 meeting format (online / room / hybrid), title, meeting context -> save_settings -> start_session
    LiveTab.tsx                  hint cards, participants panel, transcript, action bar, ask input
    Speakers.tsx                 SpeakerName (inline rename), ParticipantsList, SuggestionChips
    HintCard.tsx, Transcript.tsx live partials replaced by id, speaker headers in colors, auto-scroll + "К живому"
    SummaryTab.tsx               rolling + final summary, copy Markdown, download .md, regenerate, «Расход Claude»
    HistoryTab.tsx               meeting view with participants (renamable) and named transcript, export, delete
    SettingsTab.tsx              API key, model, efforts, my names, auto-hint mode, rolling interval, language, profile,
                                 STT, capture + mic mode + «Подавление эха», «Мой голос» (voiceprint)
    HotkeysSection.tsx           «Окно и горячие клавиши»: GNOME always-on-top, global shortcuts with a key recorder
    Markdown.tsx, Icons.tsx
  lib/
    meetingFormat.ts             the three meeting formats ↔ capture settings
    markdown.ts                  small Markdown subset parser (no HTML passthrough, so no sanitizing needed)
    clipboard.ts, format.ts
    keys.ts                      in-app shortcut labels; accelerator recording and display for global shortcuts
  test/helpers.ts                in-memory localStorage, fake streaming Claude client, waitFor
```

Keyboard shortcuts, also shown in tooltips (Mod = Ctrl on Linux, ⌘ on macOS):
Mod+Enter "Что ответить?", Mod+K focus "Спросить…", Mod+Shift+E explain the selected text,
Mod+Shift+U translate, Mod+Shift+K "Кратко: 5 мин", Alt+1…4 switch tabs. These work while the Kenes
window has focus. System-wide shortcuts (see «Window and global shortcuts» below) add «Что ответить?»,
«Кратко: 5 мин» and show/hide while the call has focus; defaults Mod+Alt+Enter, Mod+Alt+K, Mod+Alt+P.

## Claude layer

### Request shape

All calls go through `client.beta.messages.stream()` from `@anthropic-ai/sdk` (0.128), with
`dangerouslyAllowBrowser: true` because this is a local desktop app using the user's own key.

- `model`: `Settings.claudeModel` (default `claude-opus-5`).
- `thinking: {type: "adaptive"}` and `output_config: {effort}`. Live tasks use `hintEffort`
  (default `low`); the final summary uses `summaryEffort` (default `high`). Thinking is never
  disabled.
- `fallbacks: "default"` with the beta header `server-side-fallback-2026-07-01`, only for models
  that offer it (Claude Opus 5 / 5.5, Fable 5.1+). If the API answers 400 before any text arrives,
  the request is retried once without it and the model is remembered for the session.
- `modelCaps()` (in `client.ts`) removes what a model doesn't support. Haiku 4.5 gets no thinking or
  effort; `xhigh` is lowered to `high` on models without it.
- `max_tokens` covers thinking plus text: 8K/12K/16K for live tasks at low/medium/high, 12K for the
  rolling summary, 32K for the final summary (64K at `xhigh`).
- Streaming: text deltas go to the card as they arrive. Once the stream finishes, the final message
  is checked before its content is used:
  - `refusal` throws `RefusalError`. The partial text is discarded and the turn is rolled back.
  - `max_tokens` and `model_context_window_exceeded` keep the text but mark it as cut off.
  - A server-side fallback is detected from `usage.iterations`; the card then shows which model answered.
- Errors are the SDK's typed classes, turned into Russian messages in `describeError()`: 401, 403,
  404 (model), 429, 400, 529/5xx, connection, timeout, missing key.
- Timeouts and retries depend on the route: 90 s and 1 retry for live tasks, 180 s and 2 retries
  for the rolling summary, 600 s and 2 retries for the final summary, 60 s and 1 retry for name
  suggestions.
- Structured output (name suggestions only): `output_config: {effort: "low", format: {type:
  "json_schema", schema}}`, no `fallbacks`. `modelCaps().structuredOutputs` gates it (Opus ≥ 4.1,
  Sonnet ≥ 4.5, Haiku ≥ 4.5, Fable/Mythos, unknown current ids). If a model answers 400 before any
  text, the request is retried once without `format` and the model is remembered; the reply is then
  parsed as JSON out of plain text.

### Caching layout (`conversation.ts`)

```
system:    [SYSTEM_PROMPT ◆]                          frozen: identical bytes for every meeting and request
user:      [meeting context ◆, update₁, task₁]         context: title, date, format, user's names, profile, agenda (fixed per meeting)
assistant: [answer₁]                                  text only
user:      [speaker_names?, update₂, task₂ ◆]          ◆ = cache_control {type: "ephemeral"}; the last ◆ moves each turn
...
```

- The log is append-only. Every request re-sends the previous request's bytes unchanged and adds a
  turn, so each call reads the cached prefix and writes only the new tail. Tests check this for a
  sequence of requests, with `cache_control` markers stripped before comparing.
- There are at most 3 breakpoints: the system prompt, the context block, and the newest user block.
- `update` is `<transcript_update>` with the final segments added since the last turn, formatted
  deterministically as `[mm:ss] Имя: text` with the speaker's display name at the moment the line
  is sent (`[12:03] Айдос: …`, `[12:04] Участник 3: …`, `[12:05] Я: …`), ordered by (startMs, id).
  Nothing volatile goes before a breakpoint. The system prompt contains no dates or settings. The
  answer language, the names map, in-progress partials and the question that fired the hint all go
  in the task, at the tail.
- **Renames never rewrite earlier turns.** The log remembers which name it last used for each label.
  When a name changed since, the next user turn starts with
  `<speaker_names>\nУчастник 3 теперь зовут Айдос.\n</speaker_names>` (a reset reads
  «Айдос — снова Участник 3 (имя снято).»), and its new lines use the new name. Labels whose lines
  were never sent get no note; renaming back and forth before the next turn produces none. Every task
  that reads the transcript also repeats the current map («Имена участников: Участник 3 — Айдос; …»).
  A rollback restores the log's name state, so the note rides along with the retried turn. A fork
  (rolling summary) includes pending notes without consuming them. The final summary re-renders the
  whole transcript with current names, so it needs no notes. Tests check the prefix property across
  renames.
- `speakersRelabeled` updates the conversation's finals: lines not yet sent and the final summary use
  the new labels; lines already in the log keep what they said.
- A failed request (error, refusal, abort, empty reply) is the only edit allowed: its trailing user
  turn is dropped, and its transcript lines go back to the queue to be sent with the next turn.
- Assistant turns are stored as plain text. Thinking blocks are not replayed, which also avoids the
  rules for echoing fallback turns back.
- Live tasks (hint, ask, explain, translate, recap) go through a FIFO queue, because an append-only
  log can't take overlapping turns.
- The rolling summary is a **fork**. It reuses the committed log and adds a one-off turn, but is
  never appended. Its breakpoint sits on the last shared block (the last assistant answer), not on
  the one-off tail, so it reads the live cache and keeps it warm (each read resets the 5-minute
  TTL). It uses `hintEffort`, the same effort as the live turns, because changing effort
  invalidates the messages cache.
- The final summary is a separate request: the same frozen system prompt, then
  `[context, <full_transcript> ◆, task]` at `summaryEffort`. The breakpoint after the transcript
  means "Сгенерировать заново" reads it back.
- Changing the model or effort mid-meeting is allowed, but it invalidates the cache (caches are per
  model).

### Long meetings: rollover (`budget.ts`, `Conversation.rollover`)

The live log only grows. Opus 5 has 1M tokens of context, which a meeting won't fill, but Claude Haiku
4.5 (200K) and older models could after several hours. Before every live turn and every rolling-summary
fork the copilot estimates the request (prompt + `max_tokens`):
- The estimate uses the meeting's own tokens-per-character ratio, measured from the last response's
  `usage` (`input + cache_read + cache_creation` over the request's characters); 0.5 before the first
  response.
- Past 55% of the model's window it asks for a fresh rolling summary (at most once a minute).
- Past 70% it rolls over. A new log starts with the same frozen system prompt and context block (their
  cache still hits), then an `<earlier_in_meeting>` block ◆ with the latest rolling summary, the newest
  transcript lines that fit 20% of the window (rendered with current names), and the names map. Every
  final is in that tail or older than it, so nothing stays pending. The name state restarts from the
  current names. It never happens while a live turn waits for its reply.
- The log is append-only within each epoch; tests check the prefix property on both sides of a rollover,
  and the rollover itself (through a fake API that reports large `usage`).
- The final summary gets the same treatment: if the full transcript would pass 70% of the window, it is
  replaced by the summary plus the newest lines.
- A short info toast says when it happened; «Расход Claude» counts rollovers.

Server-side compaction (`compact-2026-01-12`) was considered: the docs offer it for Opus/Sonnet 4.6+ and
Fable, not for Haiku 4.5, which is the model this guards against. It also needs the full response content
(compaction blocks) replayed, while this log stores assistant turns as text, and it would summarize on the
server's terms instead of reusing the rolling summary and the names map.

### Usage and cost (`usage.ts`)

Every response's `usage` (live, fork, final, name suggestions) adds up per meeting: uncached input,
cache writes, cache reads, output (thinking included), requests, the largest live prompt, rollovers.
The Summary tab shows «Расход Claude»: requests, all input tokens, the share read from the cache
(`cache_read_input_tokens` over all input), output and ≈ cost, with a detail line. The cost uses the
price table in the Claude API docs for the model that answered (a server-side fallback may differ):
input, 1.25× input for 5-minute cache writes, the cache-read price, output. Models without a known price
are counted and marked with «+». Totals are kept per meeting in `localStorage` (`kenes.usage.<id>`), so a
reload doesn't reset them; they are not stored in the meeting database.

On the first real API run this is where caching shows up: the share from the cache should grow turn over
turn (writes should be roughly the last turn's size).

### Triggers (`triggers.ts`)

- Question detection works on raw ASR output, with no punctuation to go on. In order:
  - question phrases: «можно ли», «как вы думаете», «а у вас», «или нет», «сіз ше», «қалай ойлайсыз», …
  - request verbs: расскажите, объясните, подскажите, айтыңызшы, түсіндіріңізші, …
  - the Russian particle «ли», except «вряд ли» and «то ли»
  - Kazakh particles ма/ме/ба/бе/па/пе as separate tokens, plus glued forms such as «білесізбе» or «барма»
  - Kazakh wh-words anywhere (қашан, қалай, неге, кім, қанша, қандай, …). «не» counts only in a
    Kazakh-looking utterance, because in Russian it means "not".
  - unambiguous Russian wh-words anywhere (почему, сколько, какой…)
  - ambiguous ones (как, что, когда, где, кто) only at the start of the utterance, after fillers
  - blockers for conjunction uses: «так как», «потому что», «что-то», «как раз», «вот почему», …
  - Kazakh greeting-questions («сәлеметсіздер ме») are stripped first.
- **Never on the user's own speech**: label "me", or a label the user named with one of `myNames`
  (in a room without a voiceprint the user can accept a suggestion «Зал 2 — Хуго?»).
- `autoHintMode` (Settings):
  - `"any"`: a lexical question of ≥ 3 words from anyone else fires, ≥ 20 s after the previous auto
    hint, if no live task is queued or streaming. The task has no SKIP option.
  - `"addressed"`, condition (a) "named": the segment, or the segment right before it if the same
    speaker said it (≤ 10 s gap, nobody else in between: one utterance the recognizer split), contains
    one of `myNames`, **and** the segment is a lexical question or the name is in calling position
    (first two words after fillers, or the last word). ≥ 2 words, 8 s debounce. Fires at once.
    «итак … хуго и нурлан считают» (a third-person mention) does not fire.
  - `"addressed"`, condition (b) "silence": a lexical question (≥ 3 words, 20 s debounce) without the
    name is held in a `SilenceWatch`. Any new speech (a partial or a final from anyone) cancels it. If
    2.5 s pass in silence, a hint fires whose task says the user wasn't named and asks Claude to decide.
  - `"off"`: only the button.
- Name matching (`findName`): case-insensitive on raw ASR tokens; Kazakh-only letters are folded
  (і→и, қ→к, ғ→г, ә→а, ө→о, ұ/ү→у, ң→н, һ→х), so «Айгерим» in settings matches «айгерімге». A token
  matches a name word exactly or with a case ending: Kazakh dative/genitive/accusative/locative/
  ablative/instrumental/equative/plural and -жан (хугоға, айдосқа, хугоның, хугомен, ерланжан …),
  Russian consonant-name endings (айдоса, ерлану, ерланом), -а/-я names (Дана → дане, Мария → марии)
  and -й/-ь names (Андрей → андрея). Names shorter than 3 letters match only exactly. Latin names also
  match a naive Cyrillic spelling (Hugo → хуго, Aidos → айдос). Multi-word entries match as a phrase.
- **SKIP**: in `"addressed"` mode both (a) and (b) tasks say «Если вопрос или реплика адресованы не
  мне, а другому участнику, ответь ровно SKIP». The card is created hidden and stays hidden while
  the streamed text could still be SKIP (`mayBeSkip`); a finished reply whose first word is SKIP
  (`isSkipReply`, tolerating markdown/quotes and a short trailing explanation) removes the card,
  saves no note, and restores the debounce timestamp. The assistant turn "SKIP" is committed to the
  log (it keeps the cache prefix; the next request reads it). Errors of hidden cards are shown.
- The rolling summary runs every `rollingSummaryMinutes` (0 = off), and only if new final segments
  arrived since the last run. The timer is checked every 10 s.

### Speaker-name suggestions (`suggestions.ts`)

- When: on the 10 s tick, at most once per `rollingSummaryMinutes` (4 min if the rolling summary is
  off), only while the session runs, some labels that spoke have no name and no open suggestion, and
  ≥ 4 new lines arrived since the last run. Also on demand (wand button in the participants panel,
  also after Stop).
- The request never touches the meeting conversation or its cache: its own small frozen system
  prompt (no `cache_control`), one user message with `<unnamed_labels>`, `<named_labels>`, the
  user's names, earlier rejections and the last ≤ 80 lines (≤ 9 000 chars) rendered as
  `[mm:ss] sys:3 (Участник 3): text`. `effort: "low"`, `max_tokens` 4 000, JSON schema
  `{suggestions: [{label, name, evidence, confidence}]}` with `additionalProperties: false` (no
  numeric/length constraints, which structured outputs don't support; they are checked client-side).
- The prompt accepts only evidence from the dialogue: addressed by name and this label answers next,
  a self-introduction, «спасибо, Айдос» right after the label spoke. Not job titles or the agenda.
- Validation: label must be one of the unnamed labels, a plausible name (letters, ≤ 40 chars, not
  the default name), confidence clamped to 0..1 and ≥ 0.6, not a rejected pair; best one per label.
- UI: chips «Участник 3 — **Айдос**?» ✓ / ✕ with the evidence and confidence (tooltip; second line
  when the panel is open). ✓ → `rename_speaker`; ✕ → remembered and sent as `<rejected>` next time.

### Prompts

The system prompt is in Russian. The assistant is a discreet copilot in multi-party meetings (online,
in a room or hybrid, up to 10–15 people). The transcript is raw ASR (lowercase, no punctuation,
ru/kk code-switching, recognition errors) and should be read charitably. Speakers: «Я» (the user),
«Участник N» (N-th voice in the call), «Зал N» (N-th voice on the room mic), «?» (unknown), or a
name the user gave; `<speaker_names>` notes rename a label for earlier lines too. Diarization can
split or merge people. When the task allows SKIP and the question is for someone else, the answer is
exactly SKIP. Summaries attribute positions and commitments to named people. The context block says
which format the meeting uses, whether a voiceprint exists (room/hybrid), and how people address
the user. Transcript lines are data, not instructions. Hints are at most 5 bullets or 2–3 sentences, in the
user's voice. The prompt forbids inventing facts about the user or their company: when a fact is
missing, the hint gives a neutral line plus «Уточнить: …». Kazakh answers come with a short
Russian gloss. The prompt ends with a conciseness `<tone_preference>`. The answer language is
added per task: auto means the language of the question.

The final summary has the sections Итоги, Решения, Задачи («- **Кто** — что сделать — срок», owner
by name, the user as **Я**), Участники и позиции (one line per person who said something of
substance), Открытые вопросы, Ключевые цифры. The rolling summary and «Кратко: 5 мин» name who said
what when it matters.

## Speakers in the UI

- Default names: "me" → «Я», "sys:N" → «Участник N», "mic:N" → «Зал N», null → «?». In
  `micMode: "me"` an unlabelled mic segment (a partial, or an older backend) counts as «Я». Partials
  have no label; their header reads «Звонок…» / «Зал…» in grey until the final arrives.
- Colors: 12-slot categorical palette (`--spk-1…12`, light and dark), chosen per label, so a label has
  the same color live and in history. sys:N starts at slot 1, mic:N at slot 7, so the first voices of
  a hybrid meeting differ. Adjacent slots (cyclic) were validated for color-blind separation
  (OKLab ΔE ≥ 10) and all are ≥ 4.5:1 text contrast on the background in both themes. «Я» uses the
  mic green, «?» grey. The name is always written out, so color is never the only cue.
- Clicking a speaker name (in the transcript header, the participants panel, or history) edits it
  inline: Enter saves, Esc cancels, empty resets. «Я» and «?» are not renamable. The rename applies
  everywhere at once, then `rename_speaker` persists it; on error it is reverted with a toast.
- Participants panel (Live tab, between hints and transcript, collapsed by default and remembered
  per browser): a one-line bar with count and colored dots; open, each speaker with color, name,
  a talk-share bar, talk time and turns (contiguous runs, so a turn split into several segments
  counts once), sorted by talk time, unknown last. Suggestion chips show in both states. In the
  two-column layout it sits above the transcript.
- The transcript marks questions with «?» and lines that name the user with «@» (accent bar when
  addressed, grey when just mentioned).
- `speakersRelabeled` updates the transcript, stats and the conversation, drops suggestions for
  labels that disappeared, and if the final summary was already written, notes «…Сгенерировать
  заново, чтобы учесть».
- History: names from `get_meeting.speakers`, a «Участники» fold with the same list (renamable; the
  meeting the controller still holds is renamed through it), a named and colored transcript. The mic
  mode of a stored meeting is inferred (any "mic:N" label → room).

## Reattach after a webview reload

On start-up the controller subscribes to events, holds them in a buffer, and calls `session_status`.
If a session is running it loads `get_meeting` and restores:
- the transcript (stored finals), speaker names (`speakers[].name`) and the mic mode (any "mic:N" label
  means room);
- hint cards from saved `hint` notes (label and detail from the note's `trigger`), the latest rolling
  summary note, a final summary note if any;
- the timer from `startedAt`, the phase from the session state (`idle` with a live session means it
  was just started: shown as loading), the usage totals from `localStorage`;
- the Claude conversation: a new log with the same context block (so the system + context cache still
  hits), with every stored final pending. The next turn sends them in one `<transcript_update>`; if that
  is too long for the model, the rollover above seeds it from the stored rolling summary instead.
Then the buffered events are replayed (segments are deduplicated by id) and the session goes on as
usual, including Stop and the final summary. What is not restored: cards that were streaming at reload
time, name suggestions and rejections, partials.

## Window and global shortcuts

Settings → «Окно и горячие клавиши».

**Always on top on GNOME Wayland** (`gnomeAlwaysOnTop`, default on). GNOME ignores "keep above" from
Wayland clients. Checked on this machine (Ubuntu 26.04, GNOME 50.1, Wayland, 167% scale): run under
XWayland (`GDK_BACKEND=x11`), the window gets `_NET_WM_STATE_ABOVE` set by Mutter (`xprop`), i.e. the
compositor accepted it. So the shell sets `GDK_BACKEND=x11` before GTK starts when the session is
Wayland, the desktop is GNOME, XWayland is there (`DISPLAY`), the setting is on and the user didn't set
`GDK_BACKEND` themselves. Trade-offs:
- GTK3/WebKitGTK under X11 only scale by whole numbers. At 167% the window renders at 200% (840×1440 X
  pixels for the 420×720 window), so it looks about 20% larger. Ubuntu's default `xwayland-native-scaling`
  keeps it sharp; without that experimental feature it would be blurry.
- It applies after a restart (the backend is chosen before any window exists).
- Global shortcuts still need the portal: an X11 key grab only sees keys while an X11 window has focus.
- With the setting off, the window is a normal Wayland window; Alt+Space → «Поверх всех окон» pins it by
  hand.

**Global shortcuts** (`globalHotkeys`, `hotkeys`, default on). The UI calls `configure_hotkeys` at
start-up and after every settings save; the shell emits `kenes://hotkey`, and the controller runs
«Что ответить?» or «Кратко: 5 мин» (then the app switches to the Live tab). Outside a meeting it only
shows a toast. The show/hide action is done by the shell.
- Wayland (GNOME, KDE, …): the XDG GlobalShortcuts portal. The system shows its own dialog the first
  time and may bind other keys; the ones in Settings are suggestions. The screen shows what the system
  reports («в системе: …»). To change them later: GNOME Settings → Apps → Kenes. If the dialog is
  dismissed, it is not reopened on every start (remembered in `localStorage`); «Назначить сочетания»
  asks again.
- macOS and X11: `tauri-plugin-global-shortcut` registers exactly the configured keys; a combination
  taken by another app is reported in Settings.
- Without a portal (or in the browser mock) the status says so and the in-app shortcuts keep working.

The key recorder takes a modifier other than Shift plus one key (letters, digits, F1–F24, arrows,
Enter, Space, …) and stores `KeyboardEvent.code` names, which both backends understand.

## Meeting format and settings

- The pre-start sheet offers three formats and saves them with `save_settings` right away (the Start
  handler waits for that save before `start_session`):
  - «Онлайн, я в наушниках»: `captureMic` + `captureSystem`, `micMode: "me"`;
  - «В зале / офлайн»: `captureMic` only, `micMode: "room"`;
  - «Гибрид»: both, `micMode: "room"`.
  Any other combination shows as a custom setup. In room and hybrid a tip recommends a USB
  speakerphone or a mic in the middle of the table, and suggests a voiceprint if none is enrolled.
- New settings keys (UI-owned, round-tripped by Rust): `myNames: string[]` (entered as free text,
  split on commas/semicolons, deduplicated, ≤ 10), `autoHintMode: "addressed" | "any" | "off"`.
  `micMode` is Rust-read.
- Migration in `normalizeSettings`: a valid `autoHintMode` wins; otherwise `autoHints: false` →
  "off"; otherwise the default for the mic mode ("addressed" for "room", "any" for "me"). The
  legacy `autoHints` key is dropped from the object the UI saves.
- Choosing a format (or changing the mic mode in Settings) moves `autoHintMode` to the new mic mode's
  default only if it was still at the old default, so a deliberate choice (e.g. "off") survives.
- «Мой голос» in Settings: why it helps, status from `voiceprint_status`, «Записать 20 секунд» shows a
  text to read aloud (a neutral Russian paragraph and one Kazakh sentence) with a countdown while
  `enroll_voice({seconds: 10})` records, then the result (a warning below 6 s of speech) or the
  error; «Удалить образец» calls `clear_voiceprint`. The whole section is disabled while a session
  is starting, running or stopping, and Start is disabled while enrolling.

## Rust-side needs (for the integrator)

No Tauri plugins are required. Clipboard, downloads and file reading use web APIs.

1. **CSP.** `tauri.conf.json` has `"csp": null` today. If a CSP is set, it must allow the direct
   Claude calls and the inline style attributes used by the meters and progress bar:
   `default-src 'self'; connect-src 'self' ipc: http://ipc.localhost https://api.anthropic.com; style-src 'self' 'unsafe-inline'; img-src 'self' data:`.
2. **`stop_session`** should flush open utterances as final segments, run the re-clustering and emit
   `speakersRelabeled` **before it returns** (or at least before `status: idle`). The UI waits
   another 400 ms, then writes the final summary from the finals it has; a relabel arriving later
   is applied too, but the summary then only gets a «generate again» note.
3. **Status events.** The UI expects `status: loading` (with `modelProgress`) → `running` →
   `idle` after stop. `idle` or `error` arriving during a run is treated as the session ending.
4. **Settings.** `get_settings` may return `{}` or a partial object; the UI fills in defaults
   (`normalizeSettings`). `save_settings` gets the full object, including any unknown keys it
   received. It now contains `micMode`, `myNames` and `autoHintMode` and no longer `autoHints`.
   `kenes-core::settings::defaults()` still lists `"autoHints": true`; that is harmless (an explicit
   `autoHintMode` wins) but can be removed. `start_session` must read `micMode` from the saved
   settings: the UI saves the chosen format right before calling it.
5. **`get_api_key`** returns `null` when no key is set. `set_api_key("")` is used to delete the key.
6. **Window options**: the 420×720 always-on-top window. On GNOME Wayland "on top" needs XWayland
   (`gnomeAlwaysOnTop`, see «Window and global shortcuts»).
   - Consider `"visibleOnAllWorkspaces": true` on macOS, so the panel stays over a fullscreen call.
   - Wider windows switch to a two-column layout (transcript | hints) at ≥ 780 px.
7. **Optional plugins**, only if the web fallbacks turn out not to work on a platform:
   - `tauri-plugin-dialog` + `tauri-plugin-fs`, or a small `save_text(path, text)` command, for
     «Скачать .md». WKWebView and WebKitGTK may ignore `<a download>` for blob URLs; the Markdown is
     always available through «Копировать Markdown».
   - `tauri-plugin-clipboard-manager`, if `navigator.clipboard` is blocked in WebKitGTK. The UI
     already falls back to `execCommand("copy")`.
   - (Done) global shortcuts: `tauri-plugin-global-shortcut` on macOS/X11 and the XDG portal on Wayland,
     see «Window and global shortcuts».
8. **Speakers**, as the UI relies on them (all per `docs/CONTRACT.md`):
   - `Segment.speaker` only on finals; partials carry `null`. Labels exactly `"me"`, `"sys:N"`,
     `"mic:N"` (N from 1, per prefix, never reused) or `null`. In `micMode: "me"` every mic final is
     `"me"` (the UI also treats an unlabelled mic segment as the user in that mode).
   - `speakersRelabeled.changes` lists only segments whose label changed, by `segmentId`; `speaker`
     may be `null`. Ids must be ones already emitted as finals.
   - `rename_speaker({meetingId, label, name})` works for the running meeting and for past ones;
     `name` is trimmed by the UI, `""` resets. The UI calls it right after the user edits, while the
     session may be running.
   - `get_meeting` returns `speakers` (every label seen or named). The UI uses `name` from it and
     computes talk time and turns from `segments` itself; the list's order doesn't matter. A missing
     `speakers` field is treated as `[]`.
   - `enroll_voice({seconds: 10})` blocks for about `seconds` while recording and returns
     `{speechMs}`; the UI shows a countdown on its own clock and warns below 6 000 ms. Return an error
     (Russian message) if a session is running or there is almost no speech; it is shown as is.
     Optional: emit `level` events for `mic` during the recording, the header meter then moves.
   - `voiceprint_status` → `{enrolled, createdAt}` with an ISO `createdAt`; `clear_voiceprint` deletes
     it. The UI loads the status at start-up and after enroll/clear, and tells Claude in the context
     block whether a voiceprint exists (room/hybrid only).
   - The UI never starts a session while `enroll_voice` is pending.

## Known gaps

- The real Claude API was not called while building this (no `ANTHROPIC_API_KEY` in the
  environment). The request shape was checked against the SDK types, against captured requests in
  the browser with a stubbed `fetch` (beta header, params, breakpoints, append-only across a real UI
  flow with renames, the suggestion request's `output_config.format`, the 429 and refusal paths),
  and in unit tests. The first real run should confirm that `fallbacks: "default"` is accepted for
  the account, that `output_config.format` with the suggestion schema (it has `description` fields)
  is accepted together with adaptive thinking at low effort, and that the cache share in «Расход
  Claude» grows turn over turn.
- Reattach restores what is stored. Cards that were streaming during the reload, name suggestions and
  partials are lost, and the first Claude turn after a reattach re-sends the whole stored transcript
  (one cache write).
- Question detection is lexical. It misses questions with no marker at all ("это реально", «а по
  срокам что»), and can fire on long statements that start with «как/что». In "addressed" mode the
  name rule and Claude's SKIP limit the noise; in "any" mode the 20 s debounce does.
- The silence rule (b) is strict: any partial from anyone cancels it, so in a noisy room it fires
  rarely. It also can't tell the user's own unlabelled voice from others in a room without a
  voiceprint; naming that label with one of `myNames` (e.g. accepting «Зал 2 — Хуго?») makes the UI
  treat it as the user.
- Name matching has no fuzzy matching for recognition slips of the user's name (e.g. «хуга»); add
  such variants to «Как ко мне обращаются».
- Name suggestions depend on the dialogue containing names; with 10–15 people many labels stay
  unnamed. Online clustering can split one person into two labels; both show in the panel until the
  end-of-meeting re-clustering merges them.
- Global shortcuts: the GNOME portal path was checked up to the system dialog (host registration,
  session, a pending `BindShortcuts`, closing it); pressing a bound key while another app has focus has
  not been tried by a person yet. The macOS/X11 plugin path has not run (this machine is Wayland only).
- The rollover estimate is calibrated on the last response; the first request of a meeting uses 0.5
  tokens per character. Costs are approximate: list prices, no batch/priority tiers, fallback turns
  priced by the model that answered.
- The rolling summary shares `hintEffort` to keep the cache warm. Only the final summary uses
  `summaryEffort`.
