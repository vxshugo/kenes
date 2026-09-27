import { useMemo, useState, type RefObject } from "react";
import { SHORTCUTS } from "../lib/keys";
import { speakerClass, speakerStats } from "../llm/speakers";
import { unnamedLabels } from "../llm/suggestions";
import type { ControllerState } from "../session/controller";
import { controller } from "../session/useController";
import { HintCard } from "./HintCard";
import { IconBook, IconChevron, IconClock, IconSend, IconSpark, IconTranslate, IconWand } from "./Icons";
import { ParticipantsList, SuggestionChips, useSpeakerDirectory } from "./Speakers";
import { Transcript } from "./Transcript";

const PANEL_KEY = "kenes.ui.participantsOpen";

function readOpen(): boolean {
  try {
    return localStorage.getItem(PANEL_KEY) === "1";
  } catch {
    return false;
  }
}

const rename = (label: string, name: string) => void controller.renameSpeaker(label, name);

/** Collapsible participants strip: who spoke how much, inline rename, Claude's name suggestions. */
function ParticipantsPanel({ state }: { state: ControllerState }) {
  const [open, setOpen] = useState(readOpen);
  const speakers = useSpeakerDirectory(state.micMode, state.speakerNames);
  const stats = useMemo(() => speakerStats(state.finals, speakers), [state.finals, speakers]);
  const unnamed = useMemo(() => unnamedLabels(state.finals, speakers).length, [state.finals, speakers]);
  const people = stats.filter((s) => s.label !== null);
  if (!stats.length && !state.suggestions.length) return null;
  const toggle = () => {
    setOpen(!open);
    try {
      localStorage.setItem(PANEL_KEY, open ? "0" : "1");
    } catch {
      // per-viewer convenience only
    }
  };
  return (
    <section className={`participants-panel${open ? " is-open" : ""}`} aria-label="Участники">
      <div className="pp-head">
        <button type="button" className="pp-toggle" onClick={toggle} aria-expanded={open} title={open ? "Свернуть список участников" : "Показать участников"}>
          <IconChevron size={14} className="chev" />
          <span className="pp-title">Участники</span>
          <span className="pp-count">{people.length}</span>
          {!open && (
            <span className="pp-dots" aria-hidden="true">
              {people.slice(0, 10).map((p) => (
                <span key={p.label} className={`spk-dot ${speakerClass(p.label)}`} title={p.name} />
              ))}
            </span>
          )}
        </button>
        <button
          type="button"
          className="icon-btn"
          onClick={() => void controller.suggestNames(true)}
          disabled={state.suggesting || !state.apiKey || !unnamed}
          title={
            !state.apiKey
              ? "Нужен API-ключ Claude"
              : unnamed
                ? "Предложить имена безымянных говорящих по разговору"
                : "Все говорящие подписаны"
          }
          aria-label="Предложить имена по разговору"
        >
          <IconWand className={state.suggesting ? "is-busy" : undefined} />
        </button>
      </div>
      {state.suggestions.length > 0 && (
        <SuggestionChips
          suggestions={state.suggestions}
          detailed={open}
          onAccept={(id) => controller.acceptSuggestion(id)}
          onDismiss={(id) => controller.dismissSuggestion(id)}
        />
      )}
      {open && (
        <div className="pp-body">
          <ParticipantsList stats={stats} onRename={rename} />
          <p className="pp-hint muted small">Нажмите на имя, чтобы переименовать — здесь или прямо в расшифровке.</p>
        </div>
      )}
    </section>
  );
}

type Props = {
  state: ControllerState;
  askRef: RefObject<HTMLInputElement | null>;
  question: string;
  setQuestion: (q: string) => void;
  onExplain: () => void;
};

export function canAct(state: ControllerState): boolean {
  return ["loading", "running", "stopping", "stopped"].includes(state.phase) || (state.phase === "error" && state.finals.length > 0);
}

function autoHintNote(state: ControllerState): string {
  const mode = state.settings.autoHintMode;
  if (mode === "off") return ".";
  if (mode === "any") return " — или дождитесь вопроса: авто-подсказки срабатывают на любой вопрос.";
  const names = state.settings.myNames;
  return names.length
    ? ` — или дождитесь, когда к вам обратятся (${names.join(", ")}).`
    : " — авто-подсказки ждут вопроса к вам; добавьте своё имя в «Настройках», чтобы узнавать обращения.";
}

function Hints({ state }: { state: ControllerState }) {
  const [overrides, setOverrides] = useState<Record<string, boolean>>({});
  // Auto hints that may still be SKIP stay invisible.
  const cards = state.cards.filter((c) => !c.hidden);
  if (!cards.length) {
    return (
      <section className="hints hints-empty" aria-label="Подсказки">
        <p>
          Подсказки появятся здесь. Нажмите <strong>«Что ответить?»</strong> ({SHORTCUTS.hint})
          {autoHintNote(state)}
        </p>
      </section>
    );
  }
  return (
    <section className="hints" aria-label="Подсказки" aria-live="polite">
      {cards.map((card, i) => {
        // Newest and in-progress cards open; older finished ones collapse to one line.
        const expanded = overrides[card.id] ?? (i === 0 || card.status === "streaming");
        return (
          <HintCard
            key={card.id}
            card={card}
            expanded={expanded}
            onToggle={() => setOverrides((o) => ({ ...o, [card.id]: !expanded }))}
            onRetry={() => controller.retry(card.id)}
            onDismiss={() => controller.dismissCard(card.id)}
          />
        );
      })}
    </section>
  );
}

export function LiveTab({ state, askRef, question, setQuestion, onExplain }: Props) {
  const enabled = canAct(state);
  const speakers = useSpeakerDirectory(state.micMode, state.speakerNames);
  const empty =
    state.phase === "loading"
      ? "Загружается модель распознавания…"
      : state.phase === "running"
        ? "Слушаю… Реплики появятся здесь."
        : "Расшифровка пуста.";

  const submitAsk = () => {
    const q = question.trim();
    if (!q || !enabled) return;
    controller.ask(q);
    setQuestion("");
  };

  return (
    <div className="live">
      <Hints state={state} />
      <ParticipantsPanel state={state} />
      <Transcript
        finals={state.finals}
        partials={state.partials}
        speakers={speakers}
        myNames={state.settings.myNames}
        onRename={rename}
        emptyText={empty}
      />
      <div className="actions">
        <div className="action-row">
          <button className="btn btn-primary btn-hint" disabled={!enabled} onClick={() => controller.hint()} title={`Подсказать ответ на последний вопрос (${SHORTCUTS.hint})`}>
            <IconSpark /> Что ответить?
          </button>
          <button className="btn" disabled={!enabled} onClick={() => controller.recap(5)} title={`Что обсуждали за последние 5 минут (${SHORTCUTS.recap})`}>
            <IconClock /> Кратко: 5 мин
          </button>
          <button className="btn" disabled={!enabled} onClick={() => controller.translate()} title={`Перевести казахские реплики на русский (${SHORTCUTS.translate})`}>
            <IconTranslate /> Перевести
          </button>
          <button
            className="btn"
            disabled={!enabled}
            onMouseDown={(e) => e.preventDefault() /* keep the text selection */}
            onClick={onExplain}
            title={`Объяснить выделенный термин (${SHORTCUTS.explain})`}
          >
            <IconBook /> Объяснить
          </button>
        </div>
        <form
          className="ask-row"
          onSubmit={(e) => {
            e.preventDefault();
            submitAsk();
          }}
        >
          <input
            ref={askRef}
            type="text"
            value={question}
            disabled={!enabled}
            placeholder={enabled ? "Спросить у ассистента…" : "Начните встречу, чтобы задавать вопросы"}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") e.currentTarget.blur();
            }}
            title={`Вопрос ассистенту о встрече (${SHORTCUTS.ask}); Enter — отправить`}
            aria-label="Вопрос ассистенту"
          />
          <button className="icon-btn icon-btn-solid" type="submit" disabled={!enabled || !question.trim()} aria-label="Отправить вопрос" title="Отправить (Enter)">
            <IconSend />
          </button>
        </form>
      </div>
    </div>
  );
}
