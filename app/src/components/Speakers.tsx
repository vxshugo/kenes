import { useMemo, useState, type KeyboardEvent } from "react";
import {
  defaultSpeakerName,
  formatTalkTime,
  isRenamable,
  pluralTurns,
  speakerClass,
  SpeakerDirectory,
  type SpeakerStat,
} from "../llm/speakers";
import type { SuggestionChip } from "../session/controller";
import type { MicMode } from "../types";
import { IconCheck, IconX } from "./Icons";

/** A directory for rendering: stable while the mic mode and names don't change. */
export function useSpeakerDirectory(micMode: MicMode, names: Record<string, string>): SpeakerDirectory {
  return useMemo(() => new SpeakerDirectory(micMode, names), [micMode, names]);
}

type NameProps = {
  label: string | null;
  name: string;
  /** The name is a custom one (prefilled when editing; otherwise the default is the placeholder). */
  custom: boolean;
  onRename?: (name: string) => void;
  className?: string;
};

/** A speaker's name in their color; click to rename inline (Enter saves, Esc cancels, empty resets). */
export function SpeakerName({ label, name, custom, onRename, className = "" }: NameProps) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState("");
  const cls = `spk ${speakerClass(label)} ${className}`.trim();
  if (!onRename || !isRenamable(label)) {
    return (
      <span className={cls} title={label === "me" ? "Вы" : label ? undefined : "Говорящий не определён"}>
        {name}
      </span>
    );
  }
  const fallback = defaultSpeakerName(label);
  const commit = () => {
    setEditing(false);
    const next = value.replace(/\s+/g, " ").trim();
    if (next === (custom ? name : "")) return;
    onRename(next);
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      commit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      setEditing(false);
    }
  };
  if (editing) {
    return (
      <input
        className={`spk-input ${speakerClass(label)}`}
        autoFocus
        value={value}
        maxLength={40}
        placeholder={fallback}
        spellCheck={false}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKey}
        onBlur={commit}
        onFocus={(e) => e.currentTarget.select()}
        aria-label={`Имя для «${fallback}»`}
      />
    );
  }
  return (
    <button
      type="button"
      className={`${cls} spk-btn`}
      onClick={() => {
        setValue(custom ? name : "");
        setEditing(true);
      }}
      title={`${custom ? `${name} (${fallback})` : name} — нажмите, чтобы назвать по имени`}
    >
      {name}
    </button>
  );
}

/** Participants by talk time: color, name (renamable), share bar, talk time and turns. */
export function ParticipantsList({ stats, onRename }: { stats: SpeakerStat[]; onRename?: (label: string, name: string) => void }) {
  const max = Math.max(1, ...stats.map((s) => s.talkMs));
  return (
    <ul className="participants">
      {stats.map((st) => (
        <li className="participant" key={st.label ?? "?"}>
          <span className={`spk-dot ${speakerClass(st.label)}`} aria-hidden="true" />
          <span className="participant-name">
            <SpeakerName
              label={st.label}
              name={st.label ? st.name : "Не определено"}
              custom={st.custom}
              onRename={onRename && st.label ? (n) => onRename(st.label!, n) : undefined}
            />
            {st.custom && st.label && <span className="participant-label">{defaultSpeakerName(st.label)}</span>}
          </span>
          <span className="participant-bar" aria-hidden="true">
            <span className={speakerClass(st.label)} style={{ transform: `scaleX(${st.talkMs / max})` }} />
          </span>
          <span className="participant-meta">
            {formatTalkTime(st.talkMs)} · {pluralTurns(st.turns)}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** «Участник 3 — Айдос?» ✓ / ✕, with Claude's evidence. */
export function SuggestionChips({
  suggestions,
  onAccept,
  onDismiss,
  detailed,
}: {
  suggestions: SuggestionChip[];
  onAccept: (id: string) => void;
  onDismiss: (id: string) => void;
  detailed: boolean;
}) {
  return (
    <ul className="suggestions" aria-label="Предложенные имена">
      {suggestions.map((s) => {
        const why = `${s.evidence || "по разговору"} · уверенность ${Math.round(s.confidence * 100)}%`;
        return (
          <li className="suggestion" key={s.id} title={why}>
            <span className={`spk-dot ${speakerClass(s.label)}`} aria-hidden="true" />
            <span className="suggestion-text">
              <span className="suggestion-main">
                {defaultSpeakerName(s.label)} — <strong>{s.name}</strong>?
              </span>
              {detailed && <span className="suggestion-why">{why}</span>}
            </span>
            <button type="button" className="icon-btn chip-yes" onClick={() => onAccept(s.id)} title={`Да, это ${s.name}`} aria-label={`Да, ${defaultSpeakerName(s.label)} — это ${s.name}`}>
              <IconCheck size={15} />
            </button>
            <button type="button" className="icon-btn" onClick={() => onDismiss(s.id)} title="Нет" aria-label={`Нет, ${defaultSpeakerName(s.label)} — не ${s.name}`}>
              <IconX size={14} />
            </button>
          </li>
        );
      })}
    </ul>
  );
}
