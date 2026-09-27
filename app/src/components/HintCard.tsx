import { useState } from "react";
import { copyText } from "../lib/clipboard";
import { formatTime } from "../lib/format";
import type { Card } from "../session/controller";
import { IconBook, IconCheck, IconChevron, IconClock, IconCopy, IconRefresh, IconSpark, IconTranslate, IconX } from "./Icons";
import { Markdown } from "./Markdown";

const KIND_ICON = {
  hint: IconSpark,
  ask: IconSpark,
  explain: IconBook,
  translate: IconTranslate,
  recap: IconClock,
} as const;

type Props = {
  card: Card;
  expanded: boolean;
  onToggle: () => void;
  onRetry: () => void;
  onDismiss: () => void;
};

export function HintCard({ card, expanded, onToggle, onRetry, onDismiss }: Props) {
  const [copied, setCopied] = useState(false);
  const Icon = KIND_ICON[card.kind];
  const busy = card.status === "queued" || card.status === "streaming";
  const preview =
    card.status === "error"
      ? card.error
      : (card.text
          .split("\n")
          .map((l) => l.replace(/^\s*(?:[-*+•]|\d+[.)])\s+/, "").replace(/[*_`#>]/g, "").trim())
          .find(Boolean) ?? "");

  const copy = async () => {
    if (await copyText(card.text)) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <article className={`card card-${card.kind} is-${card.status}${expanded ? " is-expanded" : ""}`} aria-busy={busy}>
      <header className="card-head">
        <button className="card-toggle" onClick={onToggle} aria-expanded={expanded} title={expanded ? "Свернуть" : "Развернуть"}>
          <IconChevron className="chev" size={14} />
          <Icon size={15} className="card-icon" />
          <span className="card-label">{card.label}</span>
          {card.detail && <span className="card-detail">«{card.detail}»</span>}
        </button>
        <span className="card-time" title={new Date(card.createdAt).toLocaleString("ru-RU")}>
          {formatTime(card.createdAt)}
        </span>
        {card.status === "done" && (
          <button className="icon-btn" onClick={copy} title="Копировать" aria-label="Копировать ответ">
            {copied ? <IconCheck /> : <IconCopy />}
          </button>
        )}
        {card.status === "error" && (
          <button className="icon-btn" onClick={onRetry} title="Повторить" aria-label="Повторить запрос">
            <IconRefresh />
          </button>
        )}
        {!busy && (
          <button className="icon-btn" onClick={onDismiss} title="Убрать" aria-label="Убрать карточку">
            <IconX />
          </button>
        )}
      </header>
      {expanded ? (
        <div className="card-body">
          {card.status === "queued" && <p className="muted small">В очереди…</p>}
          {card.status === "streaming" && !card.text && <p className="muted small thinking">Claude думает…</p>}
          {card.text && <Markdown text={card.text} className={card.status === "streaming" ? "is-streaming-text" : undefined} />}
          {card.status === "error" && <p className="card-error">{card.error}</p>}
          {card.truncated && <p className="muted small">Ответ обрезан по лимиту длины.</p>}
          {card.note && <p className="muted small">{card.note}</p>}
        </div>
      ) : (
        preview && (
          <button className="card-preview" onClick={onToggle}>
            {preview}
          </button>
        )
      )}
    </article>
  );
}
