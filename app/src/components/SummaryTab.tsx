import { useState } from "react";
import { copyText, downloadText } from "../lib/clipboard";
import { fileSlug, formatTime, isoDate, meetingMarkdown } from "../lib/format";
import type { ControllerState, SummaryDoc } from "../session/controller";
import { controller } from "../session/useController";
import { IconCheck, IconCopy, IconDownload, IconRefresh } from "./Icons";
import { Markdown } from "./Markdown";
import { useSpeakerDirectory } from "./Speakers";

function DocView({ doc, empty }: { doc: SummaryDoc; empty: string }) {
  if (doc.status === "idle" && !doc.text) return <p className="muted">{empty}</p>;
  return (
    <>
      {doc.status === "streaming" && !doc.text && <p className="muted thinking">Claude пишет…</p>}
      {doc.text && <Markdown text={doc.text} className={doc.status === "streaming" ? "is-streaming-text" : undefined} />}
      {doc.error && <p className="card-error">{doc.error}</p>}
      {doc.truncated && <p className="muted small">Текст обрезан по лимиту длины.</p>}
      {doc.note && <p className="muted small">{doc.note}</p>}
    </>
  );
}

export function SummaryTab({ state }: { state: ControllerState }) {
  const [copied, setCopied] = useState(false);
  const { final, rolling } = state;
  const speakers = useSpeakerDirectory(state.micMode, state.speakerNames);
  const hasSession = state.phase !== "idle";
  const summaryText = final.status === "done" ? final.text : rolling.text;

  const copy = async () => {
    if (summaryText && (await copyText(summaryText))) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }
  };

  const download = () => {
    const md = meetingMarkdown({
      title: state.title,
      startedAt: state.startedAt,
      summary: final.status === "done" ? final.text : null,
      rolling: rolling.text,
      segments: state.finals,
      speakers,
    });
    downloadText(`${fileSlug(state.title)}-${isoDate(state.startedAt ?? Date.now())}.md`, md);
  };

  if (!hasSession) {
    return (
      <div className="tab-pane">
        <p className="empty">Итоги появятся во время и после встречи. Резюме по ходу обновляется каждые {state.settings.rollingSummaryMinutes || "—"} мин, итоговое — автоматически после «Стоп».</p>
      </div>
    );
  }

  const finalEmpty =
    state.phase === "stopped"
      ? state.apiKey
        ? state.finals.length
          ? "Итоги ещё не готовы."
          : "Встреча без реплик — подводить нечего."
        : "Добавьте ключ Claude в настройках, чтобы подвести итоги."
      : "Сформируется автоматически после «Стоп».";

  return (
    <div className="tab-pane summary">
      <div className="toolbar">
        <button className="btn btn-sm" onClick={copy} disabled={!summaryText} title="Копировать итоги в Markdown">
          {copied ? <IconCheck /> : <IconCopy />} Копировать Markdown
        </button>
        <button className="btn btn-sm" onClick={download} disabled={!summaryText && !state.finals.length} title="Скачать итоги и расшифровку (.md)">
          <IconDownload /> Скачать .md
        </button>
      </div>

      <section className="doc">
        <header className="doc-head">
          <h2>Итоги встречи</h2>
          {final.updatedAt && <span className="muted small">{formatTime(final.updatedAt)}</span>}
          {state.phase === "stopped" && state.finals.length > 0 && (
            <button className="icon-btn" onClick={() => void controller.generateFinal()} disabled={final.status === "streaming"} title="Сгенерировать заново" aria-label="Сгенерировать итоги заново">
              <IconRefresh />
            </button>
          )}
        </header>
        <DocView doc={final} empty={finalEmpty} />
      </section>

      <section className="doc">
        <header className="doc-head">
          <h2>Резюме по ходу</h2>
          {rolling.updatedAt && <span className="muted small">обновлено {formatTime(rolling.updatedAt)}</span>}
          <button
            className="icon-btn"
            onClick={() => void controller.generateRolling()}
            disabled={rolling.status === "streaming" || !state.finals.length}
            title="Обновить сейчас"
            aria-label="Обновить резюме сейчас"
          >
            <IconRefresh />
          </button>
        </header>
        <DocView
          doc={rolling}
          empty={
            state.settings.rollingSummaryMinutes > 0
              ? `Обновляется каждые ${state.settings.rollingSummaryMinutes} мин, если было что-то новое.`
              : "Автообновление выключено в настройках — можно обновить вручную."
          }
        />
      </section>
    </div>
  );
}
