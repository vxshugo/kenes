import { useState } from "react";
import { copyText, downloadText } from "../lib/clipboard";
import { fileSlug, formatTime, isoDate, meetingMarkdown } from "../lib/format";
import { contextWindow } from "../llm/budget";
import { cacheHitRate, formatCost, formatTokens, type UsageTotals } from "../llm/usage";
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

/** «раз» / «раза»: 1, 5–20 раз; 2–4 раза. */
function times(n: number): string {
  const d = n % 10;
  const t = n % 100;
  return d >= 2 && d <= 4 && (t < 12 || t > 14) ? "раза" : "раз";
}

/** Claude tokens, cache hit rate and approximate cost of this meeting. */
function UsageView({ usage, model }: { usage: UsageTotals; model: string }) {
  if (!usage.requests) {
    return <p className="muted small">Запросов к Claude в этой встрече ещё не было.</p>;
  }
  const prompt = usage.input + usage.cacheRead + usage.cacheWrite;
  const hit = cacheHitRate(usage);
  const window = contextWindow(model);
  return (
    <>
      <dl className="usage-grid">
        <div>
          <dt>Запросов</dt>
          <dd>{usage.requests}</dd>
        </div>
        <div title="Все входные токены: без кэша + запись в кэш + чтение из кэша">
          <dt>Вход</dt>
          <dd>{formatTokens(prompt)}</dd>
        </div>
        <div title="Доля входных токенов, прочитанных из кэша промпта (cache_read_input_tokens)">
          <dt>Из кэша</dt>
          <dd>{hit === null ? "—" : `${Math.round(hit * 100)}%`}</dd>
        </div>
        <div title="Выходные токены, включая размышления модели">
          <dt>Выход</dt>
          <dd>{formatTokens(usage.output)}</dd>
        </div>
        <div title="Примерно, по ценам API для ответившей модели">
          <dt>≈ Цена</dt>
          <dd>
            {formatCost(usage.cost)}
            {usage.unpriced > 0 ? "+" : ""}
          </dd>
        </div>
      </dl>
      <p className="muted small usage-detail">
        Без кэша {formatTokens(usage.input)} · запись в кэш {formatTokens(usage.cacheWrite)} · чтение из кэша {formatTokens(usage.cacheRead)}
        {usage.peakPrompt > 0 && ` · самый длинный запрос ${formatTokens(usage.peakPrompt)} из ${formatTokens(window)}`}
        {usage.rollovers > 0 && ` · разговор сжат ${usage.rollovers} ${times(usage.rollovers)}`}
        {usage.unpriced > 0 && ` · цена неизвестна для ${usage.unpriced} запр.`}
      </p>
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

      <section className="doc usage" aria-label="Расход Claude">
        <header className="doc-head">
          <h2>Расход Claude</h2>
          <span className="muted small">{state.settings.claudeModel}</span>
        </header>
        <UsageView usage={state.usage} model={state.settings.claudeModel} />
      </section>
    </div>
  );
}
