import { useCallback, useEffect, useMemo, useState } from "react";
import { describeError } from "../llm/client";
import { cleanSpeakerName, defaultSpeakerName, inferMicMode, isRenamable, speakerStats, SpeakerDirectory } from "../llm/speakers";
import { copyText, downloadText } from "../lib/clipboard";
import { fileSlug, formatDateTime, formatDuration, formatTime, isoDate, meetingMarkdown } from "../lib/format";
import { controller } from "../session/useController";
import type { Meeting, MeetingSummary, Note } from "../types";
import { IconArrowLeft, IconCheck, IconChevron, IconCopy, IconDownload, IconRefresh, IconTrash } from "./Icons";
import { Markdown } from "./Markdown";
import { ParticipantsList } from "./Speakers";
import { TranscriptLines } from "./Transcript";

const NOTE_LABEL: Record<Note["kind"], string> = { final: "Итоги", summary: "Резюме по ходу", hint: "Подсказка" };

function hintTitle(trigger: string | null): string {
  if (!trigger) return "Подсказка";
  if (trigger === "manual") return "Что ответить?";
  if (trigger === "translate") return "Перевод";
  if (trigger.startsWith("auto: ")) return `Авто: «${trigger.slice(6)}»`;
  if (trigger.startsWith("ask: ")) return `Вопрос: «${trigger.slice(5)}»`;
  if (trigger.startsWith("explain: ")) return `Объяснение: «${trigger.slice(9)}»`;
  if (trigger.startsWith("recap: ")) return `Кратко: ${trigger.slice(7).replace("m", " мин")}`;
  return trigger;
}

function MeetingView({ id, onBack, onDeleted, activeId }: { id: string; onBack: () => void; onDeleted: () => void; activeId: string | null }) {
  const [meeting, setMeeting] = useState<Meeting | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [copied, setCopied] = useState(false);
  const [names, setNames] = useState<Record<string, string>>({});

  useEffect(() => {
    let alive = true;
    controller.api
      .getMeeting(id)
      .then((m) => {
        if (!alive) return;
        setMeeting(m);
        const map: Record<string, string> = {};
        for (const sp of m.speakers ?? []) if (sp.name) map[sp.label] = sp.name;
        setNames(map);
      })
      .catch((e) => alive && setError(describeError(e)));
    return () => {
      alive = false;
    };
  }, [id]);

  const segments = useMemo(() => [...(meeting?.segments ?? [])].sort((a, b) => a.startMs - b.startMs), [meeting]);
  const speakers = useMemo(() => new SpeakerDirectory(inferMicMode(segments), names), [segments, names]);
  const stats = useMemo(
    () => speakerStats(segments, speakers, (meeting?.speakers ?? []).map((sp) => sp.label)),
    [segments, speakers, meeting],
  );

  if (error) {
    return (
      <div className="tab-pane">
        <button className="btn btn-ghost btn-sm" onClick={onBack}>
          <IconArrowLeft /> Назад
        </button>
        <p className="card-error">{error}</p>
      </div>
    );
  }
  if (!meeting) return <div className="tab-pane muted">Загрузка…</div>;

  const finals = meeting.notes.filter((n) => n.kind === "final");
  const summaries = meeting.notes.filter((n) => n.kind === "summary");
  const hints = meeting.notes.filter((n) => n.kind === "hint");
  const latestFinal = finals.at(-1) ?? null;
  const latestSummary = summaries.at(-1) ?? null;
  const isActive = activeId === meeting.id;

  const rename = async (label: string, name: string) => {
    if (!isRenamable(label)) return;
    const clean = cleanSpeakerName(name);
    const next = { ...names };
    if (clean && clean !== defaultSpeakerName(label)) next[label] = clean;
    else delete next[label];
    const before = names;
    setNames(next);
    try {
      // The meeting the controller still holds: rename through it so the live view stays in sync.
      if (controller.getState().meetingId === meeting.id) await controller.renameSpeaker(label, clean);
      else await controller.api.renameSpeaker(meeting.id, label, next[label] ?? "");
    } catch (e) {
      setNames(before);
      setError(describeError(e));
    }
  };

  const markdown = () =>
    meetingMarkdown({
      title: meeting.title,
      startedAt: meeting.startedAt,
      summary: latestFinal?.content ?? null,
      rolling: latestSummary?.content ?? null,
      segments,
      speakers,
    });

  const remove = async () => {
    try {
      await controller.api.deleteMeeting(meeting.id);
      onDeleted();
    } catch (e) {
      setError(describeError(e));
    }
  };

  return (
    <div className="tab-pane meeting">
      <div className="toolbar">
        <button className="btn btn-ghost btn-sm" onClick={onBack}>
          <IconArrowLeft /> Назад
        </button>
        <span className="spacer" />
        <button
          className="icon-btn"
          title="Копировать Markdown"
          aria-label="Копировать Markdown"
          onClick={async () => {
            if (await copyText(markdown())) {
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1500);
            }
          }}
        >
          {copied ? <IconCheck /> : <IconCopy />}
        </button>
        <button className="icon-btn" title="Скачать .md" aria-label="Скачать .md" onClick={() => downloadText(`${fileSlug(meeting.title)}-${isoDate(meeting.startedAt)}.md`, markdown())}>
          <IconDownload />
        </button>
        {confirming ? (
          <>
            <button className="btn btn-danger btn-sm" onClick={remove}>
              Удалить навсегда
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => setConfirming(false)}>
              Отмена
            </button>
          </>
        ) : (
          <button className="icon-btn" title={isActive ? "Идущую встречу удалить нельзя" : "Удалить встречу"} aria-label="Удалить встречу" disabled={isActive} onClick={() => setConfirming(true)}>
            <IconTrash />
          </button>
        )}
      </div>

      <h2 className="meeting-title">{meeting.title}</h2>
      <p className="muted small">
        {formatDateTime(meeting.startedAt)}
        {formatDuration(meeting.startedAt, meeting.endedAt) ? ` · ${formatDuration(meeting.startedAt, meeting.endedAt)}` : meeting.endedAt ? "" : " · идёт"}
        {` · ${segments.length} реплик`}
      </p>

      {meeting.context.trim() && (
        <details className="fold">
          <summary>
            <IconChevron size={14} className="chev" /> Контекст встречи
          </summary>
          <pre className="context-text">{meeting.context}</pre>
        </details>
      )}

      {latestFinal && (
        <section className="doc">
          <h3 className="doc-title">Итоги</h3>
          <Markdown text={latestFinal.content} />
        </section>
      )}
      {!latestFinal && latestSummary && (
        <section className="doc">
          <h3 className="doc-title">Резюме по ходу</h3>
          <Markdown text={latestSummary.content} />
        </section>
      )}

      {hints.length > 0 && (
        <details className="fold">
          <summary>
            <IconChevron size={14} className="chev" /> Подсказки ({hints.length})
          </summary>
          <div className="note-list">
            {hints.map((n) => (
              <div className="note" key={n.id}>
                <div className="note-head">
                  <span>{hintTitle(n.trigger)}</span>
                  <span className="muted small">{formatTime(n.createdAt)}</span>
                </div>
                <Markdown text={n.content} />
              </div>
            ))}
          </div>
        </details>
      )}
      {(summaries.length > 1 || (latestFinal && summaries.length > 0)) && (
        <details className="fold">
          <summary>
            <IconChevron size={14} className="chev" /> {NOTE_LABEL.summary} ({summaries.length})
          </summary>
          <div className="note-list">
            {summaries.map((n) => (
              <div className="note" key={n.id}>
                <div className="note-head">
                  <span className="muted small">{formatTime(n.createdAt)}</span>
                </div>
                <Markdown text={n.content} />
              </div>
            ))}
          </div>
        </details>
      )}

      {stats.length > 0 && (
        <details className="fold">
          <summary>
            <IconChevron size={14} className="chev" /> Участники ({stats.filter((s) => s.label !== null).length})
          </summary>
          <ParticipantsList stats={stats} onRename={(label, name) => void rename(label, name)} />
        </details>
      )}

      <details className="fold" open={!latestFinal}>
        <summary>
          <IconChevron size={14} className="chev" /> Расшифровка
        </summary>
        <div className="transcript transcript-static" data-selectable="transcript">
          {segments.length === 0 ? (
            <p className="empty">Реплик нет.</p>
          ) : (
            <TranscriptLines
              segments={segments}
              speakers={speakers}
              myNames={controller.getState().settings.myNames}
              onRename={(label, name) => void rename(label, name)}
            />
          )}
        </div>
      </details>
    </div>
  );
}

export function HistoryTab({ activeId }: { activeId: string | null }) {
  const [list, setList] = useState<MeetingSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    controller.api
      .listMeetings()
      .then((l) => setList([...l].sort((a, b) => b.startedAt.localeCompare(a.startedAt))))
      .catch((e) => setError(describeError(e)));
  }, []);

  useEffect(() => {
    if (!openId) load();
  }, [openId, load]);

  if (openId) {
    return (
      <MeetingView
        id={openId}
        activeId={activeId}
        onBack={() => setOpenId(null)}
        onDeleted={() => {
          setOpenId(null);
        }}
      />
    );
  }

  return (
    <div className="tab-pane history">
      <div className="toolbar">
        <h2>Встречи</h2>
        <span className="spacer" />
        <button className="icon-btn" onClick={load} title="Обновить список" aria-label="Обновить список">
          <IconRefresh />
        </button>
      </div>
      {error && <p className="card-error">{error}</p>}
      {list === null && !error && <p className="muted">Загрузка…</p>}
      {list && list.length === 0 && <p className="empty">Здесь появятся прошедшие встречи с расшифровкой и итогами.</p>}
      {list && list.length > 0 && (
        <ul className="meeting-list">
          {list.map((m) => (
            <li key={m.id}>
              <button className="meeting-item" onClick={() => setOpenId(m.id)}>
                <span className="meeting-item-title">{m.title}</span>
                <span className="muted small">
                  {formatDateTime(m.startedAt)}
                  {m.endedAt ? (formatDuration(m.startedAt, m.endedAt) ? ` · ${formatDuration(m.startedAt, m.endedAt)}` : "") : m.id === activeId ? " · идёт сейчас" : " · не завершена"}
                </span>
                <IconChevron className="meeting-item-chev" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
