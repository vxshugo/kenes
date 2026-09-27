import { useRef, useState, type DragEvent } from "react";
import { MOCK_MEETING_CONTEXT, MOCK_MEETING_TITLE } from "../backend/mockScript";
import { FORMAT_ORDER, formatOf, MEETING_FORMATS } from "../lib/meetingFormat";
import type { ControllerState } from "../session/controller";
import { controller } from "../session/useController";
import { IconFile, IconRecord } from "./Icons";

export type Draft = { title: string; context: string };

type Props = {
  state: ControllerState;
  draft: Draft;
  onChange: (d: Draft) => void;
  onStart: () => void;
  onOpenSettings: () => void;
};

const TEXT_EXT = /\.(txt|md|markdown|text)$/i;

async function readFiles(files: FileList | File[]): Promise<{ text: string; skipped: string[] }> {
  const parts: string[] = [];
  const skipped: string[] = [];
  for (const f of Array.from(files)) {
    if (!TEXT_EXT.test(f.name) && !f.type.startsWith("text/")) {
      skipped.push(f.name);
      continue;
    }
    if (f.size > 2_000_000) {
      skipped.push(f.name);
      continue;
    }
    const text = (await f.text()).trim();
    if (text) parts.push(`--- ${f.name} ---\n${text}`);
  }
  return { text: parts.join("\n\n"), skipped };
}

export function PreStart({ state, draft, onChange, onStart, onOpenSettings }: Props) {
  const [dragging, setDragging] = useState(false);
  const [fileNote, setFileNote] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const s = state.settings;
  const format = formatOf(s);
  const room = s.captureMic && s.micMode === "room";
  const autoLine =
    s.autoHintMode === "off"
      ? "Авто-подсказки выключены."
      : s.autoHintMode === "any"
        ? "Авто-подсказки: на любой вопрос."
        : `Авто-подсказки: когда обращаются ко мне${s.myNames.length ? ` (${s.myNames.join(", ")})` : " — имя не указано"}.`;

  const append = async (files: FileList | File[]) => {
    const { text, skipped } = await readFiles(files);
    if (text) onChange({ ...draft, context: draft.context.trim() ? `${draft.context.trim()}\n\n${text}` : text });
    setFileNote(skipped.length ? `Пропущено (нужен .txt или .md до 2 МБ): ${skipped.join(", ")}` : text ? "Файл добавлен в контекст." : null);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files.length) void append(e.dataTransfer.files);
  };

  return (
    <form
      className="prestart"
      onSubmit={(e) => {
        e.preventDefault();
        onStart();
      }}
    >
      {state.backendKind === "mock" && (
        <div className="banner banner-info">
          Демо-режим в браузере: вместо захвата звука воспроизводится сценарий встречи.{" "}
          <button type="button" className="link" onClick={() => onChange({ title: MOCK_MEETING_TITLE, context: MOCK_MEETING_CONTEXT })}>
            Заполнить пример
          </button>
        </div>
      )}
      {!state.apiKey && (
        <div className="banner banner-warn">
          Без ключа Claude запись и расшифровка работают, но подсказок не будет.{" "}
          <button type="button" className="link" onClick={onOpenSettings}>
            Добавить ключ
          </button>
        </div>
      )}

      <label className="field">
        <span className="field-label">Название</span>
        <input
          type="text"
          value={draft.title}
          placeholder="Например: Синк по релизу 2.3"
          onChange={(e) => onChange({ ...draft, title: e.target.value })}
          autoFocus
        />
      </label>

      <label className="field field-grow">
        <span className="field-label">
          Контекст встречи
          <span className="field-hint">повестка, заметки, кто есть кто — Claude опирается только на это и на разговор</span>
        </span>
        <div
          className={`dropzone${dragging ? " is-dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <textarea
            value={draft.context}
            placeholder={"Повестка:\n1. …\nУчастники: Айгерим — PM, Ерлан — продукт, Айдос — мобилка\nМои цифры: …\n\nМожно перетащить сюда .txt или .md"}
            onChange={(e) => onChange({ ...draft, context: e.target.value })}
          />
          {dragging && <div className="dropzone-overlay">Отпустите, чтобы добавить файл</div>}
        </div>
      </label>
      <div className="prestart-files">
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => fileInput.current?.click()}>
          <IconFile /> Добавить файл
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".txt,.md,.markdown,text/plain,text/markdown"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files?.length) void append(e.target.files);
            e.target.value = "";
          }}
        />
        {fileNote && <span className="muted small">{fileNote}</span>}
      </div>

      <div className="prestart-footer">
        <div className="format-picker" role="radiogroup" aria-label="Формат встречи">
          {FORMAT_ORDER.map((f) => (
            <label key={f} className={`format-option${format === f ? " is-selected" : ""}`}>
              <input type="radio" name="meeting-format" checked={format === f} onChange={() => void controller.setMeetingFormat(f)} />
              <span>{MEETING_FORMATS[f].label}</span>
            </label>
          ))}
        </div>
        <p className="muted small">
          {format ? `${MEETING_FORMATS[format].hint}.` : "Своя настройка захвата звука."} {autoLine}{" "}
          {!s.captureMic && !s.captureSystem ? <strong>Захват выключен — включите его в настройках.</strong> : null}
          <button type="button" className="link" onClick={onOpenSettings}>
            Настройки
          </button>
        </p>
        {room && (
          <p className="tip small">
            <strong>Совет:</strong> USB-спикерфон или микрофон в центре стола работает намного лучше встроенного микрофона ноутбука.
            {!state.voiceprint?.enrolled && (
              <>
                {" "}
                Запишите образец голоса, чтобы ваши реплики подписывались «Я»:{" "}
                <button type="button" className="link" onClick={onOpenSettings}>
                  Мой голос
                </button>
                .
              </>
            )}
          </p>
        )}
        <button
          className="btn btn-primary btn-lg"
          type="submit"
          disabled={!state.ready || state.enrolling || (!s.captureMic && !s.captureSystem)}
          title={state.enrolling ? "Идёт запись образца голоса" : undefined}
        >
          <IconRecord /> Начать запись
        </button>
      </div>
    </form>
  );
}
