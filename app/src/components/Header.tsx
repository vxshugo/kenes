import type { ControllerState } from "../session/controller";
import { Elapsed } from "./Elapsed";
import { IconRecord, IconStop } from "./Icons";
import { LevelMeter } from "./LevelMeter";

type Props = {
  state: ControllerState;
  onStart: () => void;
  onStop: () => void;
  onNew: () => void;
};

function statusView(s: ControllerState): { text: string; tone: "idle" | "busy" | "live" | "error" | "done"; title?: string } {
  switch (s.phase) {
    case "idle":
      return { text: "Готов", tone: "idle" };
    case "starting":
      return { text: "Запуск…", tone: "busy" };
    case "loading": {
      const p = s.modelProgress;
      const pct = p ? ` ${Math.round(Math.min(1, Math.max(0, p.progress)) * 100)}%` : "";
      return { text: `Загрузка модели${pct}`, tone: "busy", title: s.statusMessage ?? p?.model };
    }
    case "running":
      return { text: "Запись", tone: "live", title: s.statusMessage ?? undefined };
    case "stopping":
      return { text: "Остановка…", tone: "busy" };
    case "stopped":
      return { text: "Завершено", tone: "done" };
    case "error":
      return { text: "Ошибка", tone: "error", title: s.error ?? undefined };
  }
}

export function Header({ state, onStart, onStop, onNew }: Props) {
  const status = statusView(state);
  const active = state.phase === "starting" || state.phase === "loading" || state.phase === "running";
  const title = state.title || (state.phase === "idle" ? "Новая встреча" : "Встреча");
  const idle = state.phase === "idle";
  const inCall = idle ? state.settings.captureSystem : state.captureSystem;
  const micOn = idle ? state.settings.captureMic : state.captureMic;
  const micMode = idle ? state.settings.micMode : state.micMode;
  const progress = state.phase === "loading" && state.modelProgress ? Math.min(1, Math.max(0, state.modelProgress.progress)) : null;

  return (
    <header className="app-header">
      <div className="header-row">
        <div className="brand" aria-hidden="true">
          K
        </div>
        <h1 className="session-title" title={title}>
          {title}
        </h1>
        <span className={`status status-${status.tone}`} title={status.title} role="status">
          <span className="status-dot" aria-hidden="true" />
          {status.text}
        </span>
        {active || state.phase === "stopping" ? (
          <button className="btn btn-stop" onClick={onStop} disabled={state.phase === "stopping"} title="Остановить запись и подвести итоги">
            <IconStop /> Стоп
          </button>
        ) : state.phase === "error" ? (
          <button className="btn btn-stop" onClick={onStop} title="Завершить сессию">
            <IconStop /> Завершить
          </button>
        ) : state.phase === "stopped" ? (
          <button className="btn btn-primary" onClick={onNew} title="Подготовить новую встречу">
            Новая
          </button>
        ) : (
          <button
            className="btn btn-primary"
            onClick={onStart}
            disabled={!state.ready || !!state.initError || state.enrolling || (!state.settings.captureMic && !state.settings.captureSystem)}
            title={state.enrolling ? "Идёт запись образца голоса" : "Начать запись"}
          >
            <IconRecord /> Старт
          </button>
        )}
      </div>
      <div className="header-row header-sub">
        <Elapsed startedAt={state.startedAt} endedAt={state.endedAt} />
        <div className="meters">
          <LevelMeter source="mic" label={micMode === "me" ? "Я" : "Зал"} disabled={!micOn && !state.enrolling} />
          <LevelMeter source="system" label="Звонок" disabled={!inCall} />
        </div>
      </div>
      {progress !== null && (
        <div className="progress" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="Загрузка модели распознавания">
          <span style={{ transform: `scaleX(${progress})` }} />
        </div>
      )}
      {state.phase === "error" && state.error && <div className="banner banner-error">{state.error}</div>}
    </header>
  );
}
