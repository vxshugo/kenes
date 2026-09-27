import { useState } from "react";
import { acceleratorFromEvent, formatAccelerator } from "../lib/keys";
import type { ControllerState } from "../session/controller";
import { controller } from "../session/useController";
import { DEFAULT_HOTKEYS } from "../types";
import type { HotkeyAction, HotkeyBindings, HotkeyStatus, Settings } from "../types";

const ACTIONS: Array<{ action: HotkeyAction; label: string }> = [
  { action: "hint", label: "Что ответить?" },
  { action: "recap", label: "Кратко: 5 мин" },
  { action: "toggle", label: "Показать / скрыть окно" },
];

/** Click, then press the combination. Esc cancels. */
function ShortcutInput({ value, onChange, label, disabled }: { value: string; onChange: (v: string) => void; label: string; disabled?: boolean }) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <span className="shortcut">
      <button
        type="button"
        className={`shortcut-input${recording ? " is-recording" : ""}`}
        disabled={disabled}
        aria-label={`${label}: ${formatAccelerator(value)}. Нажмите, чтобы изменить`}
        onClick={() => {
          setError(null);
          setRecording(true);
        }}
        onBlur={() => setRecording(false)}
        onKeyDown={(e) => {
          if (!recording) return;
          // Keep the app's own shortcuts from firing while recording.
          e.preventDefault();
          e.stopPropagation();
          if (e.key === "Escape" && !e.ctrlKey && !e.metaKey && !e.altKey) {
            setRecording(false);
            return;
          }
          const r = acceleratorFromEvent(e.nativeEvent);
          if (!r) return;
          if ("error" in r) {
            setError(r.error);
            return;
          }
          setError(null);
          setRecording(false);
          onChange(r.accelerator);
        }}
      >
        {recording ? "Нажмите сочетание…" : <kbd>{formatAccelerator(value)}</kbd>}
      </button>
      {error && <span className="card-error small">{error}</span>}
    </span>
  );
}

function statusLine(st: HotkeyStatus | null, enabled: boolean): { text: string; tone: "ok" | "warn" | "muted" } | null {
  if (!st) return enabled ? { text: "Проверяю…", tone: "muted" } : null;
  switch (st.state) {
    case "active":
      return { text: st.message ?? "Работают во всей системе.", tone: st.message ? "warn" : "ok" };
    case "pending":
      return { text: st.message ?? "Ждём подтверждения в системном окне…", tone: "muted" };
    case "off":
      return null;
    default:
      return { text: st.message ?? "Недоступны.", tone: "warn" };
  }
}

type Props = {
  state: ControllerState;
  draft: Settings;
  set: <K extends keyof Settings>(k: K, v: Settings[K]) => void;
};

/** «Окно и горячие клавиши»: always-on-top on GNOME, system-wide shortcuts. */
export function HotkeysSection({ state, draft, set }: Props) {
  const platform = state.platform;
  const st = state.hotkeys;
  const portal = (st?.backend ?? platform?.hotkeyBackend) === "portal";
  const gnomeWayland = !!platform?.gnome && platform.sessionType === "wayland";
  const line = statusLine(st, state.settings.globalHotkeys);
  const saved = state.settings;
  const unsaved = saved.globalHotkeys !== draft.globalHotkeys || JSON.stringify(saved.hotkeys) !== JSON.stringify(draft.hotkeys);
  const systemTrigger = (a: HotkeyAction) => st?.bindings.find((b) => b.action === a)?.trigger ?? null;
  const setKey = (a: HotkeyAction, v: string) => set("hotkeys", { ...draft.hotkeys, [a]: v } as HotkeyBindings);

  return (
    <fieldset>
      <legend>Окно и горячие клавиши</legend>
      {gnomeWayland && (
        <label className="check">
          <input type="checkbox" checked={draft.gnomeAlwaysOnTop} onChange={(e) => set("gnomeAlwaysOnTop", e.target.checked)} />
          <span>
            Поверх всех окон в GNOME
            <span className="field-hint">
              GNOME на Wayland не даёт приложениям держать окно поверх других. С этой настройкой Kenes работает через XWayland, и
              закрепление срабатывает. Минус: при дробном масштабе экрана (125–175 %) окно рисуется в 200 % и выглядит крупнее.
              Применяется после перезапуска Kenes (сейчас: {platform?.x11Forced ? "XWayland" : "Wayland"}). Без неё закрепить окно
              можно вручную: Alt+Пробел → «Поверх всех окон».
            </span>
          </span>
        </label>
      )}
      <label className="check">
        <input type="checkbox" checked={draft.globalHotkeys} onChange={(e) => set("globalHotkeys", e.target.checked)} />
        <span>
          Глобальные горячие клавиши
          <span className="field-hint">работают, даже когда активно окно звонка; сочетания внутри окна Kenes работают всегда</span>
        </span>
      </label>
      <div className="hotkey-list" role="group" aria-label="Глобальные сочетания">
        {ACTIONS.map(({ action, label }) => {
          const sys = portal ? systemTrigger(action) : null;
          return (
            <div key={action} className="hotkey-row">
              <span className="hotkey-label">{label}</span>
              <ShortcutInput value={draft.hotkeys[action]} label={label} disabled={!draft.globalHotkeys} onChange={(v) => setKey(action, v)} />
              {portal && st?.state === "active" && (
                <span className="field-hint hotkey-system">{sys ? <>в системе: <kbd>{sys}</kbd></> : "в системе не назначено"}</span>
              )}
            </div>
          );
        })}
      </div>
      {portal && (
        <span className="field-hint">
          В GNOME сочетания назначает система: при включении она покажет своё окно, где можно подтвердить или выбрать другие клавиши.
          Заданные здесь — только предложение. Поменять потом: «Параметры → Приложения → Kenes».
        </span>
      )}
      <div className="input-row">
        {line && <span className={`small ${line.tone === "ok" ? "ok" : line.tone === "warn" ? "warn-text" : "muted"}`}>{line.text}</span>}
        <span className="spacer" />
        {JSON.stringify(draft.hotkeys) !== JSON.stringify(DEFAULT_HOTKEYS) && (
          <button type="button" className="btn btn-ghost btn-sm" disabled={!draft.globalHotkeys} onClick={() => set("hotkeys", DEFAULT_HOTKEYS)}>
            По умолчанию
          </button>
        )}
        {portal && draft.globalHotkeys && !unsaved && st && ["cancelled", "error"].includes(st.state) && (
          <button type="button" className="btn btn-sm" onClick={() => controller.retryHotkeys()}>
            Назначить сочетания
          </button>
        )}
      </div>
    </fieldset>
  );
}
