import type { KeyboardEvent as ReactKeyboardEvent } from "react";

export const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

/** "Ctrl" on Linux, "⌘" on macOS. */
export const MOD = isMac ? "⌘" : "Ctrl";

export function hasMod(e: KeyboardEvent | ReactKeyboardEvent): boolean {
  return isMac ? e.metaKey : e.ctrlKey;
}

export const SHORTCUTS = {
  hint: `${MOD}+Enter`,
  ask: `${MOD}+K`,
  explain: `${MOD}+Shift+E`,
  translate: `${MOD}+Shift+U`,
  recap: `${MOD}+Shift+K`,
  tabs: "Alt+1…4",
} as const;

/** Keys a global shortcut may use: what both the plugin's parser and the portal (xkb) accept. */
const SHORTCUT_KEY = /^(Key[A-Z]|Digit[0-9]|F([1-9]|1[0-9]|2[0-4])|Enter|Space|Tab|Backspace|Delete|Insert|Home|End|PageUp|PageDown|Arrow(Up|Down|Left|Right)|Minus|Equal|Comma|Period|Slash|Backslash|Semicolon|Quote|Backquote|BracketLeft|BracketRight)$/;

const MODIFIER_CODES = /^(Control|Shift|Alt|Meta|OS)(Left|Right)?$/;

type KeyLike = Pick<KeyboardEvent, "code" | "ctrlKey" | "metaKey" | "altKey" | "shiftKey">;

/**
 * An accelerator (`CommandOrControl+Alt+KeyK`) from a key press, for the shortcut recorder.
 * `mac` picks what ⌘ and Ctrl mean. Returns an error text when the combination can't be used.
 */
export function acceleratorFromEvent(e: KeyLike, mac = isMac): { accelerator: string } | { error: string } | null {
  if (MODIFIER_CODES.test(e.code)) return null; // still holding modifiers
  if (!SHORTCUT_KEY.test(e.code)) return { error: "Эту клавишу нельзя назначить." };
  const mods: string[] = [];
  const primary = mac ? e.metaKey : e.ctrlKey;
  if (primary) mods.push("CommandOrControl");
  if (mac && e.ctrlKey) mods.push("Control");
  if (e.altKey) mods.push("Alt");
  if (e.shiftKey) mods.push("Shift");
  if (!mac && e.metaKey) mods.push("Super");
  if (!mods.some((m) => m !== "Shift")) return { error: `Нужен модификатор: ${mac ? "⌘, ⌃ или ⌥" : "Ctrl, Alt или Super"}.` };
  return { accelerator: [...mods, e.code].join("+") };
}

const KEY_LABELS: Record<string, string> = {
  ArrowUp: "↑",
  ArrowDown: "↓",
  ArrowLeft: "←",
  ArrowRight: "→",
  Minus: "-",
  Equal: "=",
  Comma: ",",
  Period: ".",
  Slash: "/",
  Backslash: "\\",
  Semicolon: ";",
  Quote: "'",
  Backquote: "`",
  BracketLeft: "[",
  BracketRight: "]",
};

/** `CommandOrControl+Alt+KeyK` → `Ctrl+Alt+K` (⌘+⌥+K on macOS), for display. */
export function formatAccelerator(acc: string, mac = isMac): string {
  return acc
    .split("+")
    .map((part) => {
      switch (part) {
        case "CommandOrControl":
          return mac ? "⌘" : "Ctrl";
        case "Control":
        case "Ctrl":
          return mac ? "⌃" : "Ctrl";
        case "Alt":
          return mac ? "⌥" : "Alt";
        case "Shift":
          return mac ? "⇧" : "Shift";
        case "Super":
          return mac ? "⌘" : "Super";
        default:
          return KEY_LABELS[part] ?? part.replace(/^Key|^Digit/, "");
      }
    })
    .join("+");
}
