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
