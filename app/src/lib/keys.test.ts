import { describe, expect, it } from "vitest";
import { DEFAULT_HOTKEYS, isAccelerator, normalizeSettings } from "../types";
import { acceleratorFromEvent, formatAccelerator } from "./keys";

const press = (code: string, mods: Partial<{ ctrlKey: boolean; metaKey: boolean; altKey: boolean; shiftKey: boolean }> = {}) => ({
  code,
  ctrlKey: false,
  metaKey: false,
  altKey: false,
  shiftKey: false,
  ...mods,
});

describe("global shortcut accelerators", () => {
  it("records a combination as a portable accelerator", () => {
    expect(acceleratorFromEvent(press("Enter", { ctrlKey: true, altKey: true }), false)).toEqual({ accelerator: "CommandOrControl+Alt+Enter" });
    expect(acceleratorFromEvent(press("KeyK", { metaKey: true, altKey: true }), true)).toEqual({ accelerator: "CommandOrControl+Alt+KeyK" });
    expect(acceleratorFromEvent(press("KeyK", { ctrlKey: true, metaKey: true }), true)).toEqual({ accelerator: "CommandOrControl+Control+KeyK" });
    expect(acceleratorFromEvent(press("Digit1", { metaKey: true, shiftKey: true }), false)).toEqual({ accelerator: "Shift+Super+Digit1" });
  });

  it("waits while only modifiers are held and rejects unusable combinations", () => {
    expect(acceleratorFromEvent(press("ControlLeft", { ctrlKey: true }), false)).toBeNull();
    expect(acceleratorFromEvent(press("KeyK"), false)).toMatchObject({ error: expect.stringContaining("модификатор") });
    expect(acceleratorFromEvent(press("KeyK", { shiftKey: true }), false)).toMatchObject({ error: expect.any(String) });
    expect(acceleratorFromEvent(press("IntlBackslash", { ctrlKey: true }), false)).toMatchObject({ error: expect.any(String) });
  });

  it("formats for display per platform", () => {
    expect(formatAccelerator("CommandOrControl+Alt+Enter", false)).toBe("Ctrl+Alt+Enter");
    expect(formatAccelerator("CommandOrControl+Alt+KeyK", true)).toBe("⌘+⌥+K");
    expect(formatAccelerator("Shift+Super+ArrowUp", false)).toBe("Shift+Super+↑");
    expect(formatAccelerator("CommandOrControl+Alt+Digit1", false)).toBe("Ctrl+Alt+1");
  });

  it("settings keep valid accelerators and fall back to the defaults otherwise", () => {
    expect(isAccelerator("CommandOrControl+Alt+KeyP")).toBe(true);
    expect(isAccelerator("Shift+KeyP")).toBe(false);
    expect(isAccelerator("KeyP")).toBe(false);
    expect(isAccelerator("Ctrl+Alt+")).toBe(false);
    const s = normalizeSettings({ hotkeys: { hint: "Alt+Super+KeyH", recap: "KeyR", toggle: 5 } });
    expect(s.hotkeys).toEqual({ hint: "Alt+Super+KeyH", recap: DEFAULT_HOTKEYS.recap, toggle: DEFAULT_HOTKEYS.toggle });
  });
});
