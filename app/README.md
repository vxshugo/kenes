# Kenes app

Tauri 2 + React 19 + TypeScript + Vite. The UI and the Claude orchestration live in `src/`,
the Tauri shell in `src-tauri/`.

```bash
pnpm install
pnpm dev      # browser at http://localhost:1420 with the scripted mock backend (?mockSpeed=3 for faster replay)
pnpm test     # vitest (LLM layer, triggers, speakers, mock backend, controller end-to-end), no network
pnpm build    # tsc + vite build
pnpm tauri dev
```

In the mock, the Claude key comes from Settings (stored in localStorage) or `VITE_ANTHROPIC_API_KEY`
in `.env.local`. Architecture and the prompt-caching layout are described in `../docs/UI_NOTES.md`,
and the Rust interface in `../docs/CONTRACT.md`.
