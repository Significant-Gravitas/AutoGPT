# OpenUI experiment

An opt-in AutoGPT workspace powered by the actual [OpenUI React runtime](https://www.openui.com/docs/api-reference/react-lang). The model composes a bounded catalog of AutoGPT components, rather than generating executable HTML or JavaScript.

## Try it

Use Node 24 and the repository's pnpm version. In `autogpt_platform/frontend`:

```bash
pnpm install
pnpm generate:api
NEXT_PUBLIC_OPENUI_EXPERIMENT=true pnpm exec next dev --turbo --port 3000
```

Open **http://localhost:3000/tour/openui** for the standalone sample experience; no backend, account, or model key is required. **http://localhost:3000/copilot/openui** uses the existing platform shell and authentication. The sidebar shows **UI Lab** when the experiment is enabled. Restart after changing the flag; it is a Next.js build-time public variable. Both pages and the generation endpoint are disabled by default.

The samples cover agent performance → failure investigation, lead research → outreach planning, and an editable campaign brief → a personalized checklist. Samples replay prepared OpenUI programs with fictional data. They are explicitly labeled and do not call an LLM or execute agents. Unsupported sample prompts produce an explanation, not a simulated AI answer.

## Live generation

Add these server-only settings to your gitignored `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_OPENUI_EXPERIMENT=true
OPENUI_BASE_URL=https://your-provider.example/v1
OPENUI_MODEL=your-model-id
OPENUI_API_KEY=your-provider-key
```

Use an OpenAI-compatible **Chat Completions** endpoint, including an appropriate model ID for that provider. Restart, sign in to the platform, open `/copilot/openui`, and choose **Live AI**. Keys stay on the server. The public sample page cannot submit live requests without a valid platform session. Live generation uses the configured provider directly and is **not integrated with AutoGPT credit billing**; keep this opt-in experiment limited to a development environment.

Live requests include the current workspace and submitted form values. The model receives a system prompt generated from the same schemas as the renderer. It returns a complete replacement workspace; text deltas render progressively. Cancel, failed streams, invalid output, and truncated responses retain the last completed workspace. Charts, sortable/filterable tables, and checklists are local interactions. Generated buttons and forms request another view; they cannot execute tools, send messages, or modify platform records.

**Interactive / Text response / Source** compares the same result in three presentations. **Export** downloads the current OpenUI program. State is intentionally ephemeral: refreshing resets the lab, and checklist ticks are not persisted. Live generation has no account-data tools; provide your own data in the prompt, or ask for explicitly hypothetical examples.

## Implementation and checks

- `src/lib/openui/catalog.ts`: shared Zod 4 catalog and model instructions; no React dependency on the server.
- `src/components/organisms/OpenUI/`: actual OpenUI renderer backed by AutoGPT primitives and Recharts.
- `src/app/api/openui/`: authenticated, bounded, cancellable streaming generation through the AI SDK.
- This route sits under the already-protected `/copilot` prefix, covered by the existing middleware matcher and auth helper. No authentication bypass is introduced for the experiment.
- OpenUI is pinned at `0.3.0`, which satisfies the repository's minimum package release age.
- The Next.js bundlers alias OpenUI's automatic development widget to a local no-op. The upstream development entry otherwise downloads executable widget code from a CDN and injects promotional UI. The renderer and our source inspector remain fully functional.

```bash
pnpm format
pnpm lint
pnpm types
pnpm test:unit
```

Focused tests live in `copilot/openui/__tests__` and `api/openui/__tests__`.

This experiment is a separate surface inside AutoGPT. It does not change the existing Copilot prompt, stream protocol, or message renderer. The shared catalog and renderer are the integration point for a future Copilot tool result.
