import {
  REPLAY_PRIVACY_OPTIONS,
  UNMASKED_IN_REPLAY,
} from "@/lib/replay-privacy";
import { createRequire } from "node:module";
import { createElement, type ReactNode } from "react";

/**
 * Records the page with Sentry's real session replay (its bundled rrweb) and
 * our privacy options, and returns what Sentry would store: every text and
 * every attribute value, except styling and ids. Masked text comes back as
 * `*`s, so a string that is in the result was recorded as written.
 *
 * Tests assert on this rather than on class names, because Sentry decides by
 * the nearest marker up the tree, which a class check can't see.
 */
export async function recordReplayText() {
  const snapshot = await recordFullSnapshot();
  return collectReadableText(snapshot).join("\n");
}

/**
 * An unmasked wrapper around arbitrary content, which app code must never be
 * (ESLint stops it). Tests use it to check what stays masked inside one.
 */
export function UnmaskedContainer({ children }: { children?: ReactNode }) {
  return createElement("div", { className: UNMASKED_IN_REPLAY }, children);
}

// Styling, ids and references to other elements by id carry no user content.
// Everything else is collected, so a new attribute that does is caught.
const SKIPPED_ATTRIBUTES = new Set([
  "class",
  "style",
  "id",
  "src",
  "srcset",
  "for",
  "aria-controls",
  "aria-describedby",
  "aria-labelledby",
  "aria-owns",
  "aria-activedescendant",
]);

interface SerializedNode {
  type: number;
  textContent?: string;
  isStyle?: boolean;
  attributes?: Record<string, unknown>;
  childNodes?: SerializedNode[];
}

interface ReplayIntegrationInternals {
  _setup(client: unknown): void;
  _initialize(client: unknown): void;
  _replay?: { eventBuffer?: { events?: { type: number; data: unknown }[] } };
  stop(): Promise<void>;
}

interface SentryBrowser {
  init(options: Record<string, unknown>): void;
  getClient(): {
    getIntegrationByName(name: string): ReplayIntegrationInternals;
  };
  replayIntegration(options: Record<string, unknown>): unknown;
  close(timeout?: number): Promise<boolean>;
}

const FULL_SNAPSHOT = 2;

// Vitest resolves @sentry/nextjs to its server build, which has no replay,
// so the browser SDK is loaded the way @sentry/nextjs's client build gets it.
async function loadSentryBrowser(): Promise<SentryBrowser> {
  const fromApp = createRequire(import.meta.url);
  const fromNext = createRequire(
    fromApp.resolve("@sentry/nextjs/package.json"),
  );
  const fromReact = createRequire(
    fromNext.resolve("@sentry/react/package.json"),
  );
  return import(fromReact.resolve("@sentry/browser"));
}

async function recordFullSnapshot() {
  const Sentry = await loadSentryBrowser();
  // rrweb polyfills DOMTokenList.forEach, which happy-dom doesn't expose.
  const win = window as unknown as { DOMTokenList?: unknown };
  win.DOMTokenList ??= function DOMTokenList() {};

  Sentry.init({
    dsn: "https://public@example.invalid/1",
    transport: () => ({ send: async () => ({}), flush: async () => true }),
    replaysSessionSampleRate: 1,
    integrations: [
      Sentry.replayIntegration({
        ...REPLAY_PRIVACY_OPTIONS,
        useCompression: false,
      }),
    ],
  });
  const client = Sentry.getClient();
  const replay = client.getIntegrationByName("Replay");
  // Under Node, Sentry skips replay setup as "not a browser"; run it by hand.
  replay._setup(client);
  replay._initialize(client);

  const events = replay._replay?.eventBuffer?.events ?? [];
  const snapshot = events.find((event) => event.type === FULL_SNAPSHOT);
  await replay.stop();
  await Sentry.close(0);
  if (!snapshot) throw new Error("Sentry replay recorded no snapshot");
  return (snapshot.data as { node: SerializedNode }).node;
}

function collectReadableText(node: SerializedNode, out: string[] = []) {
  if (node.isStyle) return out;
  if (node.textContent?.trim()) out.push(node.textContent.trim());
  for (const [name, value] of Object.entries(node.attributes ?? {})) {
    if (SKIPPED_ATTRIBUTES.has(name) || name.startsWith("rr_")) continue;
    if (name === "_cssText") continue;
    if (typeof value === "string" && value.trim()) out.push(value.trim());
  }
  node.childNodes?.forEach((child) => collectReadableText(child, out));
  return out;
}
