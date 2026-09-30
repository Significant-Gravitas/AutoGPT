import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { act, cleanup, render } from "@/tests/integrations/test-utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BuilderChatPanel } from "@/app/(platform)/build/components/BuilderChatPanel/BuilderChatPanel";
import { useBuilderChatPanel } from "@/app/(platform)/build/components/BuilderChatPanel/useBuilderChatPanel";
import { MemoryChatPanel } from "@/app/(platform)/settings/memory/components/MemoryChatPanel/MemoryChatPanel";
import { ExpertChatDrawer } from "@/app/(platform)/team/components/ExpertChatDrawer/ExpertChatDrawer";
import { useExpertChatDrawer } from "@/app/(platform)/team/components/ExpertChatDrawer/useExpertChatDrawer";
import {
  getGetV2GetPendingReviewsForChatSessionMockHandler,
  getGetV2GetPendingReviewsForExecutionMockHandler,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { server } from "@/mocks/mock-server";
import { ChatContainer } from "../components/ChatContainer/ChatContainer";
import {
  CARD_FIXTURES,
  MAIN_CHAT_CARD_TYPES,
  transcriptFor,
} from "./chatSurfaceFixtures";

vi.mock("@/app/(platform)/copilot/components/ChatInput/ChatInput", () => ({
  ChatInput: () => <textarea aria-label="Chat input" />,
}));

vi.mock(
  "@/app/(platform)/copilot/components/UsageLimits/useIsUsageLimitReached",
  () => ({ useIsUsageLimitReached: () => false }),
);

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => ({
  ...(await importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
}));

vi.mock(
  "@/app/(platform)/build/components/BuilderChatPanel/useBuilderChatPanel",
  () => ({ useBuilderChatPanel: vi.fn() }),
);

vi.mock(
  "@/app/(platform)/team/components/ExpertChatDrawer/useExpertChatDrawer",
  () => ({ useExpertChatDrawer: vi.fn() }),
);

type Messages = UIMessage<unknown, UIDataTypes, UITools>[];

const SESSION_ID = "parity-session";
const SRC = join(__dirname, "..", "..", "..", "..");

interface Surface {
  name: string;
  /** The host that renders ChatMessagesContainer, relative to src/. */
  host: string;
  mount: (messages: Messages) => void;
}

const MAIN: Surface = {
  name: "main chat",
  host: "app/(platform)/copilot/components/ChatContainer/ChatContainer.tsx",
  mount: (messages) =>
    render(
      <ChatContainer
        messages={messages}
        status="ready"
        error={undefined}
        sessionId={SESSION_ID}
        isLoadingSession={false}
        isCreatingSession={false}
        onCreateSession={vi.fn()}
        onSend={vi.fn()}
        onStop={vi.fn()}
      />,
    ),
};

/** Every chat that isn't the full-screen one. Each must render every card
 *  exactly as the main chat does. */
const OTHER_SURFACES: Surface[] = [
  {
    name: "memory settings chat",
    host: "app/(platform)/settings/memory/components/MemoryChatPanel/MemoryChatPanel.tsx",
    mount: (messages) =>
      render(
        <MemoryChatPanel
          scopeName="Otto"
          isOpen
          isStarting={false}
          startError={false}
          sessionId={SESSION_ID}
          messages={messages}
          status="ready"
          error={undefined}
          queuedMessages={[]}
          onSend={async () => {}}
          onStop={vi.fn()}
          onRetry={vi.fn()}
          onClose={vi.fn()}
        />,
      ),
  },
  {
    name: "builder chat",
    host: "app/(platform)/build/components/BuilderChatPanel/BuilderChatPanel.tsx",
    mount: (messages) => {
      vi.mocked(useBuilderChatPanel).mockReturnValue({
        isOpen: true,
        handleToggle: vi.fn(),
        sessionId: SESSION_ID,
        messages,
        status: "ready",
        error: undefined,
        stop: vi.fn(),
        onSend: vi.fn(),
        queuedMessages: [],
        isBootstrapping: false,
        revertTargetVersion: null,
        handleRevert: vi.fn(),
        bindError: null,
        bootstrapError: null,
        retryBind: vi.fn(),
        retryBootstrap: vi.fn(),
      } as unknown as ReturnType<typeof useBuilderChatPanel>);
      render(<BuilderChatPanel />);
    },
  },
  {
    name: "team expert chat drawer",
    host: "app/(platform)/team/components/ExpertChatDrawer/ExpertChatDrawer.tsx",
    mount: (messages) => {
      vi.mocked(useExpertChatDrawer).mockReturnValue({
        sessionId: SESSION_ID,
        startNewThread: vi.fn(),
        messages,
        status: "ready",
        error: undefined,
        stop: vi.fn(),
        onSend: vi.fn(),
        onActionSend: vi.fn(),
        queuedMessages: [],
        isResolvingSession: false,
        isLoadingSession: false,
        isCreating: false,
        suppressOnboarding: true,
      } as unknown as ReturnType<typeof useExpertChatDrawer>);
      render(
        <ExpertChatDrawer
          target={{
            expertId: null,
            name: "Otto",
            role: "autopilot",
            avatarUrl: null,
          }}
          onClose={vi.fn()}
        />,
      );
    },
  },
];

/** Hosts that render the thread for an anonymous reader. They never show a
 *  live card, so they are out of the parity check on purpose. */
const READ_ONLY_HOSTS = ["app/(no-navbar)/share/chat/[token]/page.tsx"];

const KEPT_ATTRIBUTES = new Set([
  "role",
  "aria-label",
  "aria-checked",
  "aria-expanded",
  "aria-hidden",
  "aria-pressed",
  "aria-selected",
  "href",
  "type",
  "disabled",
  "name",
  "value",
  "alt",
  "placeholder",
  "src",
]);

/** What the reader gets from the thread: its elements, text, roles and
 *  controls, without styling or generated ids — a compact panel is allowed
 *  to look smaller, not to render something else. */
function threadSignature(): string {
  const logs = document.body.querySelectorAll('[role="log"]');
  expect(logs.length).toBe(1);
  const clone = logs[0].cloneNode(true) as Element;
  for (const element of [clone, ...clone.querySelectorAll("*")]) {
    for (const attribute of [...element.attributes]) {
      if (!KEPT_ATTRIBUTES.has(attribute.name))
        element.removeAttribute(attribute.name);
    }
  }
  return clone.outerHTML;
}

async function settle() {
  for (let i = 0; i < 3; i++) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
}

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory())
      return name === "__tests__" ? [] : sourceFiles(path);
    return /\.tsx$/.test(name) && !/\.(test|stories)\.tsx$/.test(name)
      ? [path]
      : [];
  });
}

function filesRendering(tag: string): string[] {
  return sourceFiles(join(SRC, "app"))
    .filter((path) => readFileSync(path, "utf8").includes(`<${tag}`))
    .map((path) => relative(SRC, path))
    .sort();
}

class MockResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", MockResizeObserver);
  // Generated mocks answer with random data, which would differ from one
  // surface's render to the next. Nothing is waiting on review here.
  server.use(
    getGetV2GetPendingReviewsForChatSessionMockHandler([]),
    getGetV2GetPendingReviewsForExecutionMockHandler([]),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

// Each card mounts four full chat surfaces and the coverage checks read every
// source file under app/ — well past the 5s default on a loaded runner.
describe("chat surface parity", { timeout: 30_000 }, () => {
  it("covers every chat that renders the shared thread", () => {
    expect(filesRendering("ChatMessagesContainer")).toEqual(
      [MAIN, ...OTHER_SURFACES]
        .map((surface) => surface.host)
        .concat(READ_ONLY_HOSTS)
        .sort(),
    );
  });

  it("keeps a single message renderer, with no second copy of the card switch", () => {
    const renderer = "app/(platform)/copilot/components/ChatMessagesContainer/";
    for (const tag of ["MessagePartRenderer", "ChainMessageParts"]) {
      expect(
        filesRendering(tag).filter((path) => !path.startsWith(renderer)),
      ).toEqual([]);
    }
  });

  it("has a fixture for every card the main chat renders", () => {
    expect(
      MAIN_CHAT_CARD_TYPES.filter((type) => !(type in CARD_FIXTURES)),
    ).toEqual([]);
  });

  it.each(MAIN_CHAT_CARD_TYPES)(
    "renders %s on every other chat as the main chat does",
    async (type) => {
      const fixture = CARD_FIXTURES[type];
      expect(fixture, `no fixture for ${type}`).toBeDefined();
      const messages = transcriptFor(fixture);

      MAIN.mount(messages);
      await settle();
      const expected = threadSignature();
      expect(
        document.body.querySelector('[role="log"]')?.textContent,
      ).toContain(fixture.marker);
      cleanup();

      for (const surface of OTHER_SURFACES) {
        surface.mount(messages);
        await settle();
        expect(threadSignature(), `${type} in the ${surface.name}`).toBe(
          expected,
        );
        cleanup();
      }
    },
  );
});
