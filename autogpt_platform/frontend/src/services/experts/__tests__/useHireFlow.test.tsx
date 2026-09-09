import {
  getHireExpertMockHandler,
  getUpdateExpertSoulMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { markHireStarted } from "@/services/experts/hire-timing";
import { useHireFlow } from "@/services/experts/useHireFlow";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, screen, waitFor } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { type ReactNode, useState } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";

const captureMock = vi.hoisted(() => vi.fn());
const pushMock = vi.hoisted(() => vi.fn());

vi.mock("posthog-js", () => ({ default: { capture: captureMock } }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/marketplace",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const mariaTemplate: Expert = {
  id: "template-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: "A senior marketing strategist.",
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [],
  is_template: true,
  source_template_id: null,
  is_archived: false,
  workflows: [],
  voice_samples: [
    { label: "Punchy and bold", text: "Stop guessing what your buyers want." },
    {
      label: "Warm and story-led",
      text: "Every campaign starts with a person.",
    },
  ],
};

const hiredMaria: Expert = {
  ...mariaTemplate,
  id: "expert-maria",
  is_template: false,
  source_template_id: "template-maria",
};

function Wrapper({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { retry: false },
          mutations: { retry: false },
        },
      }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <Toaster />
    </QueryClientProvider>
  );
}

function renderHireFlow(expert: Expert = mariaTemplate) {
  return renderHook(() => useHireFlow(expert), { wrapper: Wrapper });
}

function capturedEvent(name: string) {
  return captureMock.mock.calls.find((call) => call[0] === name);
}

describe("useHireFlow", () => {
  beforeEach(() => {
    captureMock.mockReset();
    pushMock.mockReset();
    window.sessionStorage.clear();
    server.use(
      getHireExpertMockHandler({ expert: hiredMaria, failed_preloads: [] }),
      getUpdateExpertSoulMockHandler(hiredMaria),
    );
  });

  test("walks hire → voice pick → saved voice and hands off to the thread", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    markHireStarted("template-maria");
    vi.advanceTimersByTime(1500);
    const { result } = renderHireFlow();

    await act(async () => {
      await result.current.hire();
    });
    expect(result.current.isVoicePickOpen).toBe(true);
    expect(result.current.hireResult?.expert.id).toBe("expert-maria");

    await act(async () => {
      await result.current.pickVoice({ choice: "a" });
    });

    expect(result.current.isVoicePickOpen).toBe(false);
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?expertId=expert-maria&kickoff=1",
    );
    const completed = capturedEvent("hire_flow_completed")?.[1] as {
      voice_picked: boolean;
      elapsed_ms: number | null;
    };
    expect(completed.voice_picked).toBe(true);
    expect(completed.elapsed_ms).toBeGreaterThanOrEqual(1500);
    // The mark is consumed, so a second finish cannot report a stale span.
    expect(
      window.sessionStorage.getItem("autogpt:hire-started:template-maria"),
    ).toBeNull();
    vi.useRealTimers();
  });

  test("skipping the voice pick still celebrates the hire", async () => {
    const { result } = renderHireFlow();

    await act(async () => {
      await result.current.hire();
    });
    await act(async () => {
      result.current.skipVoice();
    });

    expect(result.current.isVoicePickOpen).toBe(false);
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?expertId=expert-maria&kickoff=1",
    );
    expect(capturedEvent("hire_flow_completed")?.[1]).toMatchObject({
      voice_picked: false,
      elapsed_ms: null,
    });
  });

  test("finishes straight away when the persona ships no writing samples", async () => {
    const { result } = renderHireFlow({ ...mariaTemplate, voice_samples: [] });

    await act(async () => {
      await result.current.hire();
    });

    expect(result.current.isVoicePickOpen).toBe(false);
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?expertId=expert-maria&kickoff=1",
    );
  });

  test("reports a dismissed voice pick as abandoned, and still celebrates", async () => {
    const { result } = renderHireFlow();

    await act(async () => {
      await result.current.hire();
    });
    await act(async () => {
      result.current.dismissVoicePick();
    });

    expect(capturedEvent("hire_flow_abandoned")?.[1]).toMatchObject({
      template_id: "template-maria",
      stage: "voice",
    });
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?expertId=expert-maria&kickoff=1",
    );
  });

  test("keeps the picker open when saving the voice fails", async () => {
    server.use(
      http.patch("/api/proxy/api/experts/:expertId/soul", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );
    const { result } = renderHireFlow();

    await act(async () => {
      await result.current.hire();
    });
    await act(async () => {
      await result.current.pickVoice({ choice: "a" });
    });

    expect(await screen.findByText("Couldn't save the voice")).toBeDefined();
    expect(result.current.isVoicePickOpen).toBe(true);
    expect(pushMock).not.toHaveBeenCalled();
  });

  test("says so and stays put when the hire itself fails", async () => {
    server.use(
      http.post("/api/proxy/api/experts", () =>
        HttpResponse.json({ detail: "boom" }, { status: 503 }),
      ),
    );
    const { result } = renderHireFlow();

    await act(async () => {
      await result.current.hire();
    });

    await waitFor(() =>
      expect(screen.getByText("Couldn't hire Maria")).toBeDefined(),
    );
    expect(result.current.isVoicePickOpen).toBe(false);
    expect(pushMock).not.toHaveBeenCalled();
  });
});
