import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { beforeEach, expect, test, vi } from "vitest";
import { useSelectedTemplateView } from "./useSelectedTemplateView";

const { toastMock } = vi.hoisted(() => ({ toastMock: vi.fn() }));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: toastMock, toasts: [], dismiss: vi.fn() }),
  toast: toastMock,
  useToastOnFail: () => vi.fn(),
}));

beforeEach(() => toastMock.mockClear());

const PRESET_PATH = "/api/proxy/api/library/presets/:presetId";

// A triggered preset whose webhook was detached (credential removal) is filed
// under Templates, so this view receives the nested trigger-config shape too.
const STORED_INPUTS = {
  topic: "weather",
  _node_input_mask_abc123: { repo: "owner/repo" },
};

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function renderTemplateView() {
  return renderHook(
    () =>
      useSelectedTemplateView({ templateId: "preset-1", graphId: "graph-1" }),
    { wrapper },
  );
}

function respondWithPreset() {
  return http.get(PRESET_PATH, () =>
    HttpResponse.json({
      id: "preset-1",
      name: "Watcher",
      description: "",
      inputs: STORED_INPUTS,
      credentials: {},
    }),
  );
}

test("splits a detached trigger's config out of its graph inputs", async () => {
  server.use(respondWithPreset());
  const { result } = renderTemplateView();

  await waitFor(() => expect(result.current.name).toBe("Watcher"));

  // Unsplit, the mask key sat in `inputs` with no schema to render it, so the
  // stored config disappeared from the page.
  expect(result.current.inputs).toEqual({ topic: "weather" });
  expect(result.current.triggerConfig).toEqual({ repo: "owner/repo" });
  expect(result.current.hasTriggerConfig).toBe(true);
});

test("re-nests an edited trigger config under its mask key on save", async () => {
  let patched: Record<string, any> | null = null;
  server.use(
    respondWithPreset(),
    http.patch(PRESET_PATH, async (info) => {
      patched = (await info.request.json()) as Record<string, any>;
      return HttpResponse.json({
        id: "preset-1",
        name: "Watcher",
        description: "",
        inputs: STORED_INPUTS,
        credentials: {},
      });
    }),
  );

  const { result } = renderTemplateView();
  await waitFor(() => expect(result.current.name).toBe("Watcher"));

  act(() => result.current.setTriggerConfigValue("repo", "owner/other"));
  act(() => result.current.setInputValue("topic", "sports"));
  act(() => result.current.handleSaveChanges());

  await waitFor(() => expect(patched).not.toBeNull());
  // Saving re-registers the webhook from the mask key, so a config edited here
  // has to go back under the same one.
  expect(patched!.inputs).toEqual({
    topic: "sports",
    _node_input_mask_abc123: { repo: "owner/other" },
  });
});

test("does not send inputs when only the name changed", async () => {
  let patched: Record<string, any> | null = null;
  server.use(
    respondWithPreset(),
    http.patch(PRESET_PATH, async (info) => {
      patched = (await info.request.json()) as Record<string, any>;
      return HttpResponse.json({
        id: "preset-1",
        name: "Renamed",
        description: "",
        inputs: STORED_INPUTS,
        credentials: {},
      });
    }),
  );

  const { result } = renderTemplateView();
  await waitFor(() => expect(result.current.name).toBe("Watcher"));

  act(() => result.current.setName("Renamed"));
  act(() => result.current.handleSaveChanges());

  await waitFor(() => expect(patched).not.toBeNull());
  // The split/merge must round-trip exactly, or an untouched preset would look
  // edited and needlessly re-register its webhook.
  expect(patched!.name).toBe("Renamed");
  expect(patched).not.toHaveProperty("inputs");
});

test("starts a task with the trigger config re-nested, not flattened", async () => {
  let executed: Record<string, any> | null = null;
  server.use(
    respondWithPreset(),
    http.post(`${PRESET_PATH}/execute`, async (info) => {
      executed = (await info.request.json()) as Record<string, any>;
      return HttpResponse.json({ id: "exec-1", status: "QUEUED" });
    }),
  );

  const { result } = renderTemplateView();
  await waitFor(() => expect(result.current.name).toBe("Watcher"));

  act(() => result.current.setInputValue("topic", "sports"));
  act(() => result.current.handleStartTask());

  await waitFor(() => expect(executed).not.toBeNull());
  expect(executed!.inputs).toEqual({
    topic: "sports",
    _node_input_mask_abc123: { repo: "owner/repo" },
  });
});

test("leaves a plain run-template preset untouched", async () => {
  server.use(
    http.get(PRESET_PATH, () =>
      HttpResponse.json({
        id: "preset-1",
        name: "Daily digest",
        description: "",
        inputs: { topic: "weather" },
        credentials: {},
      }),
    ),
  );

  const { result } = renderTemplateView();
  await waitFor(() => expect(result.current.name).toBe("Daily digest"));

  expect(result.current.inputs).toEqual({ topic: "weather" });
  expect(result.current.triggerConfig).toEqual({});
  expect(result.current.hasTriggerConfig).toBe(false);
});
