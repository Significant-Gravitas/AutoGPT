import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { beforeEach, expect, test, vi } from "vitest";
import { useSelectedTriggerView } from "./useSelectedTriggerView";

const { toastMock } = vi.hoisted(() => ({ toastMock: vi.fn() }));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: toastMock, toasts: [], dismiss: vi.fn() }),
  toast: toastMock,
  useToastOnFail: () => vi.fn(),
}));

beforeEach(() => toastMock.mockClear());

const PRESET_PATH = "/api/proxy/api/library/presets/:presetId";

// Stored shape for a triggered preset that also has a regular input node.
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

function renderTriggerView() {
  return renderHook(
    () => useSelectedTriggerView({ triggerId: "preset-1", graphId: "graph-1" }),
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

test("shows the stored trigger config and graph inputs as separate groups", async () => {
  server.use(respondWithPreset());
  const { result } = renderTriggerView();

  await waitFor(() => expect(result.current.name).toBe("Watcher"));

  // Previously the trigger config stayed nested, so these fields rendered empty.
  expect(result.current.triggerConfig).toEqual({ repo: "owner/repo" });
  expect(result.current.inputs).toEqual({ topic: "weather" });
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

  const { result } = renderTriggerView();
  await waitFor(() => expect(result.current.name).toBe("Watcher"));

  act(() => result.current.setTriggerConfigValue("repo", "owner/other"));
  act(() => result.current.setInputValue("topic", "sports"));
  act(() => result.current.handleSaveChanges());

  await waitFor(() => expect(patched).not.toBeNull());
  // Both edits survive, and the trigger config goes back under its mask key —
  // update_triggered_preset reads it from there to re-register the webhook.
  expect(patched!.inputs).toEqual({
    topic: "sports",
    _node_input_mask_abc123: { repo: "owner/other" },
  });
});

test("does not send inputs when nothing changed", async () => {
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

  const { result } = renderTriggerView();
  await waitFor(() => expect(result.current.name).toBe("Watcher"));

  act(() => result.current.setName("Renamed"));
  act(() => result.current.handleSaveChanges());

  await waitFor(() => expect(patched).not.toBeNull());
  // The split/merge must be lossless, or an untouched preset would look edited
  // and needlessly re-register its webhook.
  expect(patched!.name).toBe("Renamed");
  expect(patched).not.toHaveProperty("inputs");
});

test("refuses a trigger edit on a preset with no mask key", async () => {
  // The cohort the SQL data migration skips: inputs are still flat, so there is
  // no mask key to write the edited trigger config back into.
  let patched = false;
  server.use(
    http.get(PRESET_PATH, () =>
      HttpResponse.json({
        id: "preset-1",
        name: "Watcher",
        description: "",
        inputs: { repo: "owner/repo" },
        credentials: {},
      }),
    ),
    http.patch(PRESET_PATH, () => {
      patched = true;
      return HttpResponse.json({
        id: "preset-1",
        name: "Watcher",
        description: "",
        inputs: { repo: "owner/repo" },
        credentials: {},
      });
    }),
  );

  const { result } = renderTriggerView();
  await waitFor(() => expect(result.current.name).toBe("Watcher"));

  act(() => result.current.setTriggerConfigValue("repo", "owner/other"));
  act(() => result.current.handleSaveChanges());

  // Saving used to drop the edit and still toast "Trigger updated".
  await waitFor(() =>
    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    ),
  );
  expect(patched).toBe(false);
});
