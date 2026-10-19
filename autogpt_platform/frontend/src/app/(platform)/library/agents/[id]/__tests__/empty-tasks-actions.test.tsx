import { getGetV1GetGraphVersionMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getDeleteV2DeleteLibraryAgentMockHandler,
  getDeleteV2DeleteLibraryAgentMockHandler422,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, test, vi } from "vitest";
import { EmptyTasks } from "../components/NewAgentLibraryView/components/other/EmptyTasks";

const toast = vi.fn();
const push = vi.fn();
const exportAsJSONFile = vi.fn();

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast }),
}));

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useRouter: () => ({ push, replace: vi.fn(), refresh: vi.fn() }),
  usePathname: () => "/library/agents/library-agent",
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/utils")>()),
  exportAsJSONFile: (...args: unknown[]) => exportAsJSONFile(...args),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "u@example.com" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

const agent = getGetV2GetLibraryAgentResponseMock({
  id: "library-agent",
  graph_id: "graph-1",
  graph_version: 2,
  name: "Morning digest",
  marketplace_listing: null,
});

beforeEach(() => {
  toast.mockClear();
  push.mockClear();
  exportAsJSONFile.mockClear();
});

async function confirmDelete() {
  await userEvent.click(
    await screen.findByRole("button", { name: "Delete workflow" }),
  );
  await userEvent.click(
    await screen.findByRole("button", { name: "Delete Workflow" }),
  );
}

test("deleting the workflow says so and returns to the library", async () => {
  server.use(
    getDeleteV2DeleteLibraryAgentMockHandler(
      () => new Response(null, { status: 204 }),
    ),
  );
  render(<EmptyTasks agent={agent} />);

  await confirmDelete();

  await waitFor(() =>
    expect(toast).toHaveBeenCalledWith({ title: "Workflow deleted" }),
  );
  expect(push).toHaveBeenCalledWith("/library");
});

test("a failed delete says the workflow could not be deleted", async () => {
  server.use(getDeleteV2DeleteLibraryAgentMockHandler422());
  render(<EmptyTasks agent={agent} />);

  await confirmDelete();

  await waitFor(() =>
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Failed to delete workflow",
        variant: "destructive",
      }),
    ),
  );
  expect(push).not.toHaveBeenCalled();
});

test("exporting the workflow saves its file and says so", async () => {
  server.use(getGetV1GetGraphVersionMockHandler());
  render(<EmptyTasks agent={agent} />);

  await userEvent.click(
    await screen.findByRole("button", { name: "Export workflow to file" }),
  );

  await waitFor(() =>
    expect(toast).toHaveBeenCalledWith({ title: "Workflow exported" }),
  );
  expect(exportAsJSONFile).toHaveBeenCalledWith(
    expect.anything(),
    "Morning digest_v2.json",
  );
});

test("a failed export says the workflow could not be exported", async () => {
  server.use(
    getGetV1GetGraphVersionMockHandler(() => {
      throw new Error("network down");
    }),
  );
  render(<EmptyTasks agent={agent} />);

  await userEvent.click(
    await screen.findByRole("button", { name: "Export workflow to file" }),
  );

  await waitFor(() =>
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Failed to export workflow",
        variant: "destructive",
      }),
    ),
  );
  expect(exportAsJSONFile).not.toHaveBeenCalled();
});
