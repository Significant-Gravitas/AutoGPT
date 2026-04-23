import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";

// Mock withRoleAccess to bypass server-side auth
vi.mock("@/lib/withRoleAccess", () => ({
  withRoleAccess: () =>
    Promise.resolve((Component: React.ComponentType) =>
      Promise.resolve(Component),
    ),
}));

// Mock the generated API hooks used by DiagnosticsContent
vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  useGetV2GetExecutionDiagnostics: () => ({
    data: undefined,
    isLoading: true,
    isError: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2GetAgentDiagnostics: () => ({
    data: undefined,
    isLoading: true,
    isError: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2GetScheduleDiagnostics: () => ({
    data: undefined,
    isLoading: true,
    isError: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListRunningExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListOrphanedExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListFailedExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListLongRunningExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListStuckQueuedExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListInvalidExecutions: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  usePostV2StopSingleExecution: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2StopMultipleExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2StopAllLongRunningExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2CleanupOrphanedExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2CleanupAllOrphanedExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2CleanupAllStuckQueuedExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2RequeueStuckExecution: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2RequeueMultipleStuckExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  usePostV2RequeueAllStuckQueuedExecutions: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
  useGetV2ListAllUserSchedules: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  useGetV2ListOrphanedSchedules: () => ({
    data: undefined,
    isLoading: false,
    error: null,
    refetch: vi.fn(),
  }),
  usePostV2CleanupOrphanedSchedules: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
}));

// Import the inner component directly since the page is async/server
import { DiagnosticsContent } from "../components/DiagnosticsContent";

describe("AdminDiagnosticsPage", () => {
  it("renders DiagnosticsContent in loading state", () => {
    render(<DiagnosticsContent />);
    expect(screen.getByText("Loading diagnostics...")).toBeDefined();
  });
});
