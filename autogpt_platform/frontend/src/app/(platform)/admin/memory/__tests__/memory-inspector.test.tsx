import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getGetV2GetGraphMockHandler200,
  getGetV2GetGraphResponseMock200,
  getGetV2GetMemoryOverviewMockHandler200,
  getGetV2GetMemoryOverviewResponseMock200,
} from "@/app/api/__generated__/endpoints/admin/admin.msw";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import MemoryDashboardPage from "../page";

const { renderVisualizer, authorize, withRoleAccess } = vi.hoisted(() => {
  const authorize = vi.fn(async (component: unknown) => component);
  return {
    renderVisualizer: vi.fn(),
    authorize,
    withRoleAccess: vi.fn(async () => authorize),
  };
});

vi.mock("@/lib/withRoleAccess", () => ({ withRoleAccess }));
vi.mock("react-force-graph-2d", () => ({ default: () => null }));
vi.mock("../components/MemoryVisualizer", async (importActual) => {
  const actual =
    await importActual<typeof import("../components/MemoryVisualizer")>();
  return {
    MemoryVisualizer: function MemoryVisualizer() {
      renderVisualizer();
      return <actual.MemoryVisualizer />;
    },
  };
});

const graphRequest = vi.fn();
const overviewRequest = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  server.use(
    getListExpertsMockHandler200([]),
    getGetV2GetGraphMockHandler200(() => {
      graphRequest();
      return {
        ...getGetV2GetGraphResponseMock200(),
        expert_id: null,
        nodes: [],
        edges: [],
        truncated: false,
      };
    }),
    getGetV2GetMemoryOverviewMockHandler200(() => {
      overviewRequest();
      return { ...getGetV2GetMemoryOverviewResponseMock200(), expert_id: null };
    }),
  );
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe.each(["CLOUD", "LOCAL"])(
  "Memory Inspector availability in %s",
  (mode) => {
    beforeEach(() => vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", mode));

    it("does not mount or query memory data when explicitly disabled", async () => {
      vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY", "false");
      render(await MemoryDashboardPage());

      expect(
        screen.getByRole("heading", { name: "Memory Inspector" }),
      ).toBeDefined();
      expect(
        screen.getByText(
          "Long-term memory is unavailable on this installation.",
        ),
      ).toBeDefined();
      expect(renderVisualizer).not.toHaveBeenCalled();
      expect(graphRequest).not.toHaveBeenCalled();
      expect(overviewRequest).not.toHaveBeenCalled();
      expect(withRoleAccess).toHaveBeenCalledWith(["admin"]);
    });

    it.each([undefined, "true", "", "False", "0"])(
      "preserves the inspector and queries with override %s",
      async (override) => {
        vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY", override);
        render(await MemoryDashboardPage());

        expect(
          screen.getByRole("heading", { name: "Memory Inspector" }),
        ).toBeDefined();
        await waitFor(() => {
          expect(graphRequest).toHaveBeenCalled();
          expect(overviewRequest).toHaveBeenCalled();
        });
        expect(renderVisualizer).toHaveBeenCalled();
        expect(withRoleAccess).toHaveBeenCalledWith(["admin"]);
        expect(
          screen.queryByText(
            "Long-term memory is unavailable on this installation.",
          ),
        ).toBeNull();
      },
    );

    it.each(["false", undefined])(
      "honors an authorization rejection with override %s",
      async (override) => {
        vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY", override);
        authorize.mockRejectedValueOnce(new Error("Unauthorized"));

        await expect(MemoryDashboardPage()).rejects.toThrow("Unauthorized");
        expect(withRoleAccess).toHaveBeenCalledWith(["admin"]);
        expect(renderVisualizer).not.toHaveBeenCalled();
        expect(graphRequest).not.toHaveBeenCalled();
      },
    );
  },
);
