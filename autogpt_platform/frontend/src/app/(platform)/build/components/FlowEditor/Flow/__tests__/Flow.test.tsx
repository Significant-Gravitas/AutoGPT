import { render, screen } from "@testing-library/react";
import { ReactNode } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";

let mockIsReadOnly = false;
let mockHasWebhookNodes = false;

const mockUseFlow = vi.fn(() => ({
  onDragOver: vi.fn(),
  onDrop: vi.fn(),
  isFlowContentLoading: false,
  isInitialLoadComplete: true,
  isLocked: mockIsReadOnly,
  setIsLocked: vi.fn(),
  isReadOnly: mockIsReadOnly,
}));

vi.mock("../useFlow", () => ({ useFlow: () => mockUseFlow() }));
vi.mock("../useFlowRealtime", () => ({ useFlowRealtime: vi.fn() }));
vi.mock("../useCopyPaste", () => ({ useCopyPaste: vi.fn() }));
vi.mock("../../edges/useCustomEdge", () => ({
  useCustomEdge: () => ({
    edges: [],
    onConnect: vi.fn(),
    onEdgesChange: vi.fn(),
  }),
}));
vi.mock("../../nodes/CustomNode/CustomNode", () => ({
  CustomNode: () => null,
}));
vi.mock("../../edges/CustomEdge", () => ({ default: () => null }));
vi.mock("../helpers/resolve-collision", () => ({
  resolveCollisions: (nodes: unknown) => nodes,
}));

vi.mock("@xyflow/react", () => ({
  ReactFlow: ({ children }: { children: ReactNode }) => (
    <div data-testid="react-flow">{children}</div>
  ),
  Background: () => <div data-testid="background" />,
  useReactFlow: () => ({ zoomIn: vi.fn(), zoomOut: vi.fn(), fitView: vi.fn() }),
}));

vi.mock("zustand/react/shallow", () => ({
  useShallow: (fn: unknown) => fn,
}));

const nodeState = {
  nodes: [],
  setNodes: vi.fn(),
  onNodesChange: vi.fn(),
  hasWebhookNodes: () => mockHasWebhookNodes,
};
const useNodeStoreMock = (selector: (state: typeof nodeState) => unknown) =>
  selector(nodeState);
useNodeStoreMock.getState = () => nodeState;
vi.mock("../../../../stores/nodeStore", () => ({
  useNodeStore: (selector: (state: typeof nodeState) => unknown) =>
    useNodeStoreMock(selector),
}));

const graphState = { isGraphRunning: false };
vi.mock("../../../../stores/graphStore", () => ({
  useGraphStore: (selector: (state: typeof graphState) => unknown) =>
    selector(graphState),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { BUILDER_CHAT_PANEL: "builder-chat-panel" },
  useGetFlag: () => false,
}));

vi.mock("@/app/api/__generated__/endpoints/graphs/graphs", () => ({
  useGetV1GetSpecificGraph: () => ({ data: { id: "graph-1" } }),
}));

vi.mock("@/app/api/helpers", () => ({
  okData: (res: { data: unknown }) => res?.data,
}));

vi.mock("nuqs", () => ({
  parseAsString: {},
  parseAsInteger: {},
  useQueryStates: () => [{ flowID: "graph-1", flowExecutionID: null }, vi.fn()],
}));

vi.mock("../../../NewControlPanel/NewControlPanel", () => ({
  default: ({ isReadOnly }: { isReadOnly?: boolean }) => (
    <div data-testid="control-panel" data-readonly={String(!!isReadOnly)} />
  ),
}));
vi.mock("../../../ReadOnlyBanner/ReadOnlyBanner", () => ({
  ReadOnlyBanner: () => <div data-testid="read-only-banner" />,
}));
vi.mock("../../../BuilderActions/BuilderActions", () => ({
  BuilderActions: () => <div data-testid="builder-actions" />,
}));
vi.mock("../components/CustomControl", () => ({
  CustomControls: ({ isReadOnly }: { isReadOnly?: boolean }) => (
    <div data-testid="custom-controls" data-readonly={String(!!isReadOnly)} />
  ),
}));
vi.mock("../components/GraphLoadingBox", () => ({
  GraphLoadingBox: () => <div data-testid="graph-loading-box" />,
}));
vi.mock("../components/RunningBackground", () => ({
  RunningBackground: () => <div data-testid="running-background" />,
}));
vi.mock("../components/TriggerAgentBanner", () => ({
  TriggerAgentBanner: () => <div data-testid="trigger-agent-banner" />,
}));
vi.mock("../../../FloatingSafeModeToogle", () => ({
  FloatingSafeModeToggle: () => <div data-testid="safe-mode-toggle" />,
}));
vi.mock("../../../DraftRecoveryDialog/DraftRecoveryPopup", () => ({
  DraftRecoveryPopup: () => <div data-testid="draft-recovery" />,
}));
vi.mock("../../../BuilderChatPanel/BuilderChatPanel", () => ({
  BuilderChatPanel: () => <div data-testid="builder-chat" />,
}));
vi.mock("@/components/molecules/ErrorBoundary/ErrorBoundary", () => ({
  ErrorBoundary: ({ children }: { children: ReactNode }) => <>{children}</>,
}));
vi.mock(
  "@/components/organisms/FloatingReviewsPanel/FloatingReviewsPanel",
  () => ({
    FloatingReviewsPanel: () => <div data-testid="reviews-panel" />,
  }),
);

import { Flow } from "../Flow";

describe("Flow read-only gating", () => {
  beforeEach(() => {
    mockIsReadOnly = false;
    mockHasWebhookNodes = false;
  });

  test("owned graph: shows builder actions, hides read-only banner, controls editable", () => {
    render(<Flow />);

    expect(screen.queryByTestId("builder-actions")).not.toBeNull();
    expect(screen.queryByTestId("read-only-banner")).toBeNull();
    expect(screen.queryByTestId("safe-mode-toggle")).not.toBeNull();
    expect(
      screen.getByTestId("control-panel").getAttribute("data-readonly"),
    ).toBe("false");
    expect(
      screen.getByTestId("custom-controls").getAttribute("data-readonly"),
    ).toBe("false");
  });

  test("non-owned graph: shows read-only banner, hides action bar, gates controls", () => {
    mockIsReadOnly = true;
    render(<Flow />);

    expect(screen.queryByTestId("read-only-banner")).not.toBeNull();
    expect(screen.queryByTestId("builder-actions")).toBeNull();
    expect(screen.queryByTestId("trigger-agent-banner")).toBeNull();
    expect(screen.queryByTestId("safe-mode-toggle")).toBeNull();
    expect(
      screen.getByTestId("control-panel").getAttribute("data-readonly"),
    ).toBe("true");
    expect(
      screen.getByTestId("custom-controls").getAttribute("data-readonly"),
    ).toBe("true");
  });

  test("non-owned trigger graph still hides the trigger banner", () => {
    mockIsReadOnly = true;
    mockHasWebhookNodes = true;
    render(<Flow />);

    expect(screen.queryByTestId("read-only-banner")).not.toBeNull();
    expect(screen.queryByTestId("trigger-agent-banner")).toBeNull();
  });
});
