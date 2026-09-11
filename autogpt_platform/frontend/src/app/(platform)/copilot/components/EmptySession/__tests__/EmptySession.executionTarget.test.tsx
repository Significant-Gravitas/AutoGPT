import { render, screen } from "@/tests/integrations/test-utils";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../EmptySession";
import { useCopilotUIStore } from "../../../store";

const flags = vi.hoisted(() => ({ localPC: false }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.LOCAL_PC_EXECUTOR ? flags.localPC : false,
    useFlagStatus: () => ({ enabled: false, ready: true }),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { email: "ada@example.com" },
    isUserLoading: false,
    isLoggedIn: true,
  }),
}));

vi.mock(
  "@/app/api/__generated__/endpoints/chat/chat",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("@/app/api/__generated__/endpoints/chat/chat")
      >();
    return {
      ...actual,
      useGetV2GetSuggestedPrompts: () => ({
        data: undefined,
        isLoading: false,
      }),
    };
  },
);

vi.mock("@/app/(platform)/copilot/components/ChatInput/ChatInput", () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}));

vi.mock("../components/ExecutionTargetPicker/ExecutionTargetPicker", () => ({
  ExecutionTargetPicker: () => <div data-testid="execution-target-picker" />,
}));

vi.mock("../components/SuggestionThemes/SuggestionThemes", () => ({
  SuggestionThemes: () => null,
}));

vi.mock("@/components/ui/dot-distortion-shader", () => ({
  DotDistortionShader: () => null,
}));
vi.mock("@/components/ui/text-generate-effect", () => ({
  TextGenerateEffect: ({ words }: { words: string }) => <div>{words}</div>,
}));

const props = {
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onSend: vi.fn(),
};

afterEach(() => {
  flags.localPC = false;
  vi.clearAllMocks();
});

describe("EmptySession execution target flag", () => {
  it("leaves the existing Cloud-only UI unchanged when the flag is off", () => {
    render(
      <NuqsTestingAdapter>
        <EmptySession {...props} />
      </NuqsTestingAdapter>,
    );

    expect(screen.getByTestId("chat-input")).toBeDefined();
    expect(screen.queryByTestId("execution-target-picker")).toBeNull();
  });

  it("shows the execution target picker only when Local PC is enabled", () => {
    flags.localPC = true;
    render(
      <NuqsTestingAdapter>
        <EmptySession {...props} />
      </NuqsTestingAdapter>,
    );

    expect(screen.getByTestId("execution-target-picker")).toBeDefined();
  });

  it("starts a newly mounted chat in Cloud", () => {
    useCopilotUIStore.getState().setNewChatExecutionTarget({
      kind: "local",
      machineID: "machine-1",
      machineLabel: "Workstation",
      connectionID: "connection-1",
      browseID: "browse-1",
      directoryRef: "directory-1",
      displayPath: "C:\\Projects",
    });

    render(
      <NuqsTestingAdapter>
        <EmptySession {...props} />
      </NuqsTestingAdapter>,
    );

    expect(useCopilotUIStore.getState().newChatExecutionTarget).toEqual({
      kind: "cloud",
    });
  });
});
