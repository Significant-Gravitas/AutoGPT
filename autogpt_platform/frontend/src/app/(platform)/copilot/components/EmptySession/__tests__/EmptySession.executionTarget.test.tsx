import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../EmptySession";
import { useCopilotUIStore } from "../../../store";

const flags = vi.hoisted(() => ({ localPC: false }));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    AGENT_BRIEFING: "AGENT_BRIEFING",
    LOCAL_PC_EXECUTOR: "LOCAL_PC_EXECUTOR",
  },
  useGetFlag: (flag: string) =>
    flag === "LOCAL_PC_EXECUTOR" ? flags.localPC : false,
}));

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ user: { email: "ada@example.com" } }),
}));

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetSuggestedPrompts: () => ({ data: undefined, isLoading: false }),
}));

vi.mock("@/app/(platform)/copilot/components/ChatInput/ChatInput", () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}));

vi.mock("../components/ExecutionTargetPicker/ExecutionTargetPicker", () => ({
  ExecutionTargetPicker: () => <div data-testid="execution-target-picker" />,
}));

vi.mock("../components/EditNameDialog/EditNameDialog", () => ({
  EditNameDialog: () => null,
}));

vi.mock("../components/SuggestionThemes/SuggestionThemes", () => ({
  SuggestionThemes: () => null,
}));

vi.mock("../../PulseChips/PulseChips", () => ({ PulseChips: () => null }));
vi.mock("../../PulseChips/usePulseChips", () => ({
  usePulseChips: () => [],
}));
vi.mock("@/components/ui/dot-distortion-shader", () => ({
  DotDistortionShader: () => null,
}));
vi.mock("@/components/ui/text-generate-effect", () => ({
  TextGenerateEffect: ({ words }: { words: string }) => <div>{words}</div>,
}));
vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  },
}));

const props = {
  inputLayoutId: "input",
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
    render(<EmptySession {...props} />);

    expect(screen.getByTestId("chat-input")).toBeDefined();
    expect(screen.queryByTestId("execution-target-picker")).toBeNull();
  });

  it("shows the execution target picker only when Local PC is enabled", () => {
    flags.localPC = true;
    render(<EmptySession {...props} />);

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

    render(<EmptySession {...props} />);

    expect(useCopilotUIStore.getState().newChatExecutionTarget).toEqual({
      kind: "cloud",
    });
  });
});
