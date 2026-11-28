import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { MessagePartRenderer } from "../MessagePartRenderer";

type Part = UIMessage<unknown, UIDataTypes, UITools>["parts"][number];

vi.mock("../ReasoningCollapse", () => ({
  ReasoningCollapse: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="reasoning-collapse">{children}</div>
  ),
}));

// The real MessageResponse renders markdown via Streamdown, passing the custom
// `components.img` override (WorkspaceMediaImage) down. We exercise that override
// directly so its video/image/empty-src branches are covered.
type ImgComponent = (
  props: React.JSX.IntrinsicElements["img"],
) => React.ReactNode;
vi.mock("@/components/organisms/Message/Message", () => ({
  MessageResponse: ({
    children,
    components,
  }: {
    children: React.ReactNode;
    components?: { img?: ImgComponent };
  }) => {
    const Img = components?.img;
    return (
      <div data-testid="message-response">
        {children}
        {Img ? (
          <div data-testid="custom-img-outputs">
            {Img({ src: "/v.mp4", alt: "video:Clip" })}
            {Img({ src: "/p.png", alt: "Pic" })}
            {Img({ src: "", alt: "missing" })}
          </div>
        ) : null}
      </div>
    );
  },
}));

vi.mock("@/components/molecules/ErrorCard/ErrorCard", () => ({
  ErrorCard: () => null,
}));

vi.mock("../StoppedTaskCard", () => ({
  StoppedTaskCard: () => <div data-testid="stopped-task-card" />,
}));

const extractWorkspaceArtifactsSpy = vi.hoisted(() => vi.fn());

vi.mock("../../helpers", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../helpers")>();
  extractWorkspaceArtifactsSpy.mockImplementation(
    actual.extractWorkspaceArtifacts,
  );
  return { ...actual, extractWorkspaceArtifacts: extractWorkspaceArtifactsSpy };
});

vi.mock("../../../../components/ArtifactCard/ArtifactCard", () => ({
  ArtifactCard: () => <div data-testid="artifact-card" />,
}));
describe("MessagePartRenderer reasoning branch", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders a ReasoningCollapse wrapping a <pre> with the reasoning text", () => {
    const part = {
      type: "reasoning",
      text: "step-by-step plan",
      state: "done",
    } as unknown as Part;

    render(<MessagePartRenderer part={part} messageID="m1" partIndex={0} />);

    const collapse = screen.getByTestId("reasoning-collapse");
    expect(collapse).toBeDefined();
    const pre = collapse.querySelector("pre");
    expect(pre).not.toBeNull();
    expect(pre?.textContent).toBe("step-by-step plan");
  });

  it("returns null when the reasoning text is whitespace-only", () => {
    const part = {
      type: "reasoning",
      text: "   \n  ",
      state: "done",
    } as unknown as Part;

    const { container } = render(
      <MessagePartRenderer part={part} messageID="m1" partIndex={0} />,
    );
    expect(container.firstChild).toBeNull();
    expect(screen.queryByTestId("reasoning-collapse")).toBeNull();
  });

  it("returns null when the reasoning part has no text key", () => {
    const part = { type: "reasoning", state: "done" } as unknown as Part;

    const { container } = render(
      <MessagePartRenderer part={part} messageID="m1" partIndex={0} />,
    );
    expect(container.firstChild).toBeNull();
    expect(screen.queryByTestId("reasoning-collapse")).toBeNull();
  });

  it("returns null when the reasoning part's text is not a string", () => {
    const part = {
      type: "reasoning",
      text: 42,
      state: "done",
    } as unknown as Part;

    const { container } = render(
      <MessagePartRenderer part={part} messageID="m1" partIndex={0} />,
    );
    expect(container.firstChild).toBeNull();
    expect(screen.queryByTestId("reasoning-collapse")).toBeNull();
  });

  it("renders reasoning content when it contains non-whitespace surrounded by whitespace", () => {
    const part = {
      type: "reasoning",
      text: "  reasoning-with-pad  ",
      state: "done",
    } as unknown as Part;

    render(<MessagePartRenderer part={part} messageID="m2" partIndex={3} />);

    const pre = screen.getByTestId("reasoning-collapse").querySelector("pre");
    expect(pre?.textContent).toBe("  reasoning-with-pad  ");
  });
});

describe("MessagePartRenderer text branch", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders plain text through the MessageResponse renderer", () => {
    const part = { type: "text", text: "hello world" } as unknown as Part;
    render(<MessagePartRenderer part={part} messageID="m1" partIndex={0} />);
    expect(screen.getByTestId("message-response").textContent).toContain(
      "hello world",
    );
  });

  it("renders a <video> for video: alt, an <img> for image alt, and nothing for empty src", () => {
    const part = { type: "text", text: "media" } as unknown as Part;
    const { container } = render(
      <MessagePartRenderer part={part} messageID="m1" partIndex={0} />,
    );
    const outputs = screen.getByTestId("custom-img-outputs");
    // video: alt -> <video><source/></video>
    expect(outputs.querySelector("video source")?.getAttribute("src")).toBe(
      "/v.mp4",
    );
    // normal alt -> <img>
    const img = outputs.querySelector("img");
    expect(img?.getAttribute("src")).toBe("/p.png");
    expect(img?.getAttribute("alt")).toBe("Pic");
    // empty src -> render nothing extra (only the one video + one img above)
    expect(container.querySelectorAll("img")).toHaveLength(1);
  });

  it("renders inline ArtifactCards without a feature flag", () => {
    const fileId = "550e8400-e29b-41d4-a716-446655440000";
    const part = {
      type: "text",
      text: `Here is [report](workspace://${fileId}).`,
    } as unknown as Part;
    render(<MessagePartRenderer part={part} messageID="m1" partIndex={0} />);
    expect(screen.getByTestId("artifact-card")).toBeDefined();
  });

  it("does not re-extract artifacts when the part re-renders with the same text", () => {
    const fileId = "550e8400-e29b-41d4-a716-446655440000";
    const text = `Here is [report](workspace://${fileId}).`;
    extractWorkspaceArtifactsSpy.mockClear();
    const { rerender } = render(
      <MessagePartRenderer
        part={{ type: "text", text } as unknown as Part}
        messageID="m1"
        partIndex={0}
      />,
    );
    rerender(
      <MessagePartRenderer
        part={{ type: "text", text } as unknown as Part}
        messageID="m1"
        partIndex={0}
      />,
    );
    expect(extractWorkspaceArtifactsSpy).toHaveBeenCalledTimes(1);

    rerender(
      <MessagePartRenderer
        part={{ type: "text", text: `${text} More.` } as unknown as Part}
        messageID="m1"
        partIndex={0}
      />,
    );
    expect(extractWorkspaceArtifactsSpy).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId("artifact-card")).toBeDefined();
  });

  it("renders the StoppedTaskCard for a cancellation marker", () => {
    const part = {
      type: "text",
      text: "[__COPILOT_ERROR_f7a1__] Operation cancelled",
    } as unknown as Part;
    render(<MessagePartRenderer part={part} messageID="m1" partIndex={0} />);
    expect(screen.getByTestId("stopped-task-card")).toBeDefined();
  });

  it("renders a system-marker note", () => {
    const part = {
      type: "text",
      text: "[__COPILOT_SYSTEM_e3b0__] Reconnected to the session",
    } as unknown as Part;
    render(<MessagePartRenderer part={part} messageID="m1" partIndex={0} />);
    expect(screen.getByText("Reconnected to the session")).toBeDefined();
  });
});

describe("MessagePartRenderer other parts", () => {
  afterEach(() => {
    cleanup();
  });

  it("returns null for a non-tool, non-text/reasoning part type", () => {
    const part = { type: "step-start" } as unknown as Part;
    const { container } = render(
      <MessagePartRenderer part={part} messageID="m1" partIndex={0} />,
    );
    expect(container.firstChild).toBeNull();
  });
});
