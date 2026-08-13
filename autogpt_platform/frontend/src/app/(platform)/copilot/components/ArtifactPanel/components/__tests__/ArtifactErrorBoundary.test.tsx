import { describe, expect, it, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { ArtifactErrorBoundary } from "../ArtifactErrorBoundary";

// Spy instead of full mock — other code paths in the test harness reach
// for ``@sentry/nextjs.withServerActionInstrumentation`` (middleware /
// server-session code), so a blanket mock drops those exports and
// generates noisy warnings from unrelated modules.
vi.mock("@sentry/nextjs", async () => {
  const actual =
    await vi.importActual<typeof import("@sentry/nextjs")>("@sentry/nextjs");
  return { ...actual, captureException: vi.fn() };
});

function Boom({ message = "render failed" }: { message?: string }): never {
  throw new Error(message);
}

function Ok() {
  return <div>child-output</div>;
}

describe("ArtifactErrorBoundary", () => {
  const baseProps = {
    artifactID: "artifact-1",
    artifactTitle: "My Artifact",
    artifactType: "markdown",
  };

  it("renders children when no error is thrown", () => {
    render(
      <ArtifactErrorBoundary {...baseProps}>
        <Ok />
      </ArtifactErrorBoundary>,
    );
    expect(screen.getByText("child-output")).toBeDefined();
  });

  it("shows the fallback message and the child's error text when a child throws", async () => {
    // React logs thrown errors to the console during render; silence the
    // noise so it doesn't drown the test report.
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <ArtifactErrorBoundary {...baseProps}>
        <Boom message="rendering exploded" />
      </ArtifactErrorBoundary>,
    );

    expect(screen.getByText(/couldn.t be rendered/i)).toBeDefined();
    // The title of the failing artifact is surfaced so the user knows which
    // artifact to regenerate.
    expect(screen.getByText("My Artifact")).toBeDefined();
    // The raw error message from the thrown Error is shown verbatim so the
    // user can paste it into chat for the agent to act on.
    expect(screen.getByText("rendering exploded")).toBeDefined();
    // The copy button is present and clickable.
    expect(
      screen.getByRole("button", { name: /copy error details/i }),
    ).toBeDefined();

    errorSpy.mockRestore();
  });

  it("copies a structured error report when the copy button is clicked", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(global.navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    render(
      <ArtifactErrorBoundary {...baseProps}>
        <Boom message="copy me" />
      </ArtifactErrorBoundary>,
    );

    fireEvent.click(
      screen.getByRole("button", { name: /copy error details/i }),
    );

    expect(writeText).toHaveBeenCalledTimes(1);
    const payload = writeText.mock.calls[0][0] as string;
    // The serialized report must carry both the identity of the artifact
    // and the error text — that's what makes it useful for the agent.
    expect(payload).toContain("Artifact: My Artifact");
    expect(payload).toContain("ID: artifact-1");
    expect(payload).toContain("Type: markdown");
    expect(payload).toContain("Error: copy me");

    errorSpy.mockRestore();
  });
});
