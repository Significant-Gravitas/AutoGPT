import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ArtifactReactPreview } from "./ArtifactReactPreview";
import {
  buildReactArtifactSrcDoc,
  collectPreviewStyles,
  transpileReactArtifactSource,
} from "./reactArtifactPreview";

vi.mock("./reactArtifactPreview", () => ({
  buildReactArtifactSrcDoc: vi.fn(),
  collectPreviewStyles: vi.fn(),
  transpileReactArtifactSource: vi.fn(),
}));

describe("ArtifactReactPreview", () => {
  beforeEach(() => {
    vi.mocked(collectPreviewStyles).mockReturnValue("<style>preview</style>");
    vi.mocked(buildReactArtifactSrcDoc).mockReturnValue("<html>preview</html>");
  });

  it("renders an iframe preview after transpilation succeeds", async () => {
    vi.mocked(transpileReactArtifactSource).mockResolvedValue(
      "module.exports.default = function Artifact() { return null; };",
    );

    const { container } = render(
      <ArtifactReactPreview
        source="export default function Artifact() { return null; }"
        title="Artifact.tsx"
      />,
    );

    await waitFor(() => {
      expect(buildReactArtifactSrcDoc).toHaveBeenCalledWith(
        "module.exports.default = function Artifact() { return null; };",
        "Artifact.tsx",
        "<style>preview</style>",
      );
    });

    const iframe = container.querySelector("iframe");
    expect(iframe).toBeTruthy();
    expect(iframe?.getAttribute("sandbox")).toBe("allow-scripts");
    expect(iframe?.getAttribute("title")).toBe("Artifact.tsx preview");
    expect(iframe?.getAttribute("srcdoc")).toBe("<html>preview</html>");
  });

  it("shows a readable error when transpilation fails", async () => {
    vi.mocked(transpileReactArtifactSource).mockRejectedValue(
      new Error("Transpile exploded"),
    );

    render(
      <ArtifactReactPreview
        source="export default function Artifact() {"
        title="Broken.tsx"
      />,
    );

    await waitFor(() => {
      expect(screen.getByText("Failed to render React preview")).toBeTruthy();
    });

    expect(screen.getByText("Transpile exploded")).toBeTruthy();
  });
});
