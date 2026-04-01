import { describe, expect, it } from "vitest";
import {
  buildReactArtifactSrcDoc,
  transpileReactArtifactSource,
} from "./reactArtifactPreview";

describe("reactArtifactPreview", () => {
  it("transpiles tsx source into executable javascript", async () => {
    const compiled = await transpileReactArtifactSource(
      "export default function App() { return <div>Hello</div>; }",
      "App.tsx",
    );

    expect(compiled).toContain("exports.default = App");
    expect(compiled).toContain("React.createElement");
  });

  it("builds an iframe document that mounts the compiled component", () => {
    const srcDoc = buildReactArtifactSrcDoc(
      "exports.default = function App() {}",
      "Widget.tsx",
      "<style>.demo { color: red; }</style>",
    );

    expect(srcDoc).toContain("react@18.3.1/umd/react.production.min.js");
    expect(srcDoc).toContain(
      "react-dom@18.3.1/umd/react-dom.production.min.js",
    );
    expect(srcDoc).toContain('integrity="sha384-');
    expect(srcDoc).toContain("Content-Security-Policy");
    expect(srcDoc).toContain(".demo { color: red; }");
    expect(srcDoc).toContain("Unsupported import in artifact preview");
    expect(srcDoc).toContain(
      "No renderable component found. Export a default component, export App, or export a named component.",
    );
    expect(srcDoc).toContain("wrapWithProviders");
    expect(srcDoc).toContain("PreviewErrorBoundary");
    expect(srcDoc).toContain("Widget.tsx");
  });
});
