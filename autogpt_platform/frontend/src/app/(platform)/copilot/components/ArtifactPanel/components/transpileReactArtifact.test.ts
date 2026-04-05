import { describe, expect, it } from "vitest";
import { transpileReactArtifactSource } from "./transpileReactArtifact";

describe("transpileReactArtifactSource", () => {
  it("transpiles a simple TSX function component", async () => {
    const src =
      'import React from "react";\nexport default function App() { return <div>hi</div>; }';
    const out = await transpileReactArtifactSource(src, "App.tsx");
    // Classic-transform emits React.createElement calls.
    // esModuleInterop emits `react_1.default.createElement(...)` — match either form.
    expect(out).toMatch(/\.createElement\(/);
    expect(out).not.toContain("<div>");
  });

  it("still transpiles when the filename lacks an extension (ensureJsxExtension)", async () => {
    const src = "export default function A() { return <span>x</span>; }";
    // Previously: filename without .tsx caused a JSX syntax error.
    const out = await transpileReactArtifactSource(src, "A");
    // esModuleInterop emits `react_1.default.createElement(...)` — match either form.
    expect(out).toMatch(/\.createElement\(/);
  });

  it("still transpiles when the filename ends in .ts (not jsx-aware)", async () => {
    const src = "export default function A() { return <b>x</b>; }";
    const out = await transpileReactArtifactSource(src, "A.ts");
    // esModuleInterop emits `react_1.default.createElement(...)` — match either form.
    expect(out).toMatch(/\.createElement\(/);
  });

  it("keeps .tsx extension as-is", async () => {
    const src = "export default function A() { return <i>x</i>; }";
    const out = await transpileReactArtifactSource(src, "Comp.tsx");
    // esModuleInterop emits `react_1.default.createElement(...)` — match either form.
    expect(out).toMatch(/\.createElement\(/);
  });

  it("throws with a useful diagnostic on syntax errors", async () => {
    const broken = "export default function A() { return <div><b></div>; }"; // unclosed <b>
    await expect(
      transpileReactArtifactSource(broken, "broken.tsx"),
    ).rejects.toThrow();
  });

  it("transpiles TypeScript type annotations away", async () => {
    const src =
      "function greet(name: string): string { return 'hi ' + name; }\nexport default () => greet('a');";
    const out = await transpileReactArtifactSource(src, "g.tsx");
    expect(out).not.toContain(": string");
    expect(out).toContain("function greet(name)");
  });
});
