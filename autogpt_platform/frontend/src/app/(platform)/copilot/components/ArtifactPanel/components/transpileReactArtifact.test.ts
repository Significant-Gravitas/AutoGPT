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

  it("transpiles a concrete props-based artifact with previewProps", async () => {
    const src = `
      import React, { FC, useState } from "react";

      interface ArtifactFile {
        id: string;
        name: string;
        mimeType: string;
        url: string;
        sizeBytes: number;
      }

      interface Props {
        files: ArtifactFile[];
        onSelect: (file: ArtifactFile) => void;
      }

      export const previewProps: Props = {
        files: [
          {
            id: "1",
            name: "report.png",
            mimeType: "image/png",
            url: "/report.png",
            sizeBytes: 2048,
          },
        ],
        onSelect: () => {},
      };

      const ArtifactList: FC<Props> = ({ files, onSelect }) => {
        const [selected, setSelected] = useState<string | null>(null);

        const handleClick = (file: ArtifactFile) => {
          setSelected(file.id);
          onSelect(file);
        };

        return (
          <ul>
            {files.map((file) => (
              <li key={file.id} onClick={() => handleClick(file)}>
                <span>{selected === file.id ? "selected" : file.name}</span>
              </li>
            ))}
          </ul>
        );
      };

      export default ArtifactList;
    `;

    const out = await transpileReactArtifactSource(src, "ArtifactList.tsx");

    expect(out).toContain("exports.previewProps");
    expect(out).toContain("exports.default = ArtifactList");
    expect(out).toContain("useState");
    expect(out).not.toContain("interface Props");
    expect(out).not.toContain("interface ArtifactFile");
  });

  it("transpiles a named export artifact without a default export", async () => {
    const src = `
      export function ResultsGrid() {
        return (
          <section>
            <h1>Results</h1>
            <p>Named export preview</p>
          </section>
        );
      }
    `;

    const out = await transpileReactArtifactSource(src, "ResultsGrid.tsx");

    expect(out).toContain("exports.ResultsGrid = ResultsGrid");
    expect(out).toMatch(/\.createElement\(/);
    expect(out).not.toContain("<section>");
  });

  it("transpiles a provider-wrapped artifact with separate provider and component exports", async () => {
    const src = `
      import React from "react";

      export function DemoProvider({ children }: { children: React.ReactNode }) {
        return <div data-theme="demo">{children}</div>;
      }

      export function DashboardCard() {
        return <main>Provider-backed preview</main>;
      }
    `;

    const out = await transpileReactArtifactSource(src, "DashboardCard.tsx");

    expect(out).toContain("exports.DemoProvider = DemoProvider");
    expect(out).toContain("exports.DashboardCard = DashboardCard");
    expect(out).not.toContain("React.ReactNode");
  });
});
