import { describe, expect, it } from "vitest";
import { rewriteButtonVariants } from "./button-variants";

const IMPORT = `import { Button } from "@/components/atoms/Button/Button";\n`;

function rewrite(jsx: string, imports = IMPORT) {
  const { output, changes, manual } = rewriteButtonVariants(
    `${imports}const a = ${jsx};\n`,
    "file.tsx",
  );
  return {
    output: output.slice(imports.length + "const a = ".length, -2),
    changes,
    manual,
  };
}

describe("rewriteButtonVariants", () => {
  it("rewrites a quiet ghost variant to ghost with its tone", () => {
    expect(
      rewrite(`<Button variant="ghost-muted" size="small">x</Button>`).output,
    ).toBe(`<Button variant="ghost" tone="quiet" size="small">x</Button>`);
    expect(rewrite(`<Button variant={"ghost-warning"}>x</Button>`).output).toBe(
      `<Button variant="ghost" tone="quiet-warning">x</Button>`,
    );
  });

  it("adds the shape a size's radius came from", () => {
    expect(
      rewrite(`<Button variant="secondary" size="xs">x</Button>`).output,
    ).toBe(`<Button variant="secondary" size="xs" shape="sharp">x</Button>`);
    expect(rewrite(`<Button size="icon-sm" variant="icon" />`).output).toBe(
      `<Button size="icon-sm" shape="soft" variant="icon" />`,
    );
    expect(
      rewrite(`<Button variant="ghost-danger" size="icon-xs" />`).output,
    ).toBe(
      `<Button variant="ghost" tone="quiet-danger" size="icon-xs" shape="sharp" />`,
    );
  });

  it("keeps a shape the call site sets", () => {
    const result = rewrite(
      `<Button variant="ghost-subtle" size="icon-2xs" shape="soft" />`,
    );
    expect(result.output).toBe(
      `<Button variant="ghost" tone="quiet-subtle" size="icon-2xs" shape="soft" />`,
    );
    expect(result.changes).toEqual([
      {
        line: 2,
        from: `variant="ghost-subtle" size="icon-2xs" shape="soft"`,
        to: `variant="ghost" size="icon-2xs" shape="soft" tone="quiet-subtle"`,
      },
    ]);
  });

  it("leaves sizes without a radius, compact buttons and other components alone", () => {
    for (const jsx of [
      `<Button variant="secondary" size="small">x</Button>`,
      `<Button variant="compact-outline" size="compact-icon" />`,
      `<Button variant={big ? "primary" : "secondary"} size="large" />`,
      `<Text variant="ghost-muted" size="xs">x</Text>`,
    ])
      expect(rewrite(jsx).changes).toEqual([]);
    expect(
      rewrite(`<Button variant="ghost-muted">x</Button>`, "").changes,
    ).toEqual([]);
  });

  it("puts the new props before a spread that precedes their anchor", () => {
    expect(rewrite(`<Button {...rest} size="xs">x</Button>`).output).toBe(
      `<Button shape="sharp" {...rest} size="xs">x</Button>`,
    );
  });

  it("reports a style prop a later spread can replace, and leaves it as is", () => {
    for (const jsx of [
      `<Button size="xs" {...{ ...sizes, size: "large" }}>x</Button>`,
      `<Button variant="ghost-muted" {...props}>x</Button>`,
    ]) {
      const result = rewrite(jsx);
      expect(result.output).toBe(jsx);
      expect(result.manual).toEqual([
        { line: 2, reason: expect.stringMatching(/before \{\.\.\./) },
      ]);
    }
    expect(
      rewrite(`<Button variant="ghost-muted" {...props} size="small" />`)
        .manual,
    ).toEqual([
      { line: 2, reason: `can't read variant="ghost-muted" before {...props}` },
    ]);
    expect(
      rewrite(`<Button variant="secondary" size="small" {...props} />`).manual,
    ).toEqual([]);
  });

  it("reads style props from a spread written as an object literal", () => {
    expect(
      rewrite(`<Button {...{ variant: "ghost-muted", size: "xs" }} />`).output,
    ).toBe(
      `<Button {...{ variant: "ghost", size: "xs" }} shape="sharp" tone="quiet" />`,
    );
    expect(
      rewrite(`<Button {...{ size: "xs" }} variant="ghost-muted" />`).output,
    ).toBe(
      `<Button {...{ size: "xs" }} shape="sharp" variant="ghost" tone="quiet" />`,
    );
    expect(
      rewrite(`<Button {...({ size: "icon-sm" } as const)} variant="icon" />`)
        .output,
    ).toBe(
      `<Button {...({ size: "icon-sm" } as const)} shape="soft" variant="icon" />`,
    );
    const later = rewrite(
      `<Button size="xs" {...{ size: "large" }}>x</Button>`,
    );
    expect(later.changes).toEqual([]);
    expect(later.manual).toEqual([]);
    expect(
      rewrite(`<Button {...{ size: "xs", shape: "soft" }} />`).changes,
    ).toEqual([]);
  });

  it("reports a spread style prop it can't read or a later spread can replace", () => {
    expect(rewrite(`<Button {...{ variant: v }} />`).manual).toEqual([
      { line: 2, reason: "can't read variant: v" },
    ]);
    expect(rewrite(`<Button {...{ size }} />`).manual).toEqual([
      { line: 2, reason: "can't read size" },
    ]);
    const replaceable = `<Button {...{ variant: "ghost-muted" }} {...props} />`;
    expect(rewrite(replaceable).output).toBe(replaceable);
    expect(rewrite(replaceable).manual).toEqual([
      {
        line: 2,
        reason: `can't read variant: "ghost-muted" before {...props}`,
      },
    ]);
  });

  it("rewrites only the imported atom, not a local of the same name", () => {
    const shadowed = `function View({ Button }) { return <Button size="xs" />; }`;
    expect(rewrite(shadowed).output).toBe(shadowed);
    const local = `function Row() { const Button = Other; return <Button size="xs" />; }`;
    expect(rewrite(local).output).toBe(local);
    expect(
      rewrite(
        `<Btn size="xs" />`,
        `import { Button as Btn } from "@/components/atoms/Button/Button";\n`,
      ).output,
    ).toBe(`<Btn size="xs" shape="sharp" />`);
  });

  it("applies a change every dynamic value needs, and reports the rest", () => {
    const dynamic = rewrite(
      `<Button variant={on ? "ghost" : "ghost-muted"} size="small" />`,
    );
    expect(dynamic.changes).toEqual([]);
    expect(dynamic.manual).toEqual([
      { line: 2, reason: expect.stringMatching(/dynamic variant/) },
    ]);
    expect(
      rewrite(
        `<Button variant={bad ? "destructive" : "icon"} size="icon-sm" />`,
      ).output,
    ).toBe(
      `<Button variant={bad ? "destructive" : "icon"} size="icon-sm" shape="soft" />`,
    );
    expect(
      rewrite(`<Button variant="outline" size={big ? "small" : "xs"} />`)
        .manual,
    ).toEqual([{ line: 2, reason: expect.stringMatching(/dynamic size/) }]);
    const forwarded = rewrite(`<Button variant={variant} size={size} />`);
    expect(forwarded.manual).toEqual([
      { line: 2, reason: expect.stringMatching(/can't read variant/) },
    ]);
    expect(
      rewrite(`<Button variant="icon" size={size} shape="pill" />`).manual,
    ).toEqual([]);
    expect(
      rewrite(`<Button variant="ghost-muted" tone="base" />`).manual,
    ).toEqual([{ line: 2, reason: expect.stringContaining("ghost-muted") }]);
  });

  it("reads a relative import of the atom", () => {
    expect(
      rewrite(
        `<Button size="xs">x</Button>`,
        `import { Button } from "../Button/Button";\n`,
      ).output,
    ).toBe(`<Button size="xs" shape="sharp">x</Button>`);
  });
});
