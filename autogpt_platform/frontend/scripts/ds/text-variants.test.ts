import { describe, expect, it } from "vitest";
import { rewriteTextVariants } from "./text-variants";

const IMPORT = `import { Text } from "@/components/atoms/Text/Text";\n`;

function rewrite(jsx: string, imports = IMPORT) {
  const { output, changes, manual } = rewriteTextVariants(
    `${imports}const a = ${jsx};\n`,
    "file.tsx",
  );
  return {
    output: output.slice(imports.length + "const a = ".length, -2),
    changes,
    manual,
  };
}

describe("rewriteTextVariants", () => {
  it("rewrites an old variant to its role and props", () => {
    expect(rewrite(`<Text variant="small-medium-tight">x</Text>`).output).toBe(
      `<Text variant="small" weight="medium" leading="4">x</Text>`,
    );
  });

  it("writes a boolean prop as shorthand and keeps the old element", () => {
    expect(rewrite(`<Text variant="small-mono-tight">x</Text>`).output).toBe(
      `<Text variant="small" as="code" mono leading="4">x</Text>`,
    );
  });

  it("lets a prop already at the call site win over the old variant's", () => {
    expect(
      rewrite(`<Text variant="small-medium" weight="semibold">x</Text>`).output,
    ).toBe(`<Text variant="small" weight="semibold">x</Text>`);
    expect(
      rewrite(`<Text as="span" variant="small-mono-tight">x</Text>`).output,
    ).toBe(`<Text as="span" variant="small" mono leading="4">x</Text>`);
  });

  it("leaves a role, and other components, alone", () => {
    const role = rewrite(`<Text variant="small" weight="medium">x</Text>`);
    expect(role.changes).toEqual([]);
    expect(rewrite(`<Button variant="body-medium">x</Button>`).changes).toEqual(
      [],
    );
    expect(rewrite(`<Text variant="body-medium">x</Text>`, "").changes).toEqual(
      [],
    );
  });

  it("reads a quoted expression and rewrites size", () => {
    expect(rewrite(`<Text variant={"body-medium"}>x</Text>`).output).toBe(
      `<Text variant="body" weight="medium">x</Text>`,
    );
    expect(
      rewrite(`<Text variant="h4" size="body-medium">x</Text>`).output,
    ).toBe(`<Text variant="h4" size="body" weight="medium">x</Text>`);
    expect(
      rewrite(`<Text variant="small-mono-tight" size="h3">x</Text>`).output,
    ).toBe(`<Text variant="small" size="h3" as="code">x</Text>`);
  });

  it("puts the props before a spread that precedes the variant", () => {
    expect(
      rewrite(`<Text {...rest} variant="body-medium">x</Text>`).output,
    ).toBe(`<Text weight="medium" {...rest} variant="body">x</Text>`);
  });

  it("reports an old variant or size a later spread can replace", () => {
    for (const jsx of [
      `<Text variant="body-medium" {...{ ...base, variant: "small" }}>x</Text>`,
      `<Text variant="h4" size="body-medium" {...props}>x</Text>`,
    ]) {
      const result = rewrite(jsx);
      expect(result.output).toBe(jsx);
      expect(result.manual).toEqual([
        { line: 2, reason: expect.stringMatching(/before \{\.\.\./) },
      ]);
    }
    expect(rewrite(`<Text variant="small" {...props}>x</Text>`).manual).toEqual(
      [],
    );
  });

  it("reads a spread written as an object literal like the attributes", () => {
    expect(
      rewrite(`<Text {...{ variant: "small-medium-tight" }}>x</Text>`).output,
    ).toBe(
      `<Text {...{ variant: "small" }} weight="medium" leading="4">x</Text>`,
    );
    expect(
      rewrite(`<Text {...{ size: "body-medium" }} variant="h4">x</Text>`)
        .output,
    ).toBe(`<Text {...{ size: "body" }} weight="medium" variant="h4">x</Text>`);
    expect(
      rewrite(
        `<Text {...{ weight: "semibold" }} variant="small-medium">x</Text>`,
      ).output,
    ).toBe(`<Text {...{ weight: "semibold" }} variant="small">x</Text>`);
    const replaced = rewrite(
      `<Text variant="body-medium" {...{ variant: "small" }}>x</Text>`,
    );
    expect(replaced.changes).toEqual([]);
    expect(replaced.manual).toEqual([]);
  });

  it("reports a variant in a spread a later spread can replace, or one it can't read", () => {
    expect(
      rewrite(`<Text {...{ variant: "body-medium" }} {...props}>x</Text>`)
        .manual,
    ).toEqual([
      {
        line: 2,
        reason: `can't read variant: "body-medium" before {...props}`,
      },
    ]);
    expect(rewrite(`<Text {...{ variant: v }}>x</Text>`).manual).toEqual([
      { line: 2, reason: "dynamic variant: variant: v" },
    ]);
    expect(
      rewrite(`<Text {...{ weight: w }} variant="small-medium">x</Text>`)
        .manual,
    ).toHaveLength(1);
  });

  it("rewrites only the imported atom, not a local of the same name", () => {
    const shadowed = `function View({ Text }) { return <Text variant="body-medium">x</Text>; }`;
    expect(rewrite(shadowed).output).toBe(shadowed);
    expect(
      rewrite(
        `<Copy variant="body-medium">x</Copy>`,
        `import { Text as Copy } from "@/components/atoms/Text/Text";\n`,
      ).output,
    ).toBe(`<Copy variant="body" weight="medium">x</Copy>`);
  });

  it("reports dynamic variants and dynamic props it would override", () => {
    const dynamic = rewrite(
      `<Text variant={big ? "body-medium" : "small"}>x</Text>`,
    );
    expect(dynamic.changes).toEqual([]);
    expect(dynamic.manual).toEqual([
      expect.objectContaining({
        line: 2,
        reason: expect.stringMatching(/variant/),
      }),
    ]);
    const prop = rewrite(`<Text variant="small-medium" weight={w}>x</Text>`);
    expect(prop.changes).toEqual([]);
    expect(prop.manual).toHaveLength(1);
  });

  it("reads a relative import of the atom", () => {
    expect(
      rewrite(
        `<Text variant="small-medium">x</Text>`,
        `import { Text } from "../Text/Text";\n`,
      ).output,
    ).toBe(`<Text variant="small" weight="medium">x</Text>`);
  });

  it("never adds an element to OverflowText, which renders a span", () => {
    expect(
      rewrite(
        `<OverflowText variant="small-mono-tight" value="x" />`,
        `import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";\n`,
      ).output,
    ).toBe(`<OverflowText variant="small" mono leading="4" value="x" />`);
  });
});
