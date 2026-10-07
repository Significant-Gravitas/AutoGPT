import { ESLint } from "eslint";
import { beforeAll, describe, expect, it } from "vitest";
import {
  ALLOWLIST,
  RESTRICTED_PALETTE,
  restrictionFor,
} from "../eslint.design-system.mjs";

// Lints fixtures against the real eslint.config.mjs, so these assertions
// cover both the rule options and how the blocks are wired together.
const eslint = new ESLint({ cwd: process.cwd() });

// Loading the config and the Tailwind context takes a few seconds and is paid
// once here, outside the per-test timeout.
beforeAll(async () => {
  await eslint.lintText('export const A = <div className="p-4" />;', {
    filePath: "src/app/warmup-fixture.tsx",
  });
}, 60_000);

const IMPORT_RULE = "@typescript-eslint/no-restricted-imports";
const UNKNOWN_CLASS_RULE = "better-tailwindcss/no-unknown-classes";
const RESTRICTED_CLASS_RULE = "better-tailwindcss/no-restricted-classes";

async function lint(source: string, filePath: string) {
  const [result] = await eslint.lintText(source, { filePath });
  return result.messages;
}

async function restrictedImports(source: string, filePath: string) {
  const messages = await lint(source, filePath);
  return messages
    .filter((m) => m.ruleId === IMPORT_RULE)
    .map((m) => restrictionFor(m.message.match(/^'([^']+)'/)![1]));
}

async function unknownClasses(source: string) {
  const messages = await lint(source, "src/app/tailwind-fixture.tsx");
  return messages
    .filter((m) => m.ruleId === UNKNOWN_CLASS_RULE)
    .map((m) => m.message.replace("Unknown class detected: ", ""));
}

async function restrictedClasses(
  classes: string,
  filePath = "src/app/palette-fixture.tsx",
) {
  const messages = await lint(
    `export const A = <div className="${classes}" />;`,
    filePath,
  );
  return messages.filter((m) => m.ruleId === RESTRICTED_CLASS_RULE);
}

describe("design-system import boundaries", () => {
  const FEATURE_FILE = "src/app/(platform)/feature/import-fixture.tsx";

  it.each([
    ["@/components/__legacy__/ui/button", "legacy"],
    ["../../components/__legacy__/ui/button", "legacy"],
    ["@/components/ui/tooltip", "ui"],
    ["lucide-react", "lucide-react"],
    ["@phosphor-icons/react", "@phosphor-icons/react"],
    ["@phosphor-icons/react/dist/ssr", "@phosphor-icons/react"],
    ["@radix-ui/react-icons", "@radix-ui/react-icons"],
    ["react-icons/fa", "react-icons"],
    ["@hugeicons/react", "@hugeicons/react"],
    ["sonner", "sonner"],
  ])("blocks %s in feature code", async (source, key) => {
    expect(
      await restrictedImports(`import { X } from "${source}";`, FEATURE_FILE),
    ).toEqual([key]);
  });

  it.each([
    'import type { IconSvgElement } from "@hugeicons/react";',
    'import { type IconSvgElement } from "@hugeicons/react";',
    'import { Delete02Icon } from "@hugeicons/core-free-icons";',
    'import { Icon } from "@/components/atoms/Icon/Icon";',
    'import { toast } from "@/components/molecules/Toast/use-toast";',
    'import * as Dialog from "@radix-ui/react-dialog";',
  ])("allows %s", async (statement) => {
    expect(await restrictedImports(statement, FEATURE_FILE)).toEqual([]);
  });

  it.each([
    ["src/components/atoms/Foo/Foo.tsx", "@/components/ui/tooltip", []],
    [
      "src/components/atoms/Foo/Foo.tsx",
      "@/components/__legacy__/ui/button",
      ["legacy"],
    ],
    [
      "src/components/__legacy__/Foo.tsx",
      "@/components/__legacy__/ui/button",
      [],
    ],
    ["src/components/__legacy__/Foo.tsx", "lucide-react", ["lucide-react"]],
    ["src/components/atoms/Icon/Icon.tsx", "@hugeicons/react", []],
    ["src/components/molecules/Toast/Foo.tsx", "sonner", []],
    ["src/components/molecules/Foo/Foo.tsx", "sonner", ["sonner"]],
  ])("in %s, importing %s reports %j", async (filePath, source, expected) => {
    expect(
      await restrictedImports(`import { X } from "${source}";`, filePath),
    ).toEqual(expected);
  });

  it("exempts an allowlisted file from its own restriction only", async () => {
    const file = ALLOWLIST.imports["lucide-react"].find(
      (path) => !ALLOWLIST.imports.legacy.includes(path),
    );
    expect(file).toBeDefined();
    const source = [
      'import { X } from "lucide-react";',
      'import { Y } from "@/components/__legacy__/ui/button";',
    ].join("\n");
    expect(await restrictedImports(source, file!)).toEqual(["legacy"]);
  });
});

describe("Tailwind class checks", () => {
  it("reports unknown classes in className, cn(), cva() and class maps", async () => {
    const source = `
      import { cva } from "class-variance-authority";
      import { cn } from "@/lib/utils";
      export const modalStyles = { content: "animate-fadein p-6" };
      const variants = cva("disabled:opacity-1 rounded-full");
      export function A() {
        return <div className={cn("flex-end", variants())}><p className="text-md" /></div>;
      }
    `;
    expect((await unknownClasses(source)).sort()).toEqual([
      "animate-fadein",
      "disabled:opacity-1",
      "flex-end",
      "text-md",
    ]);
  });

  it("accepts theme tokens, plugins and the non-Tailwind hook classes", async () => {
    const source = `
      export function A() {
        return (
          <div className="group peer rounded-large shadow-subtle smooth-shadow-ring-sm animate-fade-in scrollbar-thin">
            <span className="nodrag nowheel ph-no-capture sentry-unmask agpt-border-input group-hover:p-4" />
          </div>
        );
      }
    `;
    expect(await unknownClasses(source)).toEqual([]);
  });

  it("turns on no-conflicting-classes, which reports once on Tailwind 4", async () => {
    const config = await eslint.calculateConfigForFile(
      "src/app/tailwind-fixture.tsx",
    );
    expect(config.rules?.["better-tailwindcss/no-conflicting-classes"]).toEqual(
      [2],
    );
  });
});

describe("default-palette ban", () => {
  it.each(Object.keys(RESTRICTED_PALETTE))(
    "reports %s in colour utilities",
    async (family) => {
      const messages = await restrictedClasses(
        `bg-${family}-50 text-${family}-950 border-x-${family}-200`,
      );
      expect(messages).toHaveLength(3);
      expect(messages[0].message).toContain(`\`${family}\``);
      expect(messages[0].message).toContain("DESIGN.md");
    },
  );

  it.each([
    "hover:bg-gray-100",
    "[&_svg]:!text-neutral-500",
    "data-[state=open]:ring-stone-400/50",
    "md:group-hover:from-indigo-500",
    "divide-amber-200",
    "fill-emerald-600",
    "placeholder:text-rose-400",
    "ring-offset-violet-300",
    "shadow-lime-500/20",
    "decoration-fuchsia-700",
  ])("reports %s", async (className) => {
    expect(await restrictedClasses(className)).toHaveLength(1);
  });

  it("names the palette family to use instead", async () => {
    const [gray, amber] = await restrictedClasses("text-gray-500 bg-amber-50");
    expect(gray.message).toContain("Use `zinc`");
    expect(amber.message).toContain("Use `yellow`");
  });

  it("allows the palette, semantic classes and the undecided blues", async () => {
    expect(
      await restrictedClasses(
        "bg-zinc-100 text-purple-500 border-red-200 bg-background text-muted-foreground text-blue-500 bg-sky-50 border-teal-200 text-cyan-700 grayscale bg-gray",
      ),
    ).toEqual([]);
  });

  it("exempts the files in the no-restricted-classes allowlist", async () => {
    const [file] = ALLOWLIST.tailwind["no-restricted-classes"];
    expect(file).toBeDefined();
    expect(await restrictedClasses("bg-gray-100", file)).toEqual([]);
  });
});
