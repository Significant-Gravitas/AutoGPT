import type { Linter } from "eslint";

interface Restriction {
  paths?: { name: string; message: string; allowTypeImports?: boolean }[];
  patterns?: { regex: string; message: string }[];
}

interface BlockOptions {
  allowlist?: boolean;
}

export const IMPORT_RESTRICTIONS: Record<string, Restriction>;
export const IMPORT_SCOPES: Record<string, string[]>;
export const ALLOWLIST: {
  imports: Record<string, string[]>;
  tailwind: Record<string, string[]>;
};
export const NON_TAILWIND_CLASSES: string[];
export function restrictionFor(source: string): string | undefined;
export function importBlocks(options?: BlockOptions): Linter.Config[];
export function tailwindBlocks(options?: BlockOptions): Linter.Config[];
