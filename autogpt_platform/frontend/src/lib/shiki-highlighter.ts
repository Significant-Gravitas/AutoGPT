import {
  bundledLanguages,
  bundledLanguagesInfo,
  createHighlighter,
  type BundledLanguage,
  type BundledTheme,
  type HighlighterGeneric,
} from "shiki";

export type { BundledLanguage, BundledTheme };

const LANGUAGE_ALIASES: Record<string, string> = Object.fromEntries(
  bundledLanguagesInfo.flatMap((lang) =>
    (lang.aliases ?? []).map((alias) => [alias, lang.id]),
  ),
);

const SUPPORTED_LANGUAGES = new Set(Object.keys(bundledLanguages));

const PRELOAD_LANGUAGES: BundledLanguage[] = [
  "javascript",
  "typescript",
  "python",
  "json",
  "bash",
  "yaml",
  "markdown",
  "html",
  "css",
  "sql",
  "tsx",
  "jsx",
];

export const SHIKI_THEMES: [BundledTheme, BundledTheme] = [
  "github-light",
  "github-dark",
];

let highlighterPromise: Promise<
  HighlighterGeneric<BundledLanguage, BundledTheme>
> | null = null;

export function getShikiHighlighter(): Promise<
  HighlighterGeneric<BundledLanguage, BundledTheme>
> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: SHIKI_THEMES,
      langs: PRELOAD_LANGUAGES,
    }).catch((err) => {
      highlighterPromise = null;
      throw err;
    });
  }
  return highlighterPromise;
}

export function resolveLanguage(lang: string): string {
  const normalized = lang.trim().toLowerCase();
  return LANGUAGE_ALIASES[normalized] ?? normalized;
}

export function isLanguageSupported(lang: string): boolean {
  return SUPPORTED_LANGUAGES.has(resolveLanguage(lang));
}

export function getSupportedLanguages(): BundledLanguage[] {
  return Array.from(SUPPORTED_LANGUAGES) as BundledLanguage[];
}
