/**
 * Custom Streamdown code plugin with proper shiki singleton.
 *
 * Fixes SENTRY-1051: "@streamdown/code creates a new shiki highlighter per language,
 * causing "10 instances created" warnings and memory bloat.
 *
 * This plugin creates ONE highlighter and loads languages dynamically.
 */

import {
  createHighlighter,
  bundledLanguages,
  type Highlighter,
  type BundledLanguage,
  type BundledTheme,
} from "shiki";

// Types matching streamdown's expected interface
interface HighlightToken {
  content: string;
  color?: string;
  bgColor?: string;
  htmlStyle?: Record<string, string>;
  htmlAttrs?: Record<string, string>;
  offset?: number;
}

interface HighlightResult {
  tokens: HighlightToken[][];
  fg?: string;
  bg?: string;
}

interface HighlightOptions {
  code: string;
  language: BundledLanguage;
  themes: [string, string];
}

interface CodeHighlighterPlugin {
  name: "shiki";
  type: "code-highlighter";
  highlight: (
    options: HighlightOptions,
    callback?: (result: HighlightResult) => void
  ) => HighlightResult | null;
  supportsLanguage: (language: BundledLanguage) => boolean;
  getSupportedLanguages: () => BundledLanguage[];
  getThemes: () => [BundledTheme, BundledTheme];
}

// Singleton state
let highlighterPromise: Promise<Highlighter> | null = null;
let highlighterInstance: Highlighter | null = null;
const loadedLanguages = new Set<string>();
const pendingLanguages = new Map<string, Promise<void>>();

// Result cache (same as @streamdown/code)
const resultCache = new Map<string, HighlightResult>();
const pendingCallbacks = new Map<string, Set<(result: HighlightResult) => void>>();

// All supported languages
const supportedLanguages = new Set(Object.keys(bundledLanguages));

// Cache key for results
function getCacheKey(code: string, language: string, themes: [string, string]): string {
  const prefix = code.slice(0, 100);
  const suffix = code.length > 100 ? code.slice(-100) : "";
  return `${language}:${themes[0]}:${themes[1]}:${code.length}:${prefix}:${suffix}`;
}

// Get or create the singleton highlighter
async function getHighlighter(themes: [string, string]): Promise<Highlighter> {
  if (highlighterInstance) {
    return highlighterInstance;
  }

  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: themes as [BundledTheme, BundledTheme],
      // Start with common languages pre-loaded for faster first render
      langs: ["javascript", "typescript", "python", "json", "html", "css", "bash", "markdown"],
    }).then((h: Highlighter) => {
      highlighterInstance = h;
      ["javascript", "typescript", "python", "json", "html", "css", "bash", "markdown"].forEach(
        (l) => loadedLanguages.add(l)
      );
      return h;
    });
  }

  return highlighterPromise;
}

// Load a language dynamically
async function ensureLanguageLoaded(
  highlighter: Highlighter,
  language: string
): Promise<void> {
  if (loadedLanguages.has(language)) {
    return;
  }

  if (pendingLanguages.has(language)) {
    return pendingLanguages.get(language);
  }

  const loadPromise = highlighter
    .loadLanguage(language as BundledLanguage)
    .then(() => {
      loadedLanguages.add(language);
      pendingLanguages.delete(language);
    })
    .catch((err: Error) => {
      console.warn(`[streamdown-code-singleton] Failed to load language: ${language}`, err);
      pendingLanguages.delete(language);
    });

  pendingLanguages.set(language, loadPromise);
  return loadPromise;
}

// Shiki token types
interface ShikiToken {
  content: string;
  color?: string;
  htmlStyle?: Record<string, string>;
}

// Convert shiki tokens to streamdown format
function convertTokens(
  shikiResult: ReturnType<Highlighter["codeToTokens"]>
): HighlightResult {
  return {
    tokens: shikiResult.tokens.map((line: ShikiToken[]) =>
      line.map((token: ShikiToken) => ({
        content: token.content,
        color: token.color,
        htmlStyle: token.htmlStyle,
      }))
    ),
    fg: shikiResult.fg,
    bg: shikiResult.bg,
  };
}

export interface CodePluginOptions {
  themes?: [BundledTheme, BundledTheme];
}

export function createCodePlugin(
  options: CodePluginOptions = {}
): CodeHighlighterPlugin {
  const themes = options.themes ?? ["github-light", "github-dark"];

  return {
    name: "shiki",
    type: "code-highlighter",

    supportsLanguage(language: BundledLanguage): boolean {
      return supportedLanguages.has(language);
    },

    getSupportedLanguages(): BundledLanguage[] {
      return Array.from(supportedLanguages) as BundledLanguage[];
    },

    getThemes(): [BundledTheme, BundledTheme] {
      return themes as [BundledTheme, BundledTheme];
    },

    highlight(
      { code, language, themes: highlightThemes }: HighlightOptions,
      callback?: (result: HighlightResult) => void
    ): HighlightResult | null {
      const cacheKey = getCacheKey(code, language, highlightThemes);

      // Return cached result if available
      if (resultCache.has(cacheKey)) {
        return resultCache.get(cacheKey)!;
      }

      // Register callback for async result
      if (callback) {
        if (!pendingCallbacks.has(cacheKey)) {
          pendingCallbacks.set(cacheKey, new Set());
        }
        pendingCallbacks.get(cacheKey)!.add(callback);
      }

      // Start async highlighting
      getHighlighter(highlightThemes)
        .then(async (highlighter) => {
          // Ensure language is loaded
          const lang = supportedLanguages.has(language) ? language : "text";
          if (lang !== "text") {
            await ensureLanguageLoaded(highlighter, lang);
          }

          // Highlight the code
          const effectiveLang = highlighter.getLoadedLanguages().includes(lang)
            ? lang
            : "text";

          const shikiResult = highlighter.codeToTokens(code, {
            lang: effectiveLang,
            themes: {
              light: highlightThemes[0] as BundledTheme,
              dark: highlightThemes[1] as BundledTheme,
            },
          });

          const result = convertTokens(shikiResult);
          resultCache.set(cacheKey, result);

          // Notify all pending callbacks
          const callbacks = pendingCallbacks.get(cacheKey);
          if (callbacks) {
            for (const cb of callbacks) {
              cb(result);
            }
            pendingCallbacks.delete(cacheKey);
          }
        })
        .catch((err) => {
          console.error("[streamdown-code-singleton] Failed to highlight code:", err);
          pendingCallbacks.delete(cacheKey);
        });

      // Return null while async loading
      return null;
    },
  };
}

// Pre-configured plugin with default settings (drop-in replacement for @streamdown/code)
export const code = createCodePlugin();
