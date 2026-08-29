import type { CodeHighlighterPlugin } from "streamdown";

import {
  type BundledLanguage,
  type BundledTheme,
  getShikiHighlighter,
  getSupportedLanguages,
  isLanguageSupported,
  resolveLanguage,
  SHIKI_THEMES,
} from "./shiki-highlighter";

interface HighlightResult {
  tokens: {
    content: string;
    color?: string;
    htmlStyle?: Record<string, string>;
  }[][];
  fg?: string;
  bg?: string;
}

type HighlightCallback = (result: HighlightResult) => void;

const MAX_CACHE_SIZE = 500;
const tokenCache = new Map<string, HighlightResult>();
const pendingCallbacks = new Map<string, Set<HighlightCallback>>();
const inFlightLanguageLoads = new Map<string, Promise<void>>();

function simpleHash(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return hash.toString(36);
}

function getCacheKey(
  code: string,
  lang: string,
  themes: readonly string[],
): string {
  return `${lang}:${themes.join(",")}:${simpleHash(code)}`;
}

function evictOldestIfNeeded(): void {
  if (tokenCache.size > MAX_CACHE_SIZE) {
    const oldestKey = tokenCache.keys().next().value;
    if (oldestKey) {
      tokenCache.delete(oldestKey);
    }
  }
}

export function createSingletonCodePlugin(): CodeHighlighterPlugin {
  return {
    name: "shiki",
    type: "code-highlighter",

    supportsLanguage(lang: BundledLanguage): boolean {
      return isLanguageSupported(lang);
    },

    getSupportedLanguages(): BundledLanguage[] {
      return getSupportedLanguages();
    },

    getThemes(): [BundledTheme, BundledTheme] {
      return SHIKI_THEMES;
    },

    highlight({ code, language, themes }, callback) {
      const lang = resolveLanguage(language);
      const cacheKey = getCacheKey(code, lang, themes);

      if (tokenCache.has(cacheKey)) {
        return tokenCache.get(cacheKey)!;
      }

      if (callback) {
        if (!pendingCallbacks.has(cacheKey)) {
          pendingCallbacks.set(cacheKey, new Set());
        }
        pendingCallbacks.get(cacheKey)!.add(callback);
      }

      getShikiHighlighter()
        .then(async (highlighter) => {
          const loadedLanguages = highlighter.getLoadedLanguages();

          if (!loadedLanguages.includes(lang) && isLanguageSupported(lang)) {
            let loadPromise = inFlightLanguageLoads.get(lang);
            if (!loadPromise) {
              loadPromise = highlighter
                .loadLanguage(lang as BundledLanguage)
                .finally(() => {
                  inFlightLanguageLoads.delete(lang);
                });
              inFlightLanguageLoads.set(lang, loadPromise);
            }
            await loadPromise;
          }

          const finalLang = (
            highlighter.getLoadedLanguages().includes(lang) ? lang : "text"
          ) as BundledLanguage;

          const shikiResult = highlighter.codeToTokens(code, {
            lang: finalLang,
            themes: { light: themes[0], dark: themes[1] },
          });

          const result: HighlightResult = {
            tokens: shikiResult.tokens.map((line) =>
              line.map((token) => ({
                content: token.content,
                color: token.color,
                htmlStyle: token.htmlStyle,
              })),
            ),
            fg: shikiResult.fg,
            bg: shikiResult.bg,
          };

          evictOldestIfNeeded();
          tokenCache.set(cacheKey, result);

          const callbacks = pendingCallbacks.get(cacheKey);
          if (callbacks) {
            callbacks.forEach((cb) => {
              cb(result);
            });
            pendingCallbacks.delete(cacheKey);
          }
        })
        .catch((error) => {
          console.error("[Shiki] Failed to highlight code:", error);

          const fallback: HighlightResult = {
            tokens: code.split("\n").map((line) => [{ content: line }]),
          };

          const callbacks = pendingCallbacks.get(cacheKey);
          if (callbacks) {
            callbacks.forEach((cb) => {
              cb(fallback);
            });
            pendingCallbacks.delete(cacheKey);
          }
        });

      return null;
    },
  };
}

export const code = createSingletonCodePlugin();
