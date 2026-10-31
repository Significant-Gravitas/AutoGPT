import type { GrammarState } from "shiki";
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
type Highlighter = Awaited<ReturnType<typeof getShikiHighlighter>>;
type HighlightedLine = HighlightResult["tokens"][number];

/**
 * Where tokenizing stopped for a block that may still be growing: the tokens
 * of every line before its last one, and the grammar state after them. A
 * longer version of the same block only tokenizes what comes after `text`.
 * `lastCode` is the latest version highlighted from here.
 */
interface StreamCheckpoint {
  lang: string;
  themesKey: string;
  text: string;
  lines: HighlightedLine[];
  grammarState: GrammarState | undefined;
  lastCacheKey: string;
  lastCode: string;
}

const MAX_CACHE_SIZE = 500;
const MAX_PARTIAL_CACHE_SIZE = 50;
const MAX_CHECKPOINTS = 20;
const tokenCache = new Map<string, HighlightResult>();
// Versions of a block that a longer version has replaced. Kept apart so a
// streaming block cannot push other blocks' tokens out of `tokenCache`.
const partialCache = new Map<string, HighlightResult>();
const checkpoints: StreamCheckpoint[] = [];
const pendingCallbacks = new Map<string, Set<HighlightCallback>>();
const inFlightLanguageLoads = new Map<string, Promise<void>>();

function getCacheKey(
  code: string,
  lang: string,
  themes: readonly string[],
): string {
  return `${lang}:${themes.join(",")}:${code}`;
}

function setBounded(
  cache: Map<string, HighlightResult>,
  key: string,
  value: HighlightResult,
  maxSize: number,
): void {
  cache.set(key, value);
  if (cache.size > maxSize) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey !== undefined) {
      cache.delete(oldestKey);
    }
  }
}

function getCachedResult(cacheKey: string): HighlightResult | undefined {
  return tokenCache.get(cacheKey) ?? partialCache.get(cacheKey);
}

function demoteToPartial(cacheKey: string): void {
  const result = tokenCache.get(cacheKey);
  if (!result) return;
  tokenCache.delete(cacheKey);
  setBounded(partialCache, cacheKey, result, MAX_PARTIAL_CACHE_SIZE);
}

function findCheckpoint(
  code: string,
  lang: string,
  themesKey: string,
): StreamCheckpoint | undefined {
  let best: StreamCheckpoint | undefined;
  for (const checkpoint of checkpoints) {
    if (
      checkpoint.lang === lang &&
      checkpoint.themesKey === themesKey &&
      (!best || checkpoint.text.length > best.text.length) &&
      code.startsWith(checkpoint.text)
    ) {
      best = checkpoint;
    }
  }
  return best;
}

function rememberCheckpoint(checkpoint: StreamCheckpoint): void {
  const index = checkpoints.indexOf(checkpoint);
  if (index !== -1) checkpoints.splice(index, 1);
  checkpoints.unshift(checkpoint);
  if (checkpoints.length > MAX_CHECKPOINTS) checkpoints.pop();
}

function toHighlightedLines(
  tokens: ReturnType<Highlighter["codeToTokens"]>["tokens"],
): HighlightedLine[] {
  return tokens.map((line) =>
    line.map((token) => ({
      content: token.content,
      color: token.color,
      htmlStyle: token.htmlStyle,
    })),
  );
}

interface TokenizeArgs {
  highlighter: Highlighter;
  code: string;
  lang: BundledLanguage;
  themes: readonly string[];
  cacheKey: string;
}

function tokenizeFully({
  highlighter,
  code,
  lang,
  themes,
}: TokenizeArgs): HighlightResult {
  const shikiResult = highlighter.codeToTokens(code, {
    lang,
    themes: { light: themes[0], dark: themes[1] },
  });
  return {
    tokens: toHighlightedLines(shikiResult.tokens),
    fg: shikiResult.fg,
    bg: shikiResult.bg,
  };
}

/**
 * Tokenizes `code` starting from the longest checkpoint it extends, so a
 * block that grows with every streamed chunk costs only its new lines.
 * TextMate grammars tokenize line by line from the previous line's state,
 * so the result is the same as tokenizing the whole block again.
 */
function tokenizeIncrementally({
  highlighter,
  code,
  lang,
  themes,
  cacheKey,
}: TokenizeArgs): HighlightResult {
  const themesKey = themes.join(",");
  const checkpoint = findCheckpoint(code, lang, themesKey);
  const start = checkpoint?.text.length ?? 0;
  const rest = code.slice(start);
  // The last line can still grow even when it ends in a newline: Streamdown
  // passes an open fence's body with a newline after the partial line.
  const lastNewline = rest.replace(/\n$/, "").lastIndexOf("\n");
  const options = {
    lang,
    themes: { light: themes[0], dark: themes[1] },
  };

  let lines = checkpoint?.lines ?? [];
  let grammarState = checkpoint?.grammarState;

  if (lastNewline !== -1) {
    // Shiki splits lines on /\r?\n/, so a CRLF line must not keep its \r.
    const completeLines = rest.slice(0, lastNewline).replace(/\r$/, "");
    const complete = highlighter.codeToTokens(completeLines, {
      ...options,
      grammarState,
    });
    lines = lines.concat(toHighlightedLines(complete.tokens));
    grammarState = complete.grammarState;
  }

  const tail = highlighter.codeToTokens(rest.slice(lastNewline + 1), {
    ...options,
    grammarState,
  });

  if (
    checkpoint &&
    checkpoint.lastCode !== code &&
    code.startsWith(checkpoint.lastCode.replace(/\n$/, ""))
  ) {
    demoteToPartial(checkpoint.lastCacheKey);
  }

  const next: StreamCheckpoint = checkpoint ?? {
    lang,
    themesKey,
    text: "",
    lines: [],
    grammarState: undefined,
    lastCacheKey: cacheKey,
    lastCode: code,
  };
  if (lastNewline !== -1 && grammarState) {
    next.text = code.slice(0, start + lastNewline + 1);
    next.lines = lines;
    next.grammarState = grammarState;
  }
  next.lastCacheKey = cacheKey;
  next.lastCode = code;
  rememberCheckpoint(next);

  return {
    tokens: lines.concat(toHighlightedLines(tail.tokens)),
    fg: tail.fg,
    bg: tail.bg,
  };
}

function tokenize(args: TokenizeArgs): HighlightResult {
  try {
    return tokenizeIncrementally(args);
  } catch (error) {
    checkpoints.length = 0;
    console.warn("[Shiki] Incremental highlight failed, retrying:", error);
    return tokenizeFully(args);
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

      const cached = getCachedResult(cacheKey);
      if (cached) {
        return cached;
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

          const result = tokenize({
            highlighter,
            code,
            lang: finalLang,
            themes,
            cacheKey,
          });

          setBounded(tokenCache, cacheKey, result, MAX_CACHE_SIZE);

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
