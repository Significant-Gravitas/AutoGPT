/**
 * The object a tool call's argument JSON describes so far, read without
 * waiting for it to close: every finished top-level member, then the member
 * still being written when its key is complete (a string value as far as it
 * goes, anything else as null). The AI SDK reads a streaming part's input the
 * same way; `completeStreamingInput` then keeps only the finished members.
 * Text that is not the start of an object reads as undefined.
 */
export function readPartialToolInput(
  text: string,
): Record<string, unknown> | undefined {
  const body = text.trimStart();
  if (!body.startsWith("{")) return undefined;
  const whole = tryParse(body);
  if (whole !== undefined) return isRecord(whole) ? whole : undefined;

  const result: Record<string, unknown> = {};
  let depth = 0;
  let inString = false;
  let memberStart = 1;
  for (let i = 0; i < body.length; i++) {
    const ch = body[i];
    if (inString) {
      if (ch === "\\") i++;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{" || ch === "[") depth++;
    else if (ch === "}" || ch === "]") depth--;
    else if (ch === "," && depth === 1) {
      Object.assign(result, member(body.slice(memberStart, i)));
      memberStart = i + 1;
    }
  }
  const last = body.slice(memberStart);
  const finished = member(last);
  Object.assign(
    result,
    Object.keys(finished).length ? finished : openMember(last),
  );
  return result;
}

const KEY_RE = /^\s*("(?:[^"\\]|\\.)*")\s*:\s*/;

function member(text: string): Record<string, unknown> {
  const parsed = tryParse(`{${text}}`);
  return isRecord(parsed) ? parsed : {};
}

function openMember(text: string): Record<string, unknown> {
  const key = KEY_RE.exec(text);
  if (!key) return {};
  const name = tryParse(key[1]);
  if (typeof name !== "string") return {};
  const value = text.slice(key[0].length);
  const partial = value.startsWith('"')
    ? tryParse(`${value.replace(/\\$/, "")}"`)
    : undefined;
  return { [name]: typeof partial === "string" ? partial : null };
}

function tryParse(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
