export interface CsvPreview {
  headers: string[];
  rows: string[][];
}

const CSV_DELIMITERS = [",", ";", "\t", "|"] as const;

function sniffDelimiter(firstLine: string): string {
  let best = ",";
  let bestCount = -1;
  for (const delimiter of CSV_DELIMITERS) {
    const count = firstLine.split(delimiter).length - 1;
    if (count > bestCount) {
      best = delimiter;
      bestCount = count;
    }
  }
  return best;
}

// Quote-aware CSV field tokenizer (RFC 4180): respects quoted fields that
// contain the delimiter and "" escaped quotes. Keeps preview tables correct for
// values like `"Doe, John",42`.
function splitCsvLine(line: string, delimiter: string): string[] {
  const cells: string[] = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (inQuotes) {
      if (char === '"' && line[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        cell += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === delimiter) {
      cells.push(cell);
      cell = "";
    } else {
      cell += char;
    }
  }
  cells.push(cell);
  return cells.map((value) => value.trim());
}

export function parseCsv(
  text: string,
  opts: { maxRows?: number; maxCols?: number } = {},
): CsvPreview | null {
  const maxRows = opts.maxRows ?? 6;
  const maxCols = opts.maxCols ?? 6;

  // The byte-capped fetch can cut mid-row, leaving a trailing partial line with
  // no terminating newline. Only drop the last line in that truncated case so
  // complete files keep every row.
  const truncated = text.length > 0 && !/\r?\n\s*$/.test(text);
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (truncated && lines.length > 1) lines.pop();
  if (lines.length === 0) return null;

  const delimiter = sniffDelimiter(lines[0]);
  const cells = lines
    .slice(0, maxRows)
    .map((line) => splitCsvLine(line, delimiter).slice(0, maxCols));

  const [headers, ...rows] = cells;
  return { headers, rows };
}

function unfold(text: string): string[] {
  // RFC 5545/6350 line folding: continuation lines start with a space or tab.
  const raw = text.split(/\r?\n/);
  const lines: string[] = [];
  for (const line of raw) {
    if ((line.startsWith(" ") || line.startsWith("\t")) && lines.length > 0) {
      lines[lines.length - 1] += line.slice(1);
    } else {
      lines.push(line);
    }
  }
  return lines;
}

function splitProperty(line: string): {
  name: string;
  params: string;
  value: string;
} {
  const colon = line.indexOf(":");
  if (colon === -1) return { name: "", params: "", value: "" };
  const head = line.slice(0, colon);
  const value = line.slice(colon + 1);
  const semi = head.indexOf(";");
  if (semi === -1) return { name: head.toUpperCase(), params: "", value };
  return {
    name: head.slice(0, semi).toUpperCase(),
    params: head.slice(semi + 1).toLowerCase(),
    value,
  };
}

export interface IcsPreview {
  summary?: string;
  start?: string;
  end?: string;
  location?: string;
}

export function parseIcs(text: string): IcsPreview | null {
  const result: IcsPreview = {};
  for (const line of unfold(text)) {
    const { name, value } = splitProperty(line);
    if (name === "SUMMARY") result.summary = value;
    else if (name === "DTSTART") result.start = value;
    else if (name === "DTEND") result.end = value;
    else if (name === "LOCATION") result.location = value;
    if (line.toUpperCase().startsWith("END:VEVENT")) break;
  }
  return result.summary || result.start ? result : null;
}

export interface VcardPreview {
  name?: string;
  org?: string;
  title?: string;
  tel?: string;
  email?: string;
  photo?: string;
}

export function parseVcard(text: string): VcardPreview | null {
  const result: VcardPreview = {};
  for (const line of unfold(text)) {
    const { name, params, value } = splitProperty(line);
    if (name === "FN") result.name = value;
    else if (name === "ORG") result.org = value.replace(/;+$/, "");
    else if (name === "TITLE") result.title = value;
    else if (name === "TEL" && !result.tel) result.tel = value;
    else if (name === "EMAIL" && !result.email) result.email = value;
    else if (name === "PHOTO" && !result.photo)
      result.photo = toPhotoSrc(params, value);
  }
  return result.name ? result : null;
}

function toPhotoSrc(params: string, value: string): string | undefined {
  if (value.startsWith("data:") || value.startsWith("http")) return value;
  // vCard 3.0: PHOTO;ENCODING=b;TYPE=JPEG:<base64>
  if (params.includes("b") || params.includes("base64")) {
    const typeMatch = params.match(/type=([a-z0-9]+)/);
    const subtype = typeMatch ? typeMatch[1] : "jpeg";
    return `data:image/${subtype};base64,${value.replace(/\s+/g, "")}`;
  }
  return undefined;
}
