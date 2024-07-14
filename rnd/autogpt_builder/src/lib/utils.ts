import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Derived from https://stackoverflow.com/a/7616484 */
export function hashString(str: string): number {
  let hash = 0, chr: number;
  if (str.length === 0) return hash;
  for (let i = 0; i < str.length; i++) {
    chr = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + chr;
    hash |= 0; // Convert to 32bit integer
  }
  return hash;
}

export function beautifyString(name: string): string {
  // Regular expression to identify places to split, considering acronyms
  const result = name
    .replace(/([a-z])([A-Z])/g, '$1 $2')  // Add space before capital letters
    .replace(/([A-Z])([A-Z][a-z])/g, '$1 $2')  // Add space between acronyms and next word
    .replace(/_/g, ' ')  // Replace underscores with spaces
    .replace(/\b\w/g, char => char.toUpperCase());  // Capitalize the first letter of each word
  
  return applyExceptions(result);
};

const exceptionMap = new Map<string, string>([
  ['Auto Gpt', 'AutoGPT'],
  ['Gpt', 'GPT'],
  ['Auto GPT', 'AutoGPT'],
  ['Creds', 'Credentials'],
  ['Id', 'ID'],
  ['Openai', 'OpenAI'],
  ['Api', 'API'],
  ['Url', 'URL'],
  ['Http', 'HTTP'],
  ['Json', 'JSON']
]);

const applyExceptions = (str: string) => {
  Object.keys(exceptionMap).forEach(key => {
    const regex = new RegExp(`\\b${key}\\b`, 'g');
    str = str.replace(regex, exceptionMap.get(key)!);
  });
  return str;
};
