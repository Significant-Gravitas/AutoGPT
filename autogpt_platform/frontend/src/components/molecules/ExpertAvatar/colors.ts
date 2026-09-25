import catalog from "./catalog.json";

const TOPIC_PATTERNS: Array<[RegExp, string]> = [
  [/head of ai/i, "otto"],
  [/finance|invoic|bookkeep|fundrais|investor/i, "finance"],
  [/research|intelligence|data|analysis/i, "research"],
  [/development|code|dependency|security|engineering|qa/i, "development"],
  [/sales|deal desk|proposal|revops|crm|partnership/i, "sales"],
  [/support|customer|help desk|retention/i, "support"],
  [
    /marketing|social|seo|brand|growth|email|lifecycle|go.to.market|paid ads|performance|communications|\bpr\b/i,
    "marketing",
  ],
  [
    /ops|operations|recruit|hiring|vendor|procurement|contracts|product|people|privacy|compliance|assistant/i,
    "operations",
  ],
  [/content|writing|editor/i, "content"],
];

/** Category hues stay fixed; generated body colors can vary within the roster. */
export function getCategoryHex(
  category: string | null | undefined,
): string | undefined {
  return catalog.avatars.find((avatar) => avatar.id === category?.toLowerCase())
    ?.hex;
}

export function getExpertTopicHex(
  role: string | null | undefined,
  categories?: string[],
): string {
  const stored = categories?.find((category) =>
    catalog.avatars.some((avatar) => avatar.id === category.toLowerCase()),
  );
  const category =
    stored?.toLowerCase() ??
    TOPIC_PATTERNS.find(([pattern]) => pattern.test(role ?? ""))?.[1];
  if (category === "otto") return "#B6A4C8";
  return (
    catalog.avatars.find((avatar) => avatar.id === category)?.hex ?? "#B5ADA0"
  );
}

export function expertPastel(hex: string): string {
  return `color-mix(in srgb, ${hex} 24%, white)`;
}
