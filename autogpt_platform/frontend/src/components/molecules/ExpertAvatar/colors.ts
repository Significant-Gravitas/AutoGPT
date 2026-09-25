import { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
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
  const category = getExpertTopic(role, categories);
  if (category === "otto") return "#B6A4C8";
  return (
    catalog.avatars.find((avatar) => avatar.id === category)?.hex ?? "#B5ADA0"
  );
}

/** A stored category when the expert has one, else the topic its role reads
 *  like. "otto" is a reserved identity rather than a category. */
function getExpertTopic(
  role: string | null | undefined,
  categories?: string[],
): string | undefined {
  const stored = categories?.find((category) =>
    catalog.avatars.some((avatar) => avatar.id === category.toLowerCase()),
  );
  return (
    stored?.toLowerCase() ??
    TOPIC_PATTERNS.find(([pattern]) => pattern.test(role ?? ""))?.[1]
  );
}

/** The category an avatar should be generated in. Every expert has one, so an
 *  unrecognised role falls back to the neutral warm stone of content. */
export function getExpertCategory(
  role: string | null | undefined,
  categories?: string[],
): ExpertAvatarRequestCategory {
  const topic = getExpertTopic(role, categories);
  return (
    Object.values(ExpertAvatarRequestCategory).find(
      (category) => category === topic,
    ) ?? "content"
  );
}

export function expertPastel(hex: string): string {
  return `color-mix(in srgb, ${hex} 24%, white)`;
}
