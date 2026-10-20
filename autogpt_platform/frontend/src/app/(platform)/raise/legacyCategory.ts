import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";

const TOPIC_PATTERNS: Array<[RegExp, ExpertAvatarRequestCategory]> = [
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

export function legacyCategoryForRole(
  role: string,
): ExpertAvatarRequestCategory {
  return (
    TOPIC_PATTERNS.find(([pattern]) => pattern.test(role))?.[1] ?? "content"
  );
}
