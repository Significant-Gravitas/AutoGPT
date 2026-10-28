import { useGetV2AdminReadSkillSubmissionFile } from "@/app/api/__generated__/endpoints/admin/admin";
import { useGetV2ReadMarketplaceSkillFile } from "@/app/api/__generated__/endpoints/store/store";
import { okData } from "@/app/api/helpers";

export type SkillPackageSource =
  | { kind: "listing"; slug: string }
  | { kind: "submission"; versionId: string };

interface Args {
  source: SkillPackageSource;
  path: string | null;
}

export function useSkillFileViewer({ source, path }: Args) {
  const isListing = source.kind === "listing";
  // Both are declared and one is enabled: a listing reads the approved
  // version, a submission goes through the admin route that can see a
  // version nobody has approved.
  const listing = useGetV2ReadMarketplaceSkillFile(
    isListing ? source.slug : "",
    path ?? "",
    undefined,
    { query: { enabled: isListing && path !== null, retry: false } },
  );
  const submission = useGetV2AdminReadSkillSubmissionFile(
    isListing ? "" : source.versionId,
    path ?? "",
    { query: { enabled: !isListing && path !== null, retry: false } },
  );

  const query = isListing ? listing : submission;
  return {
    content: okData(query.data) ?? null,
    isLoading: query.isPending && path !== null,
    // A refusal is a state to render, not a failure to hide: only the
    // endpoint knows whether a file is one this viewer can serve.
    unavailable: query.isError
      ? describeRefusal((query.error as { status?: number } | null)?.status)
      : null,
  };
}

function describeRefusal(status: number | undefined): string {
  if (status === 413) return "This file is too large to preview here.";
  if (status === 415) return "This file isn't text, so it can't be previewed.";
  return "Couldn't load this file.";
}
