import { getV2GetSpecificAgent } from "@/app/api/__generated__/endpoints/store/store";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";
import type { RaiseAttachmentDraft } from "../../helpers";
import {
  isHitSelected,
  marketplaceKey,
  MAX_ATTACHMENTS,
  type KitSearchScope,
  type SearchHit,
} from "./helpers";
import { useKitSearch } from "./useKitSearch";

interface Args {
  scope: KitSearchScope;
  existingCount?: number;
  onSubmit: (attachments: RaiseAttachmentDraft[]) => void;
  onSkip: () => void;
}

export function useAttachmentPicker({
  scope,
  existingCount = 0,
  onSubmit,
  onSkip,
}: Args) {
  const search = useKitSearch(scope);
  const [attachments, setAttachments] = useState<RaiseAttachmentDraft[]>([]);
  const [pendingKey, setPendingKey] = useState<string | null>(null);
  const atCap = existingCount + attachments.length >= MAX_ATTACHMENTS;

  function removeAttachment(key: string) {
    setAttachments((current) =>
      current.filter((attachment) => attachmentKey(attachment) !== key),
    );
  }

  async function addHit(hit: SearchHit) {
    if (hit.source === "marketplace") {
      await addMarketplaceHit(hit);
      return;
    }
    setAttachments((current) => {
      if (existingCount + current.length >= MAX_ATTACHMENTS) return current;
      if (isHitSelected(current, hit)) return current;
      return [...current, toDraft(hit, hit.id)];
    });
  }

  async function addMarketplaceHit(hit: SearchHit) {
    if (!hit.creator || !hit.slug || pendingKey) return;
    setPendingKey(hit.key);
    try {
      const details = await getV2GetSpecificAgent(
        hit.creator.toLowerCase(),
        hit.slug,
      );
      if (details.status !== 200) {
        throw new Error("unavailable");
      }
      const listingId = details.data.store_listing_version_id;
      setAttachments((current) => {
        if (existingCount + current.length >= MAX_ATTACHMENTS) return current;
        const draft = toDraft(hit, listingId);
        if (isHitSelected(current, hit)) return current;
        return [...current, draft];
      });
    } catch {
      toast({
        title: `Couldn't add ${hit.name}`,
        description: "Something went wrong. Please try again.",
        variant: "destructive",
      });
    } finally {
      setPendingKey(null);
    }
  }

  return {
    ...search,
    attachments,
    pendingKey,
    atCap,
    addHit,
    removeAttachment,
    submit: () => onSubmit(attachments),
    skip: onSkip,
  };
}

function toDraft(hit: SearchHit, id: string): RaiseAttachmentDraft {
  return {
    kind: hit.kind,
    source: hit.source,
    id,
    name: hit.name,
    marketplaceKey:
      hit.creator && hit.slug
        ? marketplaceKey(hit.creator, hit.slug)
        : undefined,
  };
}

export function attachmentKey(attachment: RaiseAttachmentDraft) {
  return `${attachment.source}:${attachment.kind}:${attachment.id}`;
}
