"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  hasUpdate?: boolean;
  latestVersion?: number;
  hasUnpublishedChanges?: boolean;
  currentVersion?: number;
  isUpdating?: boolean;
  onUpdate?: () => void;
  onPublish?: () => void;
  onViewChanges?: () => void;
}

export function MarketplaceBanners({
  hasUpdate,
  latestVersion,
  hasUnpublishedChanges,
  isUpdating,
  onUpdate,
  onPublish,
}: Props) {
  const renderUpdateBanner = () => {
    if (hasUpdate && latestVersion) {
      return (
        <div className="mb-6 rounded-lg bg-zinc-50 p-4">
          <div className="flex flex-col gap-3">
            <div>
              <Text variant="large-medium" tone="primary" className="mb-2">
                Update available
              </Text>
              <Text variant="body" tone="secondary">
                You should update your agent in order to get the latest / best
                results
              </Text>
            </div>
            {onUpdate && (
              <div className="flex justify-start">
                <Button size="small" onClick={onUpdate} disabled={isUpdating}>
                  {isUpdating ? "Updating..." : "Update agent"}
                </Button>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  const renderUnpublishedChangesBanner = () => {
    if (hasUnpublishedChanges) {
      return (
        <div className="mb-6 rounded-lg bg-zinc-50 p-4">
          <div className="flex flex-col gap-3">
            <div>
              <Text variant="large-medium" tone="primary" className="mb-2">
                Unpublished changes
              </Text>
              <Text variant="body" tone="secondary">
                You&apos;ve made changes to this agent that aren&apos;t
                published yet. Would you like to publish the latest version?
              </Text>
            </div>
            {onPublish && (
              <div className="flex justify-start">
                <Button size="small" onClick={onPublish}>
                  Publish changes
                </Button>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <>
      {renderUpdateBanner()}
      {renderUnpublishedChangesBanner()}
    </>
  );
}
