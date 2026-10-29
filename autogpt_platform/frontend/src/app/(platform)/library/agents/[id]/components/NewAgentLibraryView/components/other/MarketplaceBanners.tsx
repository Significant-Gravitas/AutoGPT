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
        <div className="mb-6 rounded-lg bg-surface-muted p-4">
          <div className="flex flex-col gap-3">
            <div>
              <Text
                variant="large-medium"
                className="mb-2 text-foreground-strong"
              >
                Update available
              </Text>
              <Text variant="body" className="text-foreground-body">
                You should update your workflow in order to get the latest /
                best results
              </Text>
            </div>
            {onUpdate && (
              <div className="flex justify-start">
                <Button
                  size="small"
                  onClick={onUpdate}
                  disabled={isUpdating}
                  className="bg-action text-action-foreground hover:bg-action-hover"
                >
                  {isUpdating ? "Updating..." : "Update workflow"}
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
        <div className="mb-6 rounded-lg bg-surface-muted p-4">
          <div className="flex flex-col gap-3">
            <div>
              <Text
                variant="large-medium"
                className="mb-2 text-foreground-strong"
              >
                Unpublished changes
              </Text>
              <Text variant="body" className="text-foreground-body">
                You&apos;ve made changes to this workflow that aren&apos;t
                published yet. Would you like to publish the latest version?
              </Text>
            </div>
            {onPublish && (
              <div className="flex justify-start">
                <Button
                  size="small"
                  onClick={onPublish}
                  className="bg-action text-action-foreground hover:bg-action-hover"
                >
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
