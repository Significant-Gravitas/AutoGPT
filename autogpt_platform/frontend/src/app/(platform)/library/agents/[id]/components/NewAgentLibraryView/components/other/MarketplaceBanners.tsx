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
        <div
          // TODO(tokens): no exact role for bg-gray-50 (#F9FAFB); no dark roles yet for dark:bg-gray-900 (#111827)
          className="mb-6 rounded-lg bg-gray-50 p-4 dark:bg-gray-900"
        >
          <div className="flex flex-col gap-3">
            <div>
              <Text
                variant="large-medium"
                // TODO(tokens): no exact role for text-neutral-900 (#171717); no dark roles yet for dark:text-neutral-100 (#F5F5F5)
                className="mb-2 text-neutral-900 dark:text-neutral-100"
              >
                Update available
              </Text>
              <Text
                // TODO(tokens): no exact role for text-gray-700 (#374151); no dark roles yet for dark:text-gray-300 (#D1D5DB)
                variant="body"
                className="text-gray-700 dark:text-gray-300"
              >
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
                  // TODO(tokens): no exact role for bg-neutral-800 (#262626), hover:bg-neutral-900 (#171717); no dark roles yet for dark:bg-neutral-700 (#404040), dark:hover:bg-neutral-800 (#262626)
                  className="bg-neutral-800 text-action-foreground hover:bg-neutral-900 dark:bg-neutral-700 dark:hover:bg-neutral-800"
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
        <div
          // TODO(tokens): no exact role for bg-gray-50 (#F9FAFB); no dark roles yet for dark:bg-gray-900 (#111827)
          className="mb-6 rounded-lg bg-gray-50 p-4 dark:bg-gray-900"
        >
          <div className="flex flex-col gap-3">
            <div>
              <Text
                variant="large-medium"
                // TODO(tokens): no exact role for text-neutral-900 (#171717); no dark roles yet for dark:text-neutral-100 (#F5F5F5)
                className="mb-2 text-neutral-900 dark:text-neutral-100"
              >
                Unpublished changes
              </Text>
              <Text
                // TODO(tokens): no exact role for text-gray-700 (#374151); no dark roles yet for dark:text-gray-300 (#D1D5DB)
                variant="body"
                className="text-gray-700 dark:text-gray-300"
              >
                You&apos;ve made changes to this workflow that aren&apos;t
                published yet. Would you like to publish the latest version?
              </Text>
            </div>
            {onPublish && (
              <div className="flex justify-start">
                <Button
                  size="small"
                  onClick={onPublish}
                  // TODO(tokens): no exact role for bg-neutral-800 (#262626), hover:bg-neutral-900 (#171717); no dark roles yet for dark:bg-neutral-700 (#404040), dark:hover:bg-neutral-800 (#262626)
                  className="bg-neutral-800 text-action-foreground hover:bg-neutral-900 dark:bg-neutral-700 dark:hover:bg-neutral-800"
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
