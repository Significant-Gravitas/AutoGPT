"use client";

import { downloadAsAdmin } from "@/app/(platform)/admin/marketplace/actions";
import { Button } from "@/components/atoms/Button/Button";
import { agentGraphExportFilename, exportAsJSONFile } from "@/lib/utils";
import { LinkSquare01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useState } from "react";

export function DownloadAgentAdminButton({
  storeListingVersionId,
}: {
  storeListingVersionId: string;
}) {
  const [isLoading, setIsLoading] = useState(false);

  const handleDownload = async () => {
    try {
      setIsLoading(true);
      // Call the server action to get the data
      const fileData = await downloadAsAdmin(storeListingVersionId);

      exportAsJSONFile(fileData as object, agentGraphExportFilename(fileData));
    } catch (error) {
      console.error("Download failed:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Button
      size="xs"
      variant="secondary"
      onClick={handleDownload}
      disabled={isLoading}
    >
      <Icon aria-hidden icon={LinkSquare01Icon} className="h-4 w-4" />
      {isLoading ? "Downloading..." : "Download"}
    </Button>
  );
}
