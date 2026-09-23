"use client";

import { downloadAsAdmin } from "@/app/(platform)/admin/marketplace/actions";
import { Button } from "@/components/__legacy__/ui/button";
import { agentGraphExportFilename, exportAsJSONFile } from "@/lib/utils";
import { ExternalLink } from "lucide-react";
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
      size="sm"
      variant="outline"
      onClick={handleDownload}
      disabled={isLoading}
    >
      <ExternalLink className="mr-2 h-4 w-4" />
      {isLoading ? "Downloading..." : "Download"}
    </Button>
  );
}
