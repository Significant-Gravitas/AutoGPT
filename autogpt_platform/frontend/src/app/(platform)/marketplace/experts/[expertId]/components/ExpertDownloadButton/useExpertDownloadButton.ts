import { downloadExpertTemplatePackage } from "@/app/api/__generated__/endpoints/experts/experts";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { downloadFile } from "@/lib/utils";
import { useState } from "react";
import { getDownloadErrorDescription, getPackageFilename } from "./helpers";

interface Args {
  templateId: string;
  name: string;
}

export function useExpertDownloadButton({ templateId, name }: Args) {
  const { toast } = useToast();
  const [isDownloading, setIsDownloading] = useState(false);

  async function handleDownload() {
    setIsDownloading(true);
    try {
      const res = await downloadExpertTemplatePackage(templateId);
      if (res.status !== 200) {
        throw new Error(`Failed to download expert (HTTP ${res.status})`);
      }
      downloadFile(
        getPackageFilename(res.headers.get("Content-Disposition"), name),
        res.data,
      );
    } catch (error) {
      toast({
        title: "Couldn't download this expert",
        description: getDownloadErrorDescription(error),
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  }

  return { isDownloading, handleDownload };
}
