import { Button } from "@/components/atoms/Button/Button";
import { Download04Icon } from "@hugeicons/core-free-icons";
import { useExpertDownloadButton } from "./useExpertDownloadButton";

interface Props {
  templateId: string;
  name: string;
}

export function ExpertDownloadButton({ templateId, name }: Props) {
  const { isDownloading, handleDownload } = useExpertDownloadButton({
    templateId,
    name,
  });

  return (
    <Button
      variant="icon"
      size="icon"
      leadingIcon={Download04Icon}
      loading={isDownloading}
      onClick={handleDownload}
      aria-label="Download as file"
      data-testid="expert-download-button"
      className="size-9 flex-shrink-0 p-0"
    />
  );
}
