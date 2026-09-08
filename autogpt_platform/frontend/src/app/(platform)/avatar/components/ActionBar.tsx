import { Button } from "@/components/atoms/Button/Button";
import {
  ArrowReloadHorizontalIcon,
  Download04Icon,
  Link04Icon,
  ShuffleIcon,
} from "@hugeicons/core-free-icons";

interface Props {
  onShuffle: () => void;
  onReset: () => void;
  onCopyLink: () => void;
  onDownloadSvg: () => void;
  onDownloadPng: () => void;
  isExporting: boolean;
}

export function ActionBar({
  onShuffle,
  onReset,
  onCopyLink,
  onDownloadSvg,
  onDownloadPng,
  isExporting,
}: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        variant="secondary"
        size="small"
        leadingIcon={ShuffleIcon}
        onClick={onShuffle}
      >
        Shuffle
      </Button>
      <Button
        variant="ghost"
        size="small"
        leadingIcon={ArrowReloadHorizontalIcon}
        onClick={onReset}
      >
        Reset
      </Button>
      <span className="flex-1" />
      <Button
        variant="outline"
        size="small"
        leadingIcon={Link04Icon}
        onClick={onCopyLink}
      >
        Copy link
      </Button>
      <Button
        variant="outline"
        size="small"
        leadingIcon={Download04Icon}
        onClick={onDownloadSvg}
      >
        SVG
      </Button>
      <Button
        variant="primary"
        size="small"
        leadingIcon={Download04Icon}
        onClick={onDownloadPng}
        loading={isExporting}
      >
        PNG
      </Button>
    </div>
  );
}
