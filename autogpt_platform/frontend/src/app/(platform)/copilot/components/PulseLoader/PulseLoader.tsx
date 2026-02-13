import { cn } from "@/lib/utils";
import styles from "./PulseLoader.module.css";

interface Props {
  size?: number;
  className?: string;
}

export function PulseLoader({ size = 24, className }: Props) {
  return (
    <div
      className={cn(styles.loader, className)}
      style={{ width: size, height: size }}
    />
  );
}
