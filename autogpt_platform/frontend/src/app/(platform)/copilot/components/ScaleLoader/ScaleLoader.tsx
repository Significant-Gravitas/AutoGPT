import { cn } from "@/lib/utils";
import styles from "./ScaleLoader.module.css";

interface Props {
  size?: number;
  className?: string;
}

export function ScaleLoader({ size = 48, className }: Props) {
  return (
    <div
      className={cn(styles.loader, className)}
      style={{ width: size, height: size }}
    />
  );
}
