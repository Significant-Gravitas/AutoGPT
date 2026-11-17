import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react-native";

import { useColors } from "@/lib/theme";

interface Props {
  icon: IconSvgElement;
  size?: number;
  color?: string;
}

export function Icon({ icon, size = 22, color }: Props) {
  const colors = useColors();
  return (
    <HugeiconsIcon
      icon={icon}
      size={size}
      color={color ?? colors.text}
      strokeWidth={2}
    />
  );
}
