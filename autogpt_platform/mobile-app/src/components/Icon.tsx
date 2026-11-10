import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react-native";

import { colors } from "@/lib/theme";

interface Props {
  icon: IconSvgElement;
  size?: number;
  color?: string;
}

export function Icon({ icon, size = 22, color = colors.text }: Props) {
  return <HugeiconsIcon icon={icon} size={size} color={color} strokeWidth={2} />;
}
