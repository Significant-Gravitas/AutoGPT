import { Fragment } from "react";
import { View } from "react-native";
import { useMarkdown, type MarkedStyles } from "react-native-marked";

import { colors, radius, spacing, type } from "@/lib/theme";

import { safeRenderer } from "./safeRenderer";

interface Props {
  text: string;
}

export function Markdown({ text }: Props) {
  const nodes = useMarkdown(text, {
    colorScheme: "dark",
    styles: markedStyles,
    renderer: safeRenderer,
  });

  return (
    <View>
      {nodes.map((node, index) => (
        <Fragment key={index}>{node}</Fragment>
      ))}
    </View>
  );
}

// Module level, like the renderer, so the hook keeps its memoised parser.
const markedStyles: MarkedStyles = {
  text: { ...type.body, color: colors.text },
  paragraph: { paddingVertical: spacing.xs },
  link: { color: colors.brand },
  li: { ...type.body, color: colors.text },
  codespan: { backgroundColor: colors.surface, color: colors.text },
  code: {
    backgroundColor: colors.surface,
    borderRadius: radius.sm,
    padding: spacing.md,
    marginVertical: spacing.xs,
  },
  codeText: { ...type.small, color: colors.text },
};
