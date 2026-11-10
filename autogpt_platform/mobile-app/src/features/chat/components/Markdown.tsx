import { Fragment } from "react";
import { View } from "react-native";
import { useMarkdown, type MarkedStyles } from "react-native-marked";

import { colors, radius, spacing, type } from "@/lib/theme";

interface Props {
  text: string;
}

export function Markdown({ text }: Props) {
  const nodes = useMarkdown(text, {
    colorScheme: "dark",
    styles: markedStyles,
  });

  return (
    <View>
      {nodes.map((node, index) => (
        <Fragment key={index}>{node}</Fragment>
      ))}
    </View>
  );
}

// Module level, so the parser the hook memoises on these stays the same.
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
