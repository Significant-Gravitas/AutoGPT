import { Fragment } from "react";
import { View } from "react-native";
import { useMarkdown, type MarkedStyles } from "react-native-marked";

import {
  palettes,
  radius,
  spacing,
  type,
  useScheme,
  type Colors,
  type Scheme,
} from "@/lib/theme";

import { safeRenderers } from "./safeRenderer";

interface Props {
  text: string;
}

export function Markdown({ text }: Props) {
  const scheme = useScheme();
  const nodes = useMarkdown(text, {
    colorScheme: scheme,
    styles: markedStyles[scheme],
    renderer: safeRenderers[scheme],
  });

  return (
    <View>
      {nodes.map((node, index) => (
        <Fragment key={index}>{node}</Fragment>
      ))}
    </View>
  );
}

function buildMarkedStyles(colors: Colors): MarkedStyles {
  return {
    text: { ...type.body, color: colors.text },
    paragraph: { paddingVertical: spacing.xs },
    link: { color: colors.accentText },
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
}

// Module level, like the renderers, so the hook keeps its memoised parser.
const markedStyles: Record<Scheme, MarkedStyles> = {
  dark: buildMarkedStyles(palettes.dark),
  light: buildMarkedStyles(palettes.light),
};
