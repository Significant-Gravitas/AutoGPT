import * as WebBrowser from "expo-web-browser";
import { Fragment, memo, type ReactNode } from "react";
import {
  ScrollView,
  StyleSheet,
  Text,
  View,
  type TextStyle,
} from "react-native";

import { colors, radius, spacing, type } from "../../../design/theme";
import { parseMarkdown, type Block, type Inline } from "../markdown";

interface Props {
  text: string;
  /** Muted styling for secondary content such as reasoning. */
  tone?: "primary" | "secondary";
}

export const MarkdownText = memo(function MarkdownText({
  text,
  tone = "primary",
}: Props) {
  const color = tone === "primary" ? colors.label : colors.secondaryLabel;
  return (
    <View style={styles.blocks}>
      {parseMarkdown(text).map((block, index) => (
        <BlockView key={index} block={block} color={color} />
      ))}
    </View>
  );
});

function BlockView({
  block,
  color,
}: {
  block: Block;
  color: TextStyle["color"];
}) {
  switch (block.kind) {
    case "paragraph":
      return (
        <Text selectable style={[type.body, { color }]}>
          {renderInline(block.inline)}
        </Text>
      );
    case "heading":
      return (
        <Text
          selectable
          style={[block.level <= 2 ? type.title3 : type.headline, { color }]}
          accessibilityRole="header"
        >
          {renderInline(block.inline)}
        </Text>
      );
    case "code":
    case "preformatted":
      return (
        <ScrollView
          horizontal
          style={styles.code}
          contentContainerStyle={styles.codeContent}
        >
          <Text selectable style={[type.mono, { color: colors.label }]}>
            {block.text}
          </Text>
        </ScrollView>
      );
    case "list":
      return (
        <View style={styles.list}>
          {block.items.map((item, index) => (
            <View key={index} style={styles.listItem}>
              <Text style={[type.body, styles.marker, { color }]}>
                {block.ordered ? `${block.start + index}.` : "•"}
              </Text>
              <Text selectable style={[type.body, styles.listText, { color }]}>
                {renderInline(item)}
              </Text>
            </View>
          ))}
        </View>
      );
    case "quote":
      return (
        <View style={styles.quote}>
          <Text
            selectable
            style={[type.body, { color: colors.secondaryLabel }]}
          >
            {renderInline(block.inline)}
          </Text>
        </View>
      );
    case "rule":
      return <View style={styles.rule} />;
  }
}

function renderInline(nodes: Inline[]): ReactNode {
  return nodes.map((node, index) => {
    switch (node.kind) {
      case "text":
        return <Fragment key={index}>{node.text}</Fragment>;
      case "code":
        return (
          <Text key={index} style={styles.inlineCode}>
            {node.text}
          </Text>
        );
      case "strong":
        return (
          <Text key={index} style={styles.strong}>
            {renderInline(node.children)}
          </Text>
        );
      case "em":
        return (
          <Text key={index} style={styles.em}>
            {renderInline(node.children)}
          </Text>
        );
      case "strike":
        return (
          <Text key={index} style={styles.strike}>
            {renderInline(node.children)}
          </Text>
        );
      case "link": {
        const href = node.href;
        if (!href)
          return <Fragment key={index}>{renderInline(node.children)}</Fragment>;
        return (
          <Text
            key={index}
            style={styles.link}
            accessibilityRole="link"
            onPress={() => void WebBrowser.openBrowserAsync(href)}
          >
            {renderInline(node.children)}
          </Text>
        );
      }
    }
  });
}

const styles = StyleSheet.create({
  blocks: { gap: spacing.sm },
  code: { backgroundColor: colors.fill, borderRadius: radius.sm },
  codeContent: { padding: spacing.md },
  list: { gap: spacing.xs },
  listItem: { flexDirection: "row", gap: spacing.sm },
  marker: { minWidth: 18, textAlign: "right" },
  listText: { flex: 1 },
  quote: {
    borderLeftWidth: 3,
    borderLeftColor: colors.separator,
    paddingLeft: spacing.md,
  },
  rule: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: colors.separator,
    marginVertical: spacing.sm,
  },
  inlineCode: { ...type.mono, backgroundColor: colors.fill },
  strong: { fontWeight: "600" },
  em: { fontStyle: "italic" },
  strike: { textDecorationLine: "line-through" },
  link: { color: colors.link, textDecorationLine: "underline" },
});
