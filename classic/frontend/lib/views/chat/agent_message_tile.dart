import 'dart:convert';

import 'package:auto_gpt_flutter_client/models/chat.dart';
import 'package:auto_gpt_flutter_client/views/chat/json_code_snippet_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';

class AgentMessageTile extends StatefulWidget {
  final Chat chat;
  final VoidCallback onArtifactsButtonPressed;

  const AgentMessageTile({
    Key? key,
    required this.chat,
    required this.onArtifactsButtonPressed,
  }) : super(key: key);

  @override
  _AgentMessageTileState createState() => _AgentMessageTileState();
}

class _AgentMessageTileState extends State<AgentMessageTile> {
  bool isExpanded = false;

  @override
  Widget build(BuildContext context) {
    String jsonString = jsonEncode(widget.chat.jsonResponse);
    int artifactsCount = widget.chat.artifacts.length;

    bool containsMarkdown(String text) {
      // Regular expression to detect Markdown patterns like headers, bold, links, etc.
      final RegExp markdownPattern = RegExp(
        r'(?:\*\*|__).*?(?:\*\*|__)|' + // Bold
            r'(?:\*|_).*?(?:\*|_)|' + // Italic
            r'\[.*?\]\(.*?\)|' + // Links
            r'!\[.*?\]\(.*?\)|' + // Images
            r'#{1,6}.*|' + // Headers
            r'```.*?```', // Fenced code blocks
        dotAll: true, // To match across multiple lines
        caseSensitive: false,
      );

      return markdownPattern.hasMatch(text);
    }

    bool hasMarkdown = containsMarkdown(widget.chat.message);

    return LayoutBuilder(
      builder: (context, constraints) {
        double chatViewWidth = constraints.maxWidth;
        double tileWidth = (chatViewWidth >= 1000) ? 900 : chatViewWidth - 40;

        return Align(
          alignment: Alignment.center,
          child: Container(
            width: tileWidth,
            margin: const EdgeInsets.symmetric(vertical: 8),
            padding: const EdgeInsets.symmetric(horizontal: 20),
            decoration: BoxDecoration(
              color: Colors.white,
              border: Border.all(color: Colors.black, width: 0.5),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                Container(
                  constraints: const BoxConstraints(minHeight: 50),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      const Text(
                        "Agent",
                        style: TextStyle(
                          color: Colors.black,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(width: 20),
                      Expanded(
                        child: Container(
                          padding: const EdgeInsets.fromLTRB(0, 10, 20, 10),
                          child: SingleChildScrollView(
                            child: hasMarkdown
                                ? Markdown(
                                    data: widget.chat.message,
                                    shrinkWrap: true,
                                    styleSheet: MarkdownStyleSheet.fromTheme(
                                            Theme.of(context))
                                        .copyWith(
                                      blockquoteDecoration: BoxDecoration(
                                        color: Colors
                                            .black, // Background color for blockquotes
                                        border: Border(
                                          left: BorderSide(
                                            color: Colors.grey,
                                            width: 4.0,
                                          ),
                                        ),
                                      ),
                                      blockquoteAlign: WrapAlignment.start,
                                      blockquotePadding: const EdgeInsets.all(
                                          10.0), // Padding for blockquotes
                                      codeblockDecoration: BoxDecoration(
                                        color: Colors.grey[
                                            200], // Background color for code blocks
                                        borderRadius:
                                            BorderRadius.circular(4.0),
                                      ),
                                      codeblockPadding: const EdgeInsets.all(
                                          10.0), // Padding for code blocks
                                      code: TextStyle(
                                        backgroundColor: Colors.grey[
                                            200], // Background color for inline code
                                        fontFamily: 'monospace',
                                      ),
                                    ),
                                  )
                                : SelectableText(widget.chat.message,
                                    maxLines: null),
                          ),
                        ),
                      ),
                      ElevatedButton(
                        onPressed: widget.onArtifactsButtonPressed,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: Colors.black,
                          side: const BorderSide(color: Colors.black),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                        child: Text('$artifactsCount Artifacts'),
                      ),
                      const SizedBox(width: 20),
                      // Expand/Collapse button
                      IconButton(
                        splashRadius: 0.1,
                        icon: Icon(isExpanded
                            ? Icons.keyboard_arrow_up
                            : Icons.keyboard_arrow_down),
                        onPressed: () {
                          setState(() {
                            isExpanded = !isExpanded;
                          });
                        },
                      ),
                    ],
                  ),
                ),
                // Expanded view with JSON code snippet and copy button
                if (isExpanded) ...[
                  const Divider(),
                  ClipRect(
                    child: SizedBox(
                      height: 200,
                      child: Padding(
                        padding: const EdgeInsets.only(right: 20),
                        child: JsonCodeSnippetView(
                          jsonString: jsonString,
                        ),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}
