import 'package:flutter/material.dart';
import 'package:flutter_highlight/flutter_highlight.dart';
import 'package:flutter_highlight/themes/github.dart';
import 'package:flutter/services.dart';
import 'dart:convert';

class JsonCodeSnippetView extends StatelessWidget {
  final String jsonString;

  // Constructor to initialize the jsonString that will be displayed
  const JsonCodeSnippetView({
    Key? key,
    required this.jsonString,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Pretty print the JSON using JsonEncoder to format with indentation
    String prettyJson =
        const JsonEncoder.withIndent('  ').convert(json.decode(jsonString));

    return Padding(
      // Padding applied to align the code snippet view within its container
      padding: const EdgeInsets.fromLTRB(30, 30, 0, 30),
      child: Row(
        children: [
          // Expanded widget to ensure the code snippet view takes the available space
          Expanded(
            child: SingleChildScrollView(
              // SingleChildScrollView to make the code snippet scrollable if it overflows
              child: HighlightView(
                // Display the pretty-printed JSON
                prettyJson,
                // Set the language to JSON for syntax highlighting
                language: 'json',
                // Apply a GitHub-like theme for the highlighting
                theme: githubTheme,
                // Padding applied to the code snippet inside the view
                padding: const EdgeInsets.all(12),
                // TextStyle applied to the code snippet (monospace font)
                textStyle: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 12,
                ),
              ),
            ),
          ),
          // SizedBox to create a gap between the code snippet view and the copy button
          const SizedBox(width: 20),
          Material(
            color: Colors.white,
            // IconButton to allow the user to copy the pretty-printed JSON to the clipboard
            child: IconButton(
              icon: const Icon(Icons.copy),
              onPressed: () {
                // Copy the pretty-printed JSON to the clipboard
                Clipboard.setData(ClipboardData(text: prettyJson));
              },
            ),
          ),
        ],
      ),
    );
  }
}
