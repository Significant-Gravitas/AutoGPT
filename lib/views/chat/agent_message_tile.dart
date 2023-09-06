import 'dart:convert';

import 'package:auto_gpt_flutter_client/models/chat.dart';
import 'package:auto_gpt_flutter_client/views/chat/json_code_snippet_view.dart';
import 'package:flutter/material.dart';

class AgentMessageTile extends StatefulWidget {
  final Chat chat;

  const AgentMessageTile({
    Key? key,
    required this.chat, // The agent message to be displayed
  }) : super(key: key);

  @override
  _AgentMessageTileState createState() => _AgentMessageTileState();
}

class _AgentMessageTileState extends State<AgentMessageTile> {
  bool isExpanded = false;

  @override
  Widget build(BuildContext context) {
    String jsonString = jsonEncode(widget.chat.jsonResponse);
    return LayoutBuilder(
      builder: (context, constraints) {
        double chatViewWidth = constraints.maxWidth; // Get the chat view width
        double tileWidth = (chatViewWidth >= 1000)
            ? 900
            : chatViewWidth - 40; // Determine tile width

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
                // Container for Agent title, message, and controls
                Container(
                  constraints: const BoxConstraints(minHeight: 50),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      // Agent title
                      const Text(
                        "Agent",
                        style: TextStyle(
                          color: Colors.black,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(width: 20),
                      // Message content
                      Expanded(
                        child: Container(
                          padding: const EdgeInsets.fromLTRB(0, 10, 20, 10),
                          child: Text(
                            widget.chat.message,
                            maxLines: null,
                          ),
                        ),
                      ),
                      // Artifacts button (static for now)
                      ElevatedButton(
                        onPressed: () {},
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: Colors.black,
                          side: const BorderSide(color: Colors.black),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                        child: const Text("2 Artifacts"),
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
                            isExpanded = !isExpanded; // Toggle expanded view
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
                        padding: const EdgeInsets.only(
                            right: 20), // Padding for the right side
                        child: JsonCodeSnippetView(
                          // JSON code snippet view
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
