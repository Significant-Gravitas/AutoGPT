import 'package:auto_gpt_flutter_client/views/chat/json_code_snippet_view.dart';
import 'package:flutter/material.dart';

class AgentMessageTile extends StatefulWidget {
  final String message;

  const AgentMessageTile({
    Key? key,
    required this.message, // The agent message to be displayed
  }) : super(key: key);

  @override
  _AgentMessageTileState createState() => _AgentMessageTileState();
}

class _AgentMessageTileState extends State<AgentMessageTile> {
  bool isExpanded = false; // State variable to toggle expanded view

  @override
  Widget build(BuildContext context) {
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
                            widget.message,
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
                  const ClipRect(
                    child: SizedBox(
                      height: 200,
                      child: Padding(
                        padding: EdgeInsets.only(
                            right: 20), // Padding for the right side
                        child: JsonCodeSnippetView(
                          // JSON code snippet view
                          jsonString:
                              "{\"input\":\"Washington\",\"additional_input\":{\"file_to_refactor\":\"models.py\"},\"task_id\":\"50da533e-3904-4401-8a07-c49adf88b5eb\",\"step_id\":\"6bb1801a-fd80-45e8-899a-4dd723cc602e\",\"name\":\"Writetofile\",\"status\":\"created\",\"output\":\"Iamgoingtousethewrite_to_filecommandandwriteWashingtontoafilecalledoutput.txt<write_to_file('output.txt','Washington')\",\"additional_output\":{\"tokens\":7894,\"estimated_cost\":\"0,24\"},\"artifacts\":[],\"is_last\":false}",
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
