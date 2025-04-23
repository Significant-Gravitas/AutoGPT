import 'package:auto_gpt_flutter_client/models/message_type.dart';
import 'package:auto_gpt_flutter_client/viewmodels/settings_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_queue_viewmodel.dart';
import 'package:auto_gpt_flutter_client/viewmodels/task_viewmodel.dart';
import 'package:auto_gpt_flutter_client/views/chat/agent_message_tile.dart';
import 'package:auto_gpt_flutter_client/views/chat/chat_input_field.dart';
import 'package:auto_gpt_flutter_client/views/chat/loading_indicator.dart';
import 'package:auto_gpt_flutter_client/views/chat/user_message_tile.dart';
import 'package:flutter/material.dart';
import 'package:auto_gpt_flutter_client/viewmodels/chat_viewmodel.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;

class ChatView extends StatefulWidget {
  final ChatViewModel viewModel;

  const ChatView({Key? key, required this.viewModel}) : super(key: key);

  @override
  _ChatViewState createState() => _ChatViewState();
}

class _ChatViewState extends State<ChatView> {
  final ScrollController _scrollController = ScrollController();
  bool _isAtBottom = true;

  @override
  void initState() {
    super.initState();

    // Listen for scroll events and determine whether the scroll is at the bottom
    _scrollController.addListener(() {
      if (_scrollController.position.atEdge) {
        if (_scrollController.position.pixels == 0) {
          _isAtBottom = false;
        } else {
          _isAtBottom = true;
        }
      }
    });

    // Schedule the fetchChatsForTask call for after the initial build
    WidgetsBinding.instance.addPostFrameCallback((_) {
      widget.viewModel.fetchChatsForTask();
    });
  }

  @override
  void dispose() {
    // Dispose of the ScrollController when the widget is removed
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    _scrollController.animateTo(
      _scrollController.position.maxScrollExtent,
      duration: const Duration(milliseconds: 200),
      curve: Curves.easeOut,
    );
  }

  @override
  Widget build(BuildContext context) {
    // TODO: Do we want to have a reference to task view model in this class?
    final taskViewModel = Provider.of<TaskViewModel>(context, listen: true);
    return Scaffold(
      body: Column(
        children: [
          // Chat messages list
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              itemCount: widget.viewModel.chats.length,
              itemBuilder: (context, index) {
                final chat = widget.viewModel.chats[index];

                // If the last message has been built and we're at the bottom of the list, scroll down
                if (index == widget.viewModel.chats.length - 1 && _isAtBottom) {
                  WidgetsBinding.instance.addPostFrameCallback((_) {
                    _scrollToBottom();
                  });
                }

                if (chat.messageType == MessageType.user) {
                  return UserMessageTile(message: chat.message);
                } else {
                  return AgentMessageTile(
                    key: ValueKey(chat.id),
                    chat: chat,
                    onArtifactsButtonPressed: () {
                      // Loop through each artifact and download it using the artifact_id
                      for (var artifact in chat.artifacts) {
                        widget.viewModel
                            .downloadArtifact(chat.taskId, artifact.artifactId);
                      }
                    },
                  );
                }
              },
            ),
          ),
          const SizedBox(height: 10),
          LoadingIndicator(
              isLoading: Provider.of<TaskQueueViewModel>(context, listen: true)
                      .isBenchmarkRunning ||
                  widget.viewModel.isWaitingForAgentResponse ||
                  taskViewModel.isWaitingForAgentResponse),
          const SizedBox(height: 10),
          // Input area
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: ChatInputField(
              onSendPressed: (message) async {
                widget.viewModel.addTemporaryMessage(message);
                try {
                  if (widget.viewModel.currentTaskId != null) {
                    widget.viewModel.sendChatMessage(
                        message,
                        continuousModeSteps: Provider.of<SettingsViewModel>(
                                context,
                                listen: false)
                            .continuousModeSteps);
                  } else {
                    String newTaskId = await taskViewModel.createTask(message);
                    widget.viewModel.setCurrentTaskId(newTaskId);
                    widget.viewModel.sendChatMessage(
                        message,
                        continuousModeSteps: Provider.of<SettingsViewModel>(
                                context,
                                listen: false)
                            .continuousModeSteps);
                  }
                } catch (response) {
                  if (response is http.Response && response.statusCode == 404) {
                    Fluttertoast.showToast(
                        msg:
                            "404 error: Please ensure the correct baseURL for your agent in \nthe settings and that your agent adheres to the agent protocol.",
                        toastLength: Toast.LENGTH_LONG,
                        gravity: ToastGravity.TOP,
                        timeInSecForIosWeb: 5,
                        backgroundColor: Colors.red,
                        webPosition: "center",
                        webBgColor:
                            "linear-gradient(to right, #dc1c13, #dc1c13)",
                        textColor: Colors.white,
                        fontSize: 16.0);
                  } else if (response is http.Response &&
                      response.statusCode >= 500 &&
                      response.statusCode < 600) {
                    Fluttertoast.showToast(
                        msg: "500 error: Something went wrong",
                        toastLength: Toast.LENGTH_LONG,
                        gravity: ToastGravity.TOP,
                        timeInSecForIosWeb: 5,
                        backgroundColor: Colors.red,
                        webPosition: "center",
                        webBgColor:
                            "linear-gradient(to right, #dc1c13, #dc1c13)",
                        textColor: Colors.white,
                        fontSize: 16.0);
                  }
                }
              },
              onContinuousModePressed: () {
                widget.viewModel.isContinuousMode =
                    !widget.viewModel.isContinuousMode;
              },
              isContinuousMode: widget.viewModel.isContinuousMode,
              viewModel: widget.viewModel,
            ),
          ),
        ],
      ),
    );
  }
}
