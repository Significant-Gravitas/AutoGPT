import 'package:auto_gpt_flutter_client/models/step.dart';
import 'package:auto_gpt_flutter_client/models/step_request_body.dart';
import 'package:flutter/foundation.dart';
import 'package:auto_gpt_flutter_client/services/chat_service.dart';
import 'package:auto_gpt_flutter_client/models/chat.dart';
import 'package:auto_gpt_flutter_client/models/message_type.dart';

class ChatViewModel with ChangeNotifier {
  final ChatService _chatService;
  List<Chat> _chats = [];
  String? _currentTaskId;

  ChatViewModel(this._chatService);

  /// Returns the current list of chats.
  List<Chat> get chats => _chats;

  String? get currentTaskId => _currentTaskId;

  void setCurrentTaskId(String taskId) {
    if (_currentTaskId != taskId) {
      _currentTaskId = taskId;
      fetchChatsForTask();
    }
  }

  void clearCurrentTaskAndChats() {
    _currentTaskId = null;
    _chats.clear();
    notifyListeners(); // Notify listeners to rebuild UI
  }

  /// Fetches chats from the data source for a specific task.
  void fetchChatsForTask() async {
    if (_currentTaskId == null) {
      print("Error: Task ID is not set.");
      return;
    }
    try {
      // Fetch task steps from the data source
      final Map<String, dynamic> stepsResponse =
          await _chatService.listTaskSteps(_currentTaskId!);

      // Extract steps from the response
      final List<dynamic> stepsJsonList = stepsResponse['steps'] ?? [];

      // Convert each map into a Step object
      List<Step> steps =
          stepsJsonList.map((stepMap) => Step.fromMap(stepMap)).toList();

      // Initialize an empty list to store Chat objects
      List<Chat> chats = [];

      // Generate current timestamp
      DateTime currentTimestamp = DateTime.now();

      for (Step step in steps) {
        // Create a Chat object for 'input' if it exists and is not empty
        if (step.input.isNotEmpty) {
          chats.add(Chat(
            id: step.stepId,
            taskId: step.taskId,
            message: step.input,
            timestamp: currentTimestamp,
            messageType: MessageType.user,
          ));
        }

        // Create a Chat object for 'output'
        chats.add(Chat(
          id: step.stepId,
          taskId: step.taskId,
          message: step.output,
          timestamp: currentTimestamp,
          messageType: MessageType.agent,
        ));
      }

      // Assign the chats list
      _chats = chats;

      // Notify listeners to rebuild UI
      notifyListeners();

      print(
          "Chats (and steps) fetched successfully for task ID: $_currentTaskId");
    } catch (error) {
      print("Error fetching chats: $error");
      // TODO: Handle additional error scenarios or log them as required
    }
  }

  /// Sends a chat message for a specific task.
  void sendChatMessage(String message) async {
    if (_currentTaskId == null) {
      print("Error: Task ID is not set.");
      return;
    }
    try {
      // Create the request body for executing the step
      StepRequestBody requestBody = StepRequestBody(input: message);

      // Execute the step and get the response
      Map<String, dynamic> executedStepResponse =
          await _chatService.executeStep(_currentTaskId!, requestBody);

      // Create a Chat object from the returned step
      Step executedStep = Step.fromMap(executedStepResponse);

      // Create a Chat object for the user message
      final userChat = Chat(
        id: executedStep.stepId,
        taskId: executedStep.taskId,
        message: executedStep.input,
        timestamp: DateTime.now(),
        messageType: MessageType.user,
      );

      // Create a Chat object for the agent message
      final agentChat = Chat(
        id: executedStep.stepId,
        taskId: executedStep.taskId,
        message: executedStep.output,
        timestamp: DateTime.now(),
        messageType: MessageType.agent,
      );

      // Add the user and agent chats to the list
      _chats.add(userChat);
      _chats.add(agentChat);

      // Notify UI of the new chats
      notifyListeners();

      print("Chats added for task ID: $_currentTaskId");
    } catch (error) {
      // TODO: Bubble up errors to UI
      print("Error sending chat: $error");
      // TODO: Handle additional error scenarios or log them as required
    }
  }
}
