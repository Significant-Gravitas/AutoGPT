// TODO: Remove the ability to have null values when benchmark implementation is complete
import 'dart:convert';

/// TaskInfo holds information related to a specific benchmark task.
///
/// The class encapsulates various attributes of a task, such as the path to the data file,
/// whether the task is a regression task, the categories it falls under, and specific task details
/// like the task description, expected answer, and so on.
class TaskInfo {
  /// The path to the data file associated with the task.
  /// This is typically a JSON file containing the dataset or resources needed for the task.
  final String dataPath;

  /// A boolean indicating whether the task is a regression task.
  final bool isRegression;

  /// A list of categories to which the task belongs.
  final List<String> category;

  /// The specific task that needs to be performed.
  final String task;

  /// The expected answer for the task.
  final String answer;

  /// A description providing details about the task.
  final String description;

  /// Constructs a new TaskInfo instance.
  ///
  /// [dataPath]: The path to the data file for the task.
  /// [isRegression]: A boolean indicating if the task is a regression task.
  /// [category]: A list of categories to which the task belongs.
  /// [task]: The specific task to be performed.
  /// [answer]: The expected answer for the task.
  /// [description]: A description of the task.
  TaskInfo({
    required this.dataPath,
    required this.isRegression,
    required this.category,
    required this.task,
    required this.answer,
    required this.description,
  });

  /// Creates a TaskInfo instance from a map.
  ///
  /// [json]: A map containing key-value pairs corresponding to TaskInfo fields.
  ///
  /// Returns a new TaskInfo populated with values from the map.
  factory TaskInfo.fromJson(Map<String, dynamic> json) => TaskInfo(
        dataPath: json['data_path'] ?? 'placeholder',
        isRegression: json['is_regression'] ?? false,
        category: List<String>.from(json['category']),
        task: json['task'] ?? 'placeholder',
        answer: json['answer'] ?? 'placeholder',
        description: json['description'] ?? 'placeholder',
      );

  /// Converts the TaskInfo instance to a map.
  ///
  /// Returns a map containing key-value pairs corresponding to TaskInfo fields.
  Map<String, dynamic> toJson() => {
        'data_path': dataPath,
        'is_regression': isRegression,
        'category': jsonEncode(category),
        'task': task,
        'answer': answer,
        'description': description,
      };
}
