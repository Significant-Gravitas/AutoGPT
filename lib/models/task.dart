/// Represents a task or topic the user wants to discuss with the agent.
class Task {
  final int id;
  String _title;

  Task({required this.id, required String title})
      : assert(title.isNotEmpty, 'Title cannot be empty'),
        _title = title;

  String get title => _title;

  set title(String newTitle) {
    if (newTitle.isNotEmpty) {
      _title = newTitle;
    } else {
      throw ArgumentError('Title cannot be empty.');
    }
  }

  // Convert a Map (usually from JSON) to a Task object
  factory Task.fromMap(Map<String, dynamic> map) {
    return Task(
      id: map['id'],
      title: map['title'],
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Task &&
          runtimeType == other.runtimeType &&
          id == other.id &&
          title == other.title;

  @override
  int get hashCode => id.hashCode ^ title.hashCode;

  @override
  String toString() => 'Task(id: $id, title: $title)';
}
