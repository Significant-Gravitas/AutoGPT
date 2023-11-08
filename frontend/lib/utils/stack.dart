class Stack<T> {
  final List<T> _list = [];

  void push(T element) {
    _list.add(element);
  }

  T pop() {
    var element = _list.last;
    _list.removeLast();
    return element;
  }

  T peek() {
    return _list.last;
  }

  bool get isEmpty => _list.isEmpty;
  bool get isNotEmpty => _list.isNotEmpty;
}
