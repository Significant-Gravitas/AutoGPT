# Singleton Metaclass

This Python code defines a `Singleton` metaclass that allows ensuring there's only one instance of a class.

## Usage

The `Singleton` metaclass should be used as the first item in the tuple of base classes when defining a new class.

```python
class MyClass(metaclass=Singleton):
    # class definition
```

Alternatively, `AbstractSingleton` can be inherited to define an abstract class that is also a singleton.

```python
class MyAbstractClass(AbstractSingleton):
    # abstract class definition
```

## How it works

The `Singleton` metaclass overrides the `__call__` method to keep track of the instances that have been created for each class. If a new instance of the class is requested and there is no instance already, a new one is created and stored. If there's already an instance of the class, the same instance is returned.

The `AbstractSingleton` abstract class is an example of how to use the `Singleton` metaclass to create an abstract base class that ensures there's only one implementation of it. The class inherits from `ABC`, which makes it an abstract class and sets the metaclass to `Singleton`. Then it provides an empty implementation using the `pass` statement. Any class that inherits from `AbstractSingleton` and provides an implementation will be a singleton.