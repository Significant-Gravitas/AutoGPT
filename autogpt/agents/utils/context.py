from autogpt.models.context_item import ContextItem


class AgentContext:
    items: list[ContextItem]

    def __init__(self, items: list[ContextItem] = []):
        self.items = items

    def __bool__(self) -> bool:
        return len(self.items) > 0

    def add(self, item: ContextItem) -> None:
        self.items.append(item)

    def close(self, index: int) -> None:
        self.items.pop(index - 1)

    def clear(self) -> None:
        self.items.clear()

    def format_numbered(self) -> str:
        return "\n\n".join([f"{i}. {c}" for i, c in enumerate(self.items, 1)])
