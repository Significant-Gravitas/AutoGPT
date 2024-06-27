from typing import Dict
from dataclasses import dataclass, field


@dataclass
class Code:
    code_files: Dict[str, str] = field(default_factory=dict)

    def __getitem__(self, item: str) -> str:
        return self.code_files[item]

    def __setitem__(self, key: str, value: str) -> None:
        self.code_files[key] = value

    def items(self):
        return self.code_files.items()
