def to_numbered_list(items: list[str], **template_args) -> str:
    return "\n".join(
        f"{i+1}. {item.format(**template_args)}" for i, item in enumerate(items)
    )
