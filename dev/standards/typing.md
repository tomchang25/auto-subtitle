# Type Annotation Standards

Since Python 3.9+, built-in container types support generic syntax directly. Do not import the uppercase aliases from `typing`.

## Rules

Use lowercase built-in generics and PEP 604 union syntax:

| Use             | Do not use          |
|-----------------|---------------------|
| `list[X]`       | `typing.List[X]`    |
| `dict[K, V]`    | `typing.Dict[K, V]` |
| `tuple[X, ...]` | `typing.Tuple[X, ...]` |
| `set[X]`        | `typing.Set[X]`     |
| `X \| None`     | `typing.Optional[X]`|
| `X \| Y`        | `typing.Union[X, Y]`|

## Still imported from typing

The following have no built-in equivalent and should continue to be imported:

- `Callable` (or `collections.abc.Callable`)
- `Protocol`
- `TypedDict` (or `typing_extensions.TypedDict`)
- `TYPE_CHECKING`
- `Literal`

## Examples

```python
# Good
def translate(chunks: list[SubtitleChunk]) -> list[TranslatedChunk]: ...
def find(name: str | None = None) -> Path | None: ...

# Bad
from typing import List, Optional
def translate(chunks: List[SubtitleChunk]) -> List[TranslatedChunk]: ...
def find(name: Optional[str] = None) -> Optional[Path]: ...
```
