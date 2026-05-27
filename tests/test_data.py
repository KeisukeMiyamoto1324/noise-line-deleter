from __future__ import annotations

from corpus_refiner_jp.data import build_line_labels, create_document_windows


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            return [101, 102]
        return [ord(character) for character in text]


def test_build_line_labels_uses_keep_positive_label() -> None:
    lines, labels = build_line_labels("a\nb\nc", [1, 3])

    assert lines == ["a", "b", "c"]
    assert labels == [1, 0, 1]


def test_create_document_windows_keeps_line_labels() -> None:
    tokenizer = FakeTokenizer()

    windows = create_document_windows(
        document_id="doc-1",
        text="a\nb\nc",
        lines_to_keep=[2],
        tokenizer=tokenizer,
        line_token_id=999,
        max_length=6,
        max_lines_per_window=2,
        line_overlap=0,
    )

    assert [window.labels for window in windows] == [[0, 1], [0]]
