from __future__ import annotations

from fineweb2_line_deleter.data import build_line_labels, create_document_windows


class DummyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        token_ids = [ord(character) for character in text]
        if add_special_tokens:
            return [101] + token_ids + [102]
        return token_ids

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return [101] + token_ids + [102]

    def prepare_for_model(
        self,
        token_ids: list[int],
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_attention_mask: bool,
    ) -> dict[str, list[int]]:
        input_ids = self.build_inputs_with_special_tokens(token_ids)
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def test_build_line_labels_keeps_empty_lines() -> None:
    lines, labels = build_line_labels("a\n\nb", [2])

    assert lines == ["a", "", "b"]
    assert labels == [0, 1, 0]


def test_create_document_windows_keeps_line_token_label_count() -> None:
    tokenizer = DummyTokenizer()

    windows = create_document_windows(
        document_id="doc-1",
        text="aa\nbb\ncc",
        lines_to_delete=[1, 3],
        tokenizer=tokenizer,
        line_token_id=999,
        max_length=8,
        max_lines_per_window=2,
        line_overlap=1,
    )

    assert [window.labels for window in windows] == [[1, 0], [0, 1]]
    assert [window.input_ids.count(999) for window in windows] == [2, 2]
