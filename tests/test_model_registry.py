from __future__ import annotations

from app import model_registry


class TestModelRegistry:
    def test_note_used_then_listed(self):
        model_registry.note_used("ollama", "http://localhost:11434/v1", "llama3")
        assert model_registry.all_used() == [("ollama", "http://localhost:11434/v1", "llama3")]

    def test_cloud_provider_is_never_tracked(self):
        model_registry.note_used("cloud", "https://api.openai.com/v1", "gpt-4o-mini")
        assert model_registry.all_used() == []

    def test_empty_model_name_is_never_tracked(self):
        model_registry.note_used("ollama", "http://localhost:11434/v1", "  ")
        assert model_registry.all_used() == []

    def test_trailing_slash_on_base_url_is_normalized(self):
        model_registry.note_used("lmstudio", "http://localhost:1234/v1/", "moondream")
        assert model_registry.all_used() == [("lmstudio", "http://localhost:1234/v1", "moondream")]

    def test_duplicate_use_is_not_duplicated(self):
        model_registry.note_used("ollama", "http://localhost:11434/v1", "llama3")
        model_registry.note_used("ollama", "http://localhost:11434/v1", "llama3")
        assert model_registry.all_used() == [("ollama", "http://localhost:11434/v1", "llama3")]

    def test_different_models_are_tracked_separately(self):
        model_registry.note_used("ollama", "http://localhost:11434/v1", "llama3")
        model_registry.note_used("lmstudio", "http://localhost:1234/v1", "moondream")
        assert model_registry.all_used() == [
            ("lmstudio", "http://localhost:1234/v1", "moondream"),
            ("ollama", "http://localhost:11434/v1", "llama3"),
        ]

    def test_forget_removes_one_entry(self):
        model_registry.note_used("ollama", "http://localhost:11434/v1", "llama3")
        model_registry.note_used("lmstudio", "http://localhost:1234/v1", "moondream")
        model_registry.forget("ollama", "http://localhost:11434/v1", "llama3")
        assert model_registry.all_used() == [("lmstudio", "http://localhost:1234/v1", "moondream")]

    def test_forget_unknown_entry_is_a_no_op(self):
        model_registry.forget("ollama", "http://localhost:11434/v1", "never-used")
        assert model_registry.all_used() == []

    def test_clear_empties_everything(self):
        model_registry.note_used("ollama", "http://localhost:11434/v1", "llama3")
        model_registry.clear()
        assert model_registry.all_used() == []
