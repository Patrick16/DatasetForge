from __future__ import annotations

from app.browse import list_folder_images


class TestListFolderImages:
    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert list_folder_images(tmp_path) == []

    def test_lists_images_without_captions(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "b.png").write_bytes(b"fake")
        items = list_folder_images(tmp_path)
        assert [i["name"] for i in items] == ["a.jpg", "b.png"]
        assert all(i["caption"] is None for i in items)

    def test_pairs_a_caption_from_the_sidecar_txt_file(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "a.txt").write_text("a fox in a field", encoding="utf-8")
        items = list_folder_images(tmp_path)
        assert items[0]["caption"] == "a fox in a field"

    def test_blank_caption_file_is_treated_as_no_caption(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "a.txt").write_text("   \n", encoding="utf-8")
        items = list_folder_images(tmp_path)
        assert items[0]["caption"] is None

    def test_non_image_files_are_ignored(self, tmp_path):
        (tmp_path / "readme.md").write_text("hi", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
        assert list_folder_images(tmp_path) == []

    def test_sorted_alphabetically_by_name(self, tmp_path):
        (tmp_path / "z.jpg").write_bytes(b"fake")
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "m.jpg").write_bytes(b"fake")
        items = list_folder_images(tmp_path)
        assert [i["name"] for i in items] == ["a.jpg", "m.jpg", "z.jpg"]

    def test_recursive_by_default_includes_subfolders(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        sub = tmp_path / "query_1"
        sub.mkdir()
        (sub / "b.jpg").write_bytes(b"fake")
        items = list_folder_images(tmp_path)
        assert {i["name"] for i in items} == {"a.jpg", "b.jpg"}

    def test_non_recursive_ignores_subfolders(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"fake")
        sub = tmp_path / "query_1"
        sub.mkdir()
        (sub / "b.jpg").write_bytes(b"fake")
        items = list_folder_images(tmp_path, recursive=False)
        assert [i["name"] for i in items] == ["a.jpg"]

    def test_path_is_an_absolute_string_to_the_file(self, tmp_path):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"fake")
        items = list_folder_images(tmp_path)
        assert items[0]["path"] == str(img)
