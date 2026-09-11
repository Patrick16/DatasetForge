from __future__ import annotations

from app.dedup import find_duplicate_groups, hash_file


class TestHashFile:
    def test_identical_content_hashes_the_same(self, tmp_path):
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        a.write_bytes(b"same content")
        b.write_bytes(b"same content")
        assert hash_file(a) == hash_file(b)

    def test_different_content_hashes_differently(self, tmp_path):
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        a.write_bytes(b"content one")
        b.write_bytes(b"content two")
        assert hash_file(a) != hash_file(b)

    def test_large_file_is_hashed_in_chunks_correctly(self, tmp_path):
        # bigger than _CHUNK_SIZE (1 MiB) so the chunked read loop actually
        # runs more than once -- content is still just repeated bytes, but
        # this exercises the loop instead of a single read() call.
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        content = b"x" * (2 * 1024 * 1024 + 137)
        a.write_bytes(content)
        b.write_bytes(content)
        assert hash_file(a) == hash_file(b)


class TestFindDuplicateGroups:
    def test_no_duplicates_returns_empty_list(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"one")
        (tmp_path / "b.jpg").write_bytes(b"two")
        assert find_duplicate_groups(tmp_path) == []

    def test_finds_a_simple_duplicate_pair(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"same")
        (tmp_path / "b.jpg").write_bytes(b"same")
        groups = find_duplicate_groups(tmp_path)
        assert len(groups) == 1
        assert {p.name for p in groups[0]} == {"a.jpg", "b.jpg"}

    def test_keeper_is_alphabetically_first_in_the_group(self, tmp_path):
        (tmp_path / "z.jpg").write_bytes(b"same")
        (tmp_path / "a.jpg").write_bytes(b"same")
        (tmp_path / "m.jpg").write_bytes(b"same")
        groups = find_duplicate_groups(tmp_path)
        keeper, *dupes = groups[0]
        assert keeper.name == "a.jpg"
        assert {p.name for p in dupes} == {"m.jpg", "z.jpg"}

    def test_groups_of_three_or_more_are_handled(self, tmp_path):
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            (tmp_path / name).write_bytes(b"same")
        groups = find_duplicate_groups(tmp_path)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_multiple_independent_duplicate_groups(self, tmp_path):
        (tmp_path / "a1.jpg").write_bytes(b"group a")
        (tmp_path / "a2.jpg").write_bytes(b"group a")
        (tmp_path / "b1.jpg").write_bytes(b"group b")
        (tmp_path / "b2.jpg").write_bytes(b"group b")
        (tmp_path / "unique.jpg").write_bytes(b"only one")

        groups = find_duplicate_groups(tmp_path)
        assert len(groups) == 2
        names = [{p.name for p in g} for g in groups]
        assert {"a1.jpg", "a2.jpg"} in names
        assert {"b1.jpg", "b2.jpg"} in names

    def test_non_image_files_are_ignored_even_if_content_matches(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"same")
        (tmp_path / "a.txt").write_bytes(b"same")  # caption file, not an image
        (tmp_path / "readme.md").write_bytes(b"same")
        assert find_duplicate_groups(tmp_path) == []

    def test_non_recursive_by_default_ignores_subfolders(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"same")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.jpg").write_bytes(b"same")

        assert find_duplicate_groups(tmp_path, recursive=False) == []

        groups = find_duplicate_groups(tmp_path, recursive=True)
        assert len(groups) == 1
        assert {p.name for p in groups[0]} == {"a.jpg", "b.jpg"}

    def test_case_insensitive_extension_matching(self, tmp_path):
        (tmp_path / "a.JPG").write_bytes(b"same")
        (tmp_path / "b.jpg").write_bytes(b"same")
        groups = find_duplicate_groups(tmp_path)
        assert len(groups) == 1

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert find_duplicate_groups(tmp_path) == []
