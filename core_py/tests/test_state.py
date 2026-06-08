from __future__ import annotations

from pathlib import Path

import pytest

from psai.state_store import StateStore


@pytest.fixture()
def store(tmp_path: Path):
    db_path = tmp_path / "test_psai.sqlite3"
    s = StateStore(db_path=db_path)
    try:
        yield s
    finally:
        s.close()


def test_repo_create_and_two_commits_parent_chain(store: StateStore):
    status = store.repo_create(title="TestRepo", initial_payload={"hello": "world"})
    assert status.repo_id.startswith("R_")
    assert status.head.startswith("N_")

    root_node = store.get_node(status.repo_id, status.head)
    assert root_node.parent_id is None
    assert root_node.payload == {"hello": "world"}
    assert root_node.message == "root"

    c1 = store.commit(repo_id=status.repo_id, parent_id=root_node.node_id, message="commit1", payload={"v": 1})
    assert c1.parent_id == root_node.node_id

    c2 = store.commit(repo_id=status.repo_id, parent_id=c1.node_id, message="commit2", payload={"v": 2})
    assert c2.parent_id == c1.node_id

    status2 = store.repo_status(status.repo_id)
    assert status2.head == c2.node_id

    nodes = store.log(repo_id=status.repo_id, limit=10)
    assert [n.node_id for n in nodes[:3]] == [c2.node_id, c1.node_id, root_node.node_id]
