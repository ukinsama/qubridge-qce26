"""
量子ビット選択戦略

3つの選択戦略を提供する:
- qubridge: エラーレート閾値フィルタ → 3量子ビット連続パスから最適選択
- random: 全量子ビットから無作為に3つ選択（非隣接含む → SWAP挿入あり）
- default: None を返す（transpilerに任せる）
"""
import random as random_module
from typing import List, Optional, Tuple

from .static_backend import load_static_backend_data


def build_adjacency(coupling_map: list) -> dict:
    """カップリングマップから隣接リスト（無向グラフ）を構築する"""
    adj: dict = {}
    for edge in coupling_map:
        a, b = edge[0], edge[1]
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def get_error_map(errors: list) -> dict:
    """エラーリストから (qubit_from, qubit_to) → error_rate の辞書を構築する"""
    error_map = {}
    for err in errors:
        key = (err["qubit_from"], err["qubit_to"])
        error_map[key] = err["error_rate"]
    return error_map


def find_3qubit_paths(adj: dict) -> List[Tuple[int, int, int]]:
    """
    隣接リストから全ての3量子ビット連続パス (a-b-c) を列挙する。

    テレポーテーション回路は3量子ビットが線形に接続されている必要があるため、
    a-b と b-c の両方が接続されているパスを探す。
    """
    paths = []
    for b in adj:
        neighbors = sorted(adj[b])
        for i, a in enumerate(neighbors):
            for c in neighbors[i + 1:]:
                # a-b-c のパスが存在
                paths.append((a, b, c))
    return paths


def path_avg_error(path: Tuple[int, int, int], error_map: dict) -> Optional[float]:
    """
    3量子ビットパスの平均2量子ゲートエラーを計算する。

    パス (a, b, c) のエラー = (error(a,b) + error(b,c)) / 2
    エラーデータが不足している場合はNoneを返す。
    """
    a, b, c = path
    # 双方向のエラーを探索（カップリングマップの向きに依存しないように）
    e_ab = error_map.get((a, b)) or error_map.get((b, a))
    e_bc = error_map.get((b, c)) or error_map.get((c, b))
    if e_ab is None or e_bc is None:
        return None
    return (e_ab + e_bc) / 2


def get_single_qubit_error_map(gate_properties: dict, basis_gates: list) -> dict:
    """
    gate_propertiesから各量子ビットの1量子ゲートエラーの最大値を取得する。

    1量子ゲート（sx, x, id）の中で最大のエラーレートを各量子ビットに割り当てる。
    これにより、最も品質の悪いゲートがスコアリングに反映される。

    Args:
        gate_properties: バックエンドのゲートプロパティ辞書
        basis_gates: バックエンドのbasis_gatesリスト

    Returns:
        {qubit_index: max_1q_error} の辞書
    """
    single_q_gates = [g for g in ["sx", "x", "id"] if g in basis_gates]
    error_map: dict = {}
    for gate_name in single_q_gates:
        if gate_name not in gate_properties:
            continue
        for qargs_key, props in gate_properties[gate_name].items():
            # "(53,)" → 53
            qubit = int(qargs_key.strip("() ").rstrip(",").strip())
            err = props.get("error")
            if err is not None:
                error_map[qubit] = max(error_map.get(qubit, 0.0), err)
    return error_map


def get_readout_error_map(qubit_properties: list) -> dict:
    """
    qubit_propertiesから各量子ビットのreadoutエラーを取得する。

    readoutエラーはゲートエラーの100〜300倍大きく、
    フィデリティへの影響が支配的なため、パス選択で考慮する必要がある。

    Args:
        qubit_properties: バックエンドのqubit_propertiesリスト

    Returns:
        {qubit_index: readout_error} の辞書
    """
    error_map = {}
    for i, qp in enumerate(qubit_properties):
        if qp is None:
            continue
        err = qp.get("readout_error")
        if err is not None:
            error_map[i] = err
    return error_map


def path_combined_error(
    path: Tuple[int, int, int],
    error_map_2q: dict,
    error_map_1q: dict,
    error_map_readout: Optional[dict] = None,
) -> Optional[float]:
    """
    3量子ビットパスの複合エラースコアを計算する。

    スコア = 2Qゲートエラー平均 + 1Qゲートエラー平均 + readoutエラー平均
    全エラー源を考慮することで、ノイズモデルのフィデリティへの影響を
    スコアリングに正しく反映する。

    Args:
        path: (a, b, c) の3量子ビットパス
        error_map_2q: (qubit_from, qubit_to) → 2Qエラーの辞書
        error_map_1q: qubit → 1Qエラーの辞書
        error_map_readout: qubit → readoutエラーの辞書（Noneの場合は無視）

    Returns:
        複合エラースコア。2Qエラーが不足している場合はNone。
    """
    avg_2q = path_avg_error(path, error_map_2q)
    if avg_2q is None:
        return None
    a, b, c = path
    avg_1q = sum(error_map_1q.get(q, 0.0) for q in (a, b, c)) / 3
    score = avg_2q + avg_1q
    if error_map_readout is not None:
        avg_ro = sum(error_map_readout.get(q, 0.0) for q in (a, b, c)) / 3
        score += avg_ro
    return score


def _filter_top(
    paths: List[Tuple[int, int, int]],
    score_fn,
    keep_ratio: float = 0.5,
) -> List[Tuple[int, int, int]]:
    """
    パスリストをスコア関数でソートし、上位keep_ratio分（最低1つ）を残す。

    各階層フィルタリングステージで使用する共通ヘルパー。

    Args:
        paths: フィルタ対象のパスリスト
        score_fn: path → float のスコア関数（低いほど良い）
        keep_ratio: 残す割合（0.5=上位50%, 0.8=上位80%で下位20%を除外）

    Returns:
        上位のパスリスト
    """
    scored = [(score_fn(p), p) for p in paths]
    scored.sort(key=lambda x: x[0])
    top_n = max(1, int(len(scored) * keep_ratio))
    return [p for _, p in scored[:top_n]]


def select_qubits_qubridge(
    backend_name: str,
    threshold: float = 0.01,
    seed: Optional[int] = None,
) -> List[int]:
    """
    QuBridge戦略: 階層的エラーフィルタリングで最適な3量子ビットパスを選択する。

    手順:
    1. 2Qゲートエラー閾値でエッジをフィルタし、3量子ビットパスを列挙
    2. Stage 1: 2Qゲートエラー平均で上位30%に絞る（主要フィルタ）
    3. Stage 2: 1Qゲートエラーが壊滅的なパスを除外（下位20%カット）
    4. Stage 3: readoutエラーが壊滅的なパスを除外（下位20%カット）
    5. 最終候補からseedで選択

    2Qエラーを主軸に厳選し、1Q/readoutは外れ値除外に留める。
    これにより2Qエラー閾値（strict/loose）の効果が結果に正しく反映される。

    Args:
        backend_name: バックエンド名
        threshold: 2Qゲートエラー閾値（これ以下のエッジのみ使用）
        seed: ランダムseed（再現性のため）

    Returns:
        3要素のリスト [qubit_a, qubit_b, qubit_c]

    Raises:
        ValueError: 閾値を満たすパスが見つからない場合
    """
    data = load_static_backend_data(backend_name)
    errors = data.get("errors", [])
    coupling_map = data.get("coupling_map", [])
    gate_properties = data.get("gate_properties", {})
    basis_gates = data.get("basis_gates", [])
    qubit_properties = data.get("qubit_properties", [])

    # エラーマップ構築
    error_map_2q = get_error_map(errors)
    error_map_1q = get_single_qubit_error_map(gate_properties, basis_gates)
    error_map_ro = get_readout_error_map(qubit_properties)

    # === Stage 0: 2Qエラー閾値でエッジフィルタ ===
    filtered_edges = []
    for edge in coupling_map:
        a, b = edge[0], edge[1]
        e = error_map_2q.get((a, b)) or error_map_2q.get((b, a))
        if e is not None and e <= threshold:
            filtered_edges.append(edge)

    if not filtered_edges:
        raise ValueError(
            f"閾値 {threshold} を満たすエッジがありません（backend={backend_name}）"
        )

    # 3量子ビットパスを列挙
    adj = build_adjacency(filtered_edges)
    paths = find_3qubit_paths(adj)

    if not paths:
        raise ValueError(
            f"閾値 {threshold} で3量子ビット連続パスが見つかりません（backend={backend_name}）"
        )

    # 2Qエラーデータが揃っているパスのみ残す
    paths = [p for p in paths if path_avg_error(p, error_map_2q) is not None]

    if not paths:
        raise ValueError("エラー情報が不足しています")

    # threshold >= 1.0 の場合（loose）: 階層フィルタなし、全パスから選択
    # threshold < 1.0 の場合（strict）: 階層フィルタリングで厳選
    if threshold < 1.0:
        # === Stage 1: 2Qゲートエラー平均で上位30%（主要フィルタ） ===
        paths = _filter_top(
            paths, lambda p: path_avg_error(p, error_map_2q), keep_ratio=0.3
        )

        # === Stage 2: 1Qゲートエラー — 壊滅的なもののみ除外（下位20%カット） ===
        paths = _filter_top(
            paths,
            lambda p: sum(error_map_1q.get(q, 0.0) for q in p) / 3,
            keep_ratio=0.8,
        )

        # === Stage 3: readoutエラー — 壊滅的なもののみ除外（下位20%カット） ===
        paths = _filter_top(
            paths,
            lambda p: sum(error_map_ro.get(q, 0.0) for q in p) / 3,
            keep_ratio=0.8,
        )

    # 最終候補からseedで選択
    if seed is not None:
        rng = random_module.Random(seed)
        selected = rng.choice(paths)
    else:
        selected = paths[0]

    return list(selected)


def select_qubits_random(
    backend_name: str,
    seed: Optional[int] = None,
) -> List[int]:
    """
    ランダム戦略: 全量子ビットから無作為に3つ選択する。

    接続性を考慮しないため、非隣接量子ビットが選ばれた場合は
    トランスパイラがSWAPゲートを挿入してルーティングを行う。
    これにより qubridge 戦略（隣接パス選択）との公平な比較が可能になる。

    Args:
        backend_name: バックエンド名
        seed: ランダムseed（再現性のため）

    Returns:
        3要素のリスト [qubit_a, qubit_b, qubit_c]
    """
    data = load_static_backend_data(backend_name)
    n_qubits = data.get("n_qubits", 0)

    if n_qubits < 3:
        raise ValueError(f"量子ビット数が不足しています: {n_qubits}（backend={backend_name}）")

    # 全量子ビットから重複なしで3つ選択
    rng = random_module.Random(seed)
    selected = rng.sample(range(n_qubits), 3)
    return selected


def select_qubits_default() -> None:
    """
    デフォルト戦略: Noneを返す（transpilerに量子ビット選択を任せる）。

    initial_layout=None で transpile() を呼ぶと、
    transpiler が自動的に最適な量子ビットを選択する。
    """
    return None


def find_all_perfect_layouts(
    backend_name: str,
    circuit_edges: list,
    call_limit: int = 100000,
) -> list:
    """
    VF2アルゴリズムで回路トポロジーに一致する全サブグラフを列挙する。

    バックエンドのカップリングマップから、回路の接続パターンに
    完全一致する物理量子ビットの配置（レイアウト）を全て見つける。
    SWAPゲートが不要な「パーフェクトレイアウト」のみを返す。

    Args:
        backend_name: バックエンド名
        circuit_edges: 回路の2Qゲート接続 [(0,1), (0,2), ...]
        call_limit: VF2の探索制限（大きいほど網羅的だが遅くなる）

    Returns:
        list of list[int]: 各マッチングの物理量子ビットリスト
            layout[i] = 論理量子ビットiに対応する物理量子ビット番号
    """
    import rustworkx as rx
    from rustworkx import vf2_mapping

    data = load_static_backend_data(backend_name)
    coupling_map = data.get("coupling_map", [])

    # バックエンドのカップリングマップをrustworkxの無向グラフに変換
    n_qubits = data.get("n_qubits", 0)
    backend_graph = rx.PyGraph()
    backend_graph.add_nodes_from(range(n_qubits))
    # 重複エッジを防ぐためsetで管理
    edge_set = set()
    for edge in coupling_map:
        a, b = edge[0], edge[1]
        key = (min(a, b), max(a, b))
        if key not in edge_set:
            edge_set.add(key)
            backend_graph.add_edge(a, b, None)

    # 回路トポロジーをrustworkxの無向グラフに変換
    n_logical = max(max(e) for e in circuit_edges) + 1
    circuit_graph = rx.PyGraph()
    circuit_graph.add_nodes_from(range(n_logical))
    for a, b in circuit_edges:
        circuit_graph.add_edge(a, b, None)

    # VF2でサブグラフ同型を列挙
    mappings = vf2_mapping(
        backend_graph,
        circuit_graph,
        subgraph=True,
        call_limit=call_limit,
    )

    # mapping: {backend_node: circuit_node} → layout[circuit_node] = backend_node に変換
    layouts = []
    for mapping in mappings:
        layout = [0] * n_logical
        for backend_node, circuit_node in mapping.items():
            layout[circuit_node] = backend_node
        layouts.append(layout)

    return layouts


def score_layout(
    layout: list,
    backend_name: str,
    circuit_edges: list | None = None,
) -> float:
    """
    レイアウトのノイズスコアを計算する（低いほど良い）。

    スコア = 2Qゲートエラー平均 + 1Qゲートエラー平均 + readoutエラー平均
    circuit_edgesが指定された場合、回路が実際に使う2Qゲートのエラーのみを集計する。

    Args:
        layout: 物理量子ビットリスト（layout[i] = 論理量子ビットiの物理位置）
        backend_name: バックエンド名
        circuit_edges: 回路の2Qゲート接続（指定時はこのペアのみ集計）

    Returns:
        複合エラースコア（float, 低いほど良い）
    """
    data = load_static_backend_data(backend_name)
    errors = data.get("errors", [])
    gate_properties = data.get("gate_properties", {})
    basis_gates = data.get("basis_gates", [])
    qubit_properties = data.get("qubit_properties", [])

    # 各種エラーマップを構築
    error_map_2q = get_error_map(errors)
    error_map_1q = get_single_qubit_error_map(gate_properties, basis_gates)
    error_map_ro = get_readout_error_map(qubit_properties)

    n = len(layout)

    # 1Qエラー平均
    avg_1q = sum(error_map_1q.get(q, 0.0) for q in layout) / n

    # readoutエラー平均
    avg_ro = sum(error_map_ro.get(q, 0.0) for q in layout) / n

    # 2Qエラー: circuit_edgesに対応する物理ペアのみ集計
    two_q_errors = []
    if circuit_edges is not None:
        for src, dst in circuit_edges:
            phys_src, phys_dst = layout[src], layout[dst]
            err = error_map_2q.get((phys_src, phys_dst)) or error_map_2q.get((phys_dst, phys_src))
            if err is not None:
                two_q_errors.append(err)
    else:
        # circuit_edges未指定時: layout内の全隣接ペアのエラーを集計
        layout_set = set(layout)
        for (a, b), err in error_map_2q.items():
            if a in layout_set and b in layout_set:
                two_q_errors.append(err)
    avg_2q = sum(two_q_errors) / len(two_q_errors) if two_q_errors else 0.0

    return avg_2q + avg_1q + avg_ro


def select_qubits_qubridge_logical(
    backend_name: str,
    threshold: float = 0.01,
    seed: Optional[int] = None,
    circuit_edges: list | None = None,
) -> list:
    """
    QuBridge戦略（論理回路版）: VF2パーフェクトレイアウトをエラー閾値でフィルタし、
    ノイズスコアで最良の6量子ビットを選択する。

    手順:
    1. VF2でバックエンド上の全パーフェクトレイアウトを列挙
    2. 2Qゲートエラー閾値でレイアウトをフィルタ
    3. score_layout()でスコアリングし上位30%に絞る
    4. seedで最終選択

    Args:
        backend_name: バックエンド名
        threshold: 2Qゲートエラー閾値（回路エッジ上の全ペアがこれ以下）
        seed: ランダムseed（再現性のため）
        circuit_edges: 回路の接続パターン（デフォルト: 論理テレポーテーション）

    Returns:
        6要素のリスト [物理量子ビット...]

    Raises:
        ValueError: パーフェクトレイアウトまたは閾値を満たすレイアウトが見つからない場合
    """
    if circuit_edges is None:
        # 論理テレポーテーション回路の VF2 接続パターン (5 edges, heavy-hex 自然形):
        #   (0,1) Alice 内 encode/decode
        #   (1,2) Alice→Mediator (chain として隣接性を要求)
        #   (2,3) Mediator 内 Bell pair  ← 物理隣接で確保 (Torino で 100% 保証)
        #   (3,4) Mediator→Bob
        #   (3,5) Mediator→Bob (b1 は m1 から分岐)
        # このパターンは services/mode1_strategies.LOGICAL_CIRCUIT_EDGES と同じ。
        # heavy-hex の linear chain + branch 構造に自然マッチし、(2,3) を物理隣接
        # 確保することで RB error 落ちを防ぐ (Mode3 honest routing と整合)。
        circuit_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)]

    # 全パーフェクトレイアウトを取得
    all_layouts = find_all_perfect_layouts(backend_name, circuit_edges)
    if not all_layouts:
        raise ValueError(f"パーフェクトレイアウトが見つかりません（backend={backend_name}）")

    # 閾値フィルタ: レイアウト内の全2Qペアが閾値以下
    if threshold < 1.0:
        data = load_static_backend_data(backend_name)
        error_map_2q = get_error_map(data.get("errors", []))

        filtered = []
        for layout in all_layouts:
            all_under = True
            for src, dst in circuit_edges:
                phys_src, phys_dst = layout[src], layout[dst]
                # 双方向のエラーを探索
                err = error_map_2q.get((phys_src, phys_dst)) or error_map_2q.get((phys_dst, phys_src))
                if err is None or err > threshold:
                    all_under = False
                    break
            if all_under:
                filtered.append(layout)

        if not filtered:
            raise ValueError(
                f"閾値 {threshold} を満たすレイアウトがありません（backend={backend_name}）"
            )
        all_layouts = filtered

    # スコアでソートし上位30%に絞る（回路エッジのみで2Qエラーを評価）
    scored = [(score_layout(l, backend_name, circuit_edges), l) for l in all_layouts]
    scored.sort(key=lambda x: x[0])
    top_n = max(1, int(len(scored) * 0.3))
    top_layouts = [l for _, l in scored[:top_n]]

    # seedで選択（再現性のため）
    if seed is not None:
        rng = random_module.Random(seed)
        selected = rng.choice(top_layouts)
    else:
        selected = top_layouts[0]

    return selected
