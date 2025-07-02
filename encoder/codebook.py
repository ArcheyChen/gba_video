import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from apricot import FacilityLocationSelection

# 码本生成相关函数

def generate_multi_level_codebooks_for_gop(
    i_frame_blocks_8x4: np.ndarray, 
    p_frame_blocks_8x4_list: list,
    i_frame_weight: int = 3,
    coverage_radius_8x4: float = 80.0,
    coverage_radius_4x4: float = 50.0,
    codebook_size_8x4: int = 64,
    codebook_size_4x4: int = 128,
    codebook_size_4x2: int = 256
) -> tuple:
    print(f"为GOP生成三级码表...")
    print(f"I帧8x4块数: {len(i_frame_blocks_8x4)}")
    print(f"P帧变化8x4块总数: {sum(len(blocks) for _, blocks in p_frame_blocks_8x4_list)}")
    training_blocks_8x4 = []
    for _ in range(i_frame_weight):
        training_blocks_8x4.append(i_frame_blocks_8x4)
    for frame_idx, changed_blocks_8x4 in p_frame_blocks_8x4_list:
        if len(changed_blocks_8x4) > 0:
            training_blocks_8x4.append(changed_blocks_8x4)
    if not training_blocks_8x4:
        raise ValueError("没有足够的8x4块用于生成码表")
    all_training_blocks_8x4 = np.vstack(training_blocks_8x4)
    print(f"总8x4训练块数: {len(all_training_blocks_8x4)} (I帧权重x{i_frame_weight})")
    print("生成8x4码表（最大覆盖方法）...")
    codebook_8x4 = generate_codebook_8x4_max_coverage(
        all_training_blocks_8x4, 
        radius=coverage_radius_8x4, 
        n_neighbors=codebook_size_8x4
    )
    print("寻找8x4码表无法覆盖的块...")
    distances_8x4 = pairwise_distances(
        all_training_blocks_8x4.astype(np.float32), 
        codebook_8x4.astype(np.float32), 
        metric="euclidean", 
        n_jobs=1
    )
    min_distances_8x4 = distances_8x4.min(axis=1)
    uncovered_8x4_mask = min_distances_8x4 > coverage_radius_8x4
    uncovered_blocks_8x4 = all_training_blocks_8x4[uncovered_8x4_mask]
    print(f"8x4无法覆盖的块数: {len(uncovered_blocks_8x4)} / {len(all_training_blocks_8x4)}")
    uncovered_blocks_4x4 = []
    for block_8x4 in uncovered_blocks_8x4:
        y_8x4 = block_8x4[:32].reshape(4, 8)
        left_y_4x4 = y_8x4[:, :4].flatten()
        right_y_4x4 = y_8x4[:, 4:].flatten()
        cb_8x4 = block_8x4[32:64].reshape(4, 8)
        left_cb_4x4 = cb_8x4[:, :4].flatten()
        right_cb_4x4 = cb_8x4[:, 4:].flatten()
        cr_8x4 = block_8x4[64:96].reshape(4, 8)
        left_cr_4x4 = cr_8x4[:, :4].flatten()
        right_cr_4x4 = cr_8x4[:, 4:].flatten()
        left_4x4 = np.concatenate([left_y_4x4, left_cb_4x4, left_cr_4x4])
        right_4x4 = np.concatenate([right_y_4x4, right_cb_4x4, right_cr_4x4])
        uncovered_blocks_4x4.extend([left_4x4, right_4x4])
    uncovered_blocks_4x4 = np.array(uncovered_blocks_4x4) if uncovered_blocks_4x4 else np.zeros((0, 48), dtype=np.uint8)
    print(f"拆分得到的4x4块数: {len(uncovered_blocks_4x4)}")
    if len(uncovered_blocks_4x4) > 0:
        print("生成4x4码表（最大覆盖方法）...")
        codebook_4x4 = generate_codebook_4x4_max_coverage(
            uncovered_blocks_4x4, 
            radius=coverage_radius_4x4, 
            n_neighbors=codebook_size_4x4
        )
        print("寻找4x4码表无法覆盖的块...")
        distances_4x4 = pairwise_distances(
            uncovered_blocks_4x4.astype(np.float32), 
            codebook_4x4.astype(np.float32), 
            metric="euclidean", 
            n_jobs=1
        )
        min_distances_4x4 = distances_4x4.min(axis=1)
        uncovered_4x4_mask = min_distances_4x4 > coverage_radius_4x4
        uncovered_blocks_4x4_for_4x2 = uncovered_blocks_4x4[uncovered_4x4_mask]
        print(f"4x4无法覆盖的块数: {len(uncovered_blocks_4x4_for_4x2)} / {len(uncovered_blocks_4x4)}")
        uncovered_blocks_4x2 = []
        for block_4x4 in uncovered_blocks_4x4_for_4x2:
            upper_4x2 = np.concatenate([
                block_4x4[:8],
                block_4x4[16:24],
                block_4x4[32:40]
            ])
            lower_4x2 = np.concatenate([
                block_4x4[8:16],
                block_4x4[24:32],
                block_4x4[40:48]
            ])
            uncovered_blocks_4x2.extend([upper_4x2, lower_4x2])
        uncovered_blocks_4x2 = np.array(uncovered_blocks_4x2) if uncovered_blocks_4x2 else np.zeros((0, 24), dtype=np.uint8)
        print(f"拆分得到的4x2块数: {len(uncovered_blocks_4x2)}")
    else:
        print("没有需要4x4编码的块，创建空码表")
        codebook_4x4 = np.zeros((codebook_size_4x4, 48), dtype=np.uint8)
        uncovered_blocks_4x2 = np.zeros((0, 24), dtype=np.uint8)
    if len(uncovered_blocks_4x2) > 0:
        print("生成4x2码表（K-means方法）...")
        train_data_4x2 = uncovered_blocks_4x2.astype(np.float32)
        if len(train_data_4x2) >= codebook_size_4x2:
            warm = MiniBatchKMeans(
                n_clusters=codebook_size_4x2, 
                random_state=42, 
                n_init=20, 
                max_iter=300, 
                verbose=0
            ).fit(train_data_4x2)
            print("MiniBatchKMeans预热完成")
            kmeans = KMeans(
                n_clusters=codebook_size_4x2, 
                init=warm.cluster_centers_, 
                random_state=42, 
                n_init=1
            )
            kmeans.fit(train_data_4x2)
            codebook_4x2 = kmeans.cluster_centers_
        else:
            codebook_4x2 = np.zeros((codebook_size_4x2, 24), dtype=np.float32)
            codebook_4x2[:len(train_data_4x2)] = train_data_4x2
        codebook_4x2 = np.clip(codebook_4x2, 0, 255).round().astype(np.uint8)
    else:
        print("没有需要4x2编码的块，创建空码表")
        codebook_4x2 = np.zeros((codebook_size_4x2, 24), dtype=np.uint8)
    print(f"三级码表生成完成: 8x4({len(codebook_8x4)}), 4x4({len(codebook_4x4)}), 4x2({len(codebook_4x2)})")
    return codebook_8x4, codebook_4x4, codebook_4x2

def generate_codebook_8x4_max_coverage(blocks_8x4: np.ndarray, radius: float = 120.0, n_neighbors: int = 512) -> np.ndarray:
    print(f"为8x4块生成最大覆盖码表...块数: {len(blocks_8x4)}")
    if len(blocks_8x4) == 0:
        return np.zeros((n_neighbors, 96), dtype=np.uint8)
    X = blocks_8x4.astype(np.float32)
    print("构建稀疏相似度矩阵...")
    S = build_sparse_similarity(X, radius=radius, n_neighbors=n_neighbors)
    density = 100 * S.nnz / (len(X) ** 2)
    print(f"稀疏矩阵密度: {density:.4f}% (nnz={S.nnz:,})")
    print("执行最大覆盖选择...")
    selector = FacilityLocationSelection(
        n_samples=n_neighbors,
        metric="precomputed",
        optimizer="lazy",
        verbose=False,
    )
    selector.fit(S)
    centres_idx = selector.ranking
    codebook_8x4 = X[centres_idx]
    dists = pairwise_distances(X, codebook_8x4, metric="euclidean", n_jobs=1)
    covered = (dists.min(axis=1) <= radius)
    covered_ratio = covered.mean()
    print(f"8x4码表覆盖率: {covered.sum():,} / {len(X):,} ({covered_ratio*100:.2f}%)")
    return np.clip(codebook_8x4, 0, 255).round().astype(np.uint8)

def build_sparse_similarity(X: np.ndarray, radius: float, n_neighbors: int = 128) -> csr_matrix:
    nn = NearestNeighbors(
        n_neighbors=min(n_neighbors + 1, len(X)),
        metric="euclidean",
        algorithm="auto",
        n_jobs=1,
    ).fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    rows, cols, data = [], [], []
    for i, (d_row, idx_row) in enumerate(zip(distances, indices)):
        for d, j in zip(d_row[1:], idx_row[1:]):
            if d <= radius:
                sim = radius - d
                rows.append(i)
                cols.append(j)
                data.append(sim)
                rows.append(j)
                cols.append(i)
                data.append(sim)
    n = len(X)
    S = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    return S

def generate_codebook_4x4_max_coverage(blocks_4x4: np.ndarray, radius: float = 80.0, n_neighbors: int = 256) -> np.ndarray:
    print(f"为4x4块生成最大覆盖码表...块数: {len(blocks_4x4)}")
    if len(blocks_4x4) == 0:
        return np.zeros((n_neighbors, 48), dtype=np.uint8)
    X = blocks_4x4.astype(np.float32)
    print("构建稀疏相似度矩阵...")
    S = build_sparse_similarity(X, radius=radius, n_neighbors=n_neighbors)
    density = 100 * S.nnz / (len(X) ** 2)
    print(f"稀疏矩阵密度: {density:.4f}% (nnz={S.nnz:,})")
    print("执行最大覆盖选择...")
    selector = FacilityLocationSelection(
        n_samples=n_neighbors,
        metric="precomputed",
        optimizer="lazy",
        verbose=False,
    )
    selector.fit(S)
    centres_idx = selector.ranking
    codebook_4x4 = X[centres_idx]
    dists = pairwise_distances(X, codebook_4x4, metric="euclidean", n_jobs=1)
    covered = (dists.min(axis=1) <= radius)
    covered_ratio = covered.mean()
    print(f"4x4码表覆盖率: {covered.sum():,} / {len(X):,} ({covered_ratio*100:.2f}%)")
    return np.clip(codebook_4x4, 0, 255).round().astype(np.uint8) 