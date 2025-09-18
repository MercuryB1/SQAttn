import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# ================= Final ICLR-Style Preset (Inspired by Reference Image) =================
# --- 关键：选择复刻参考图的 "Paper" 主题 ---
THEME = "Paper"
LAYOUT = "row4"    # "row4" (四图横排，双栏宽) | "grid2x2" (2×2 单栏宽)
FIG_DPI = 150      # 屏幕预览
SAVE_DPI = 600     # 论文导出（提高质量）

# --- 关键：使用更粗、更大的字体，匹配参考图风格 ---
TITLE_SIZE = 12
LABEL_SIZE = 11
TICK_SIZE = 10
LEGEND_SIZE = 11

# --- 关键：全局样式大改，追求高对比度、粗线条的现代图表风 ---
mpl.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": SAVE_DPI,
    "font.family": "sans-serif",
    # 优先使用粗体无衬线字体，以复刻参考图的视觉冲击力
    "font.sans-serif": ["Arial Black", "Helvetica Bold", "Arial", "sans-serif"],
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    # 使用更粗的纯黑边框
    "axes.linewidth": 1.5,
    "axes.edgecolor": "#000000",
    # 所有文字、标签、刻度均使用纯黑，以实现最高对比度
    "axes.labelcolor": "#000000",
    "text.color": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
})

def get_colors(theme="Paper"):
    """
    全新配色方案 "Paper"，直接从用户提供的参考图中汲取灵感。
    - 高对比度：纯黑文字/边框 + 纯白背景。
    - 语义化颜色：柔和红 + 矢车菊蓝。
    """
    if theme == "Paper":
        return {
            "high": "#6A9BFF",      # 矢车菊蓝 (高相似度)
            "low": "#FF7F7F",       # 柔和红 (低相似度)
            "sparse": "#ffffff",    # 纯白 (稀疏)
            "grid": "#E0E0E0",      # 柔和的灰色网格
            "accent": "#000000",    # 纯黑 (用于文字、边框、强调)
            "bg": "#ffffff",        # 纯白坐标轴背景
        }
    # 保留之前的备选方案
    else:
        return {
            "high": "#0d3b66",
            "low": "#4db6ac",
            "sparse": "#ffffff",
            "grid": "#d5dde5",
            "accent": "#0a2a49",
            "bg": "#f8fafd",
        }

def make_colormap(colors):
    """创建离散的颜色映射，以实现扁平风格"""
    cmap = ListedColormap([colors["low"], colors["high"]])
    norm = mpl.colors.BoundaryNorm([0.5, 1.5, 2.5], cmap.N)
    return cmap, norm

# ================= Core logic (保持不变) =================
def make_canvas(L):
    return np.full((L, L), np.nan)

def apply_causal_mask(M):
    L = M.shape[0]
    iu = np.triu_indices(L, k=1)
    M[iu] = np.nan
    return M

def paint_window(M, w_left, sink=0):
    L = M.shape[0]
    for i in range(L):
        j0 = max(0, i - w_left)
        M[i, j0:i+1] = 2
        if sink > 0:
            j1 = min(i, sink-1)
            if j1 >= 0:
                M[i, :j1+1] = 2
    apply_causal_mask(M)
    return M

def paint_mixed_constant_per_row(M, d_hp_const, d_lp_const, sink=0):
    L = M.shape[0]
    M[:, :] = np.nan
    d_hp = max(0, int(d_hp_const))
    d_lp = max(d_hp, int(d_lp_const))

    for i in range(L):
        j_hi0 = max(0, i - d_hp)
        if j_hi0 <= i:
            M[i, j_hi0:i+1] = 2
        j_lo0 = max(0, i - d_lp)
        j_lo1 = max(-1, i - d_hp - 1)
        if j_lo0 <= j_lo1:
            M[i, j_lo0:j_lo1+1] = 1

        if sink > 0:
            j1 = min(i, sink-1)
            if j1 >= 0:
                M[i, :j1+1] = 2

    apply_causal_mask(M)
    return M

def paint_affine_per_row(M, w_hp, b_hp, w_lp, b_lp, sink=0):
    L = M.shape[0]
    M[:, :] = np.nan
    for i in range(L):
        N_i = i + 1
        d_hp = int(np.rint(w_hp * N_i + b_hp))
        d_lp = int(np.rint(w_lp * N_i + b_lp))
        d_hp = max(0, min(d_hp, N_i))
        d_lp = max(d_hp, min(d_lp, N_i))
        j_hi0 = max(0, i - d_hp)
        if j_hi0 <= i:
            M[i, j_hi0:i+1] = 2
        j_lo0 = max(0, i - d_lp)
        j_lo1 = max(-1, i - d_hp - 1)
        if j_lo0 <= j_lo1:
            M[i, j_lo0:j_lo1+1] = 1
        if sink > 0:
            j1 = min(i, sink-1)
            if j1 >= 0:
                M[i, :j1+1] = 2
    apply_causal_mask(M)
    return M

def paint_block_aligned_per_row(M, w_hp, b_hp, w_lp, b_lp, B, sink=0):
    L = M.shape[0]
    M[:, :] = np.nan
    for bi in range(0, L, B):
        for bj in range(0, L, B):
            i0, i1 = bi, min(bi + B, L) - 1
            j0, j1 = bj, min(bj + B, L) - 1
            if j0 > i1:
                continue
            sink_high = False
            if sink > 0:
                if j0 < sink and j0 <= i1:
                    sink_high = True
                if bi < sink and bj <= i1:
                    sink_high = True
            if sink_high:
                M[bi:bi+B, bj:bj+B] = 2
                continue
            all_hi, all_lo = True, True
            any_valid = False
            for i in range(i0, i1+1):
                if j0 > i:
                    continue
                any_valid = True
                N_i = i + 1
                d_hp_i = int(np.rint(w_hp * N_i + b_hp))
                d_lp_i = int(np.rint(w_lp * N_i + b_lp))
                d_hp_i = max(0, min(d_hp_i, N_i))
                d_lp_i = max(d_hp_i, min(d_lp_i, N_i))
                dist_max_i = i - j0
                if dist_max_i > d_hp_i:
                    all_hi = False
                if dist_max_i > d_lp_i:
                    all_lo = False
            if not any_valid:
                continue
            if all_hi:
                M[bi:bi+B, bj:bj+B] = 2
            elif all_lo:
                M[bi:bi+B, bj:bj+B] = 1
    apply_causal_mask(M)
    return M

# ================= Enhanced Plot helpers =================
def draw_matrix(ax, M, title, colors, cmap, norm, show_block_grid=False, B=16):
    """矩阵绘制函数，已根据参考图风格进行深度定制"""
    ax.set_facecolor(colors["bg"])
    ax.imshow(M, interpolation="none", origin="upper", cmap=cmap, norm=norm, aspect='equal')
    ax.set_title(title, pad=10) # 增加标题和图的间距
    ax.set_xlabel("Token Position (Key/Value)", labelpad=8)
    ax.set_ylabel("Token Position (Query)", labelpad=8)
    
    L = M.shape[0]
    tick_positions = [0, L//4, L//2, 3*L//4, L-1]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([str(p) for p in tick_positions])
    ax.set_yticklabels([str(p) for p in tick_positions])
    
    ax.tick_params(axis='both', which='major', length=5, width=1.2, direction='out')
    
    if show_block_grid:
        for x in range(0, L+1, B):
            ax.axvline(x-0.5, color=colors["grid"], lw=0.6, linestyle='-')
            ax.axhline(x-0.5, color=colors["grid"], lw=0.6, linestyle='-')

def make_figure_legend(fig, colors):
    """创建位于图下方的全局图例，风格与参考图对齐"""
    legend_elements = [
        patches.Patch(facecolor=colors["high"], label="High Precision (8-bit)"),
        patches.Patch(facecolor=colors["low"], label="Low Precision (4-bit)"),
        # --- 关键：为稀疏色块添加黑色边框，使其在白色背景上可见 ---
        patches.Patch(facecolor=colors["sparse"], edgecolor=colors["accent"], 
                      linewidth=0.6, label="Sparse / Not Computed")
    ]
    
    leg = fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=LEGEND_SIZE,
        # 使用 'bold' 权重使图例文字与标题风格一致
        prop={'weight':'bold'},
        columnspacing=3.0,
        handlelength=1.8,
        handletextpad=0.8,
    )
    
    for text in leg.get_texts():
        text.set_color(colors["accent"])

# ================= Main function =================
def main():
    colors = get_colors(THEME)
    cmap, norm = make_colormap(colors)

    if LAYOUT == "row4":
        fig = plt.figure(figsize=(13, 4.0), dpi=FIG_DPI) # 微调尺寸以适应更粗的字体和边框
        gs = fig.add_gridspec(1, 4, hspace=0.3, wspace=0.35, 
                              left=0.05, right=0.98, top=0.80, bottom=0.22)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    else: # grid2x2
        fig = plt.figure(figsize=(8, 8.5), dpi=FIG_DPI)
        gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.4,
                              left=0.12, right=0.95, top=0.9, bottom=0.15)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    fig.patch.set_facecolor('white')
    
    Lseq, Lsink, B = 512, 16, 16
    w_left, d_hp_const, d_lp_const = 128, 64, 192
    w_hp, b_hp = 0.25, 8.0
    w_lp, b_lp = 0.60, 12.0

    titles = [
        "(a) Window Attention",
        "(b) + Constant Precision Tiers",
        "(c) + Affine Window",
        f"(d) + Block-Alignment (B={B})"
    ]
    
    M1 = make_canvas(Lseq)
    paint_window(M1, w_left=w_left, sink=Lsink)
    
    M2 = make_canvas(Lseq)
    paint_mixed_constant_per_row(M2, d_hp_const=d_hp_const, d_lp_const=d_lp_const, sink=Lsink)
    
    M3 = make_canvas(Lseq)
    paint_affine_per_row(M3, w_hp=w_hp, b_hp=b_hp, w_lp=w_lp, b_lp=b_lp, sink=Lsink)
    
    M4 = make_canvas(Lseq)
    paint_block_aligned_per_row(M4, w_hp=w_hp, b_hp=b_hp, w_lp=w_lp, b_lp=b_lp, B=B, sink=Lsink)
    
    matrices = [M1, M2, M3, M4]
    
    for i, (ax, M, title) in enumerate(zip(axes, matrices, titles)):
        show_grid = (i == 3)
        draw_matrix(ax, M, title, colors, cmap, norm, show_block_grid=show_grid, B=B)

    make_figure_legend(fig, colors)

    suffix = f"{LAYOUT}_{THEME}_style_final"
    plt.savefig(f"attention_patterns_{suffix}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.savefig(f"attention_patterns_{suffix}.png", bbox_inches="tight", pad_inches=0.05)
    
    plt.show()
    print(f"Figures saved as attention_patterns_{suffix}.[pdf/png]")

if __name__ == "__main__":
    main()
