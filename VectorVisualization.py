import numpy as np
import matplotlib
import matplotlib.pyplot as plt  # Import pyplot directly

MAX = 1000

def UnderSample(vec, default):
    l = vec.shape[0]
    window = int(np.ceil(l / MAX))
    newL = int(np.ceil(l / window))
    ret = np.zeros(newL)
    for i in range(newL):
        temvec = vec[i * window:min((i + 1) * window, l)]
        deno = len(temvec) if default is None else np.sum(temvec != default)  # Use is None
        ret[i] = np.sum(temvec) / deno if deno > 0 else 0 # Handle potential division by zero
    return ret

def PlotAMat(mat):
    w, l = mat.shape
    s = max(int(l / 40), 1)
    X = np.zeros((s * w, s * l)) # Use numpy array directly
    for i in range(w):
        for j in range(s):
            for k in range(l):
                for f in range(s):
                    X[i * s + j, k * s + f] = mat[i, k]  # Use numpy indexing
    return X


def PlotAVec(vec, default=None):
    l = vec.shape[0]
    if l > MAX:
        print('Undersampling...', l, MAX)
        vec = UnderSample(vec, default)
    l = vec.shape[0]
    w = max(int(l / 40), 1)
    X = np.tile(vec, (w, 1)) # Use np.tile for efficiency
    return X


cmap = 'summer'  # Wistia


def PlotMats(mats, default=None, titles=None, show=True, savePath=None, vrange=None):
    if not show:
        matplotlib.use('Agg')

    fig, axes = plt.subplots(nrows=int(np.ceil(mats.shape[0]/3)), ncols=3, figsize=(20, 15)) # More flexible subplot creation
    axes = axes.flatten() # Flatten the axes array for easier iteration

    for i, mat in enumerate(mats):
        if i >= len(axes): # prevent error if the number of subplots is less than the number of images
            break
        ax = axes[i]
        if titles is not None:
            ax.set_title(str(titles[i]))
        X = PlotAMat(mat)
        im = ax.imshow(X, vmin=vrange[0] if vrange else None, vmax=vrange[1] if vrange else None, cmap=cmap, aspect='auto') # Use conditional for vmin/vmax, aspect='auto' for correct image ratio
        ax.set_yticks([]) # Remove y ticks
        # plt.sca(ax)  No longer needed

    if show:
        figManager = plt.get_current_fig_manager()
        if figManager is not None: # Check if figManager exists
            figManager.window.showMaximized()
        plt.show()
    if savePath:
        plt.savefig(savePath)

def PlotVecs(vecs, default=None, titles=None, show=True, savePath=None, vrange=None):
    if not show:
        matplotlib.use('Agg')
    
    fig, axes = plt.subplots(nrows=int(np.ceil(vecs.shape[0]/3)), ncols=3, figsize=(20, 15)) # More flexible subplot creation
    axes = axes.flatten() # Flatten the axes array for easier iteration

    for i, vec in enumerate(vecs):
        if i >= len(axes): # prevent error if the number of subplots is less than the number of images
            break
        ax = axes[i]
        if titles is not None:
            ax.set_title(str(titles[i]))
        X = PlotAVec(vec, default)
        im = ax.imshow(X, vmin=vrange[0] if vrange else None, vmax=vrange[1] if vrange else None, cmap=cmap,  aspect='auto') # Use conditional for vmin/vmax, aspect='auto' for correct image ratio
        ax.set_yticks([]) # Remove y ticks
        # plt.sca(ax) No longer needed

    if show:
        figManager = plt.get_current_fig_manager()
        if figManager is not None: # Check if figManager exists
            figManager.window.showMaximized()
        plt.show()
    if savePath:
        plt.savefig(savePath)


# ... (Example usage - you can uncomment and adapt it)

# from matplotlib import pyplot as plt
# l = 1000
# w = max(int(l / 40), 1)
# X = [None] * w
# labela = list()
# labelb = list()
# # tem = np.random.rand(l)
# print('1')
# for i in range(w):
# 	X[i] = [0] * (l+1)
# 	for j in range(l+1):
# 		X[i][j] = j/l
# 		if j % 100 == 0:
# 			labela.append(j)
# 			labelb.append(str(j/l))
# 	# X[i] = tem
# print('2')
# plt.tick_params(labelsize=24)
# plt.imshow(X)
# plt.yticks([])
# plt.xticks(labela, labelb)
# plt.show()
