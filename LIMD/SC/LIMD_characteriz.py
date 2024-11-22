		@author: Mohamed Kentour

from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten
from keras.optimizers import adam
from keras.regularizers import l2

from modul.LIMD.hierarchy import hier_leaf

from spektral.transforms.adj_to_sp_tensor import AdjToSpTensor
from emnist import extract_training_samples
from spektral.utils import sparse
from spektral.utils import normalized_laplacian
from spektral.utils import sp_matrix_to_sp_tensor
 

			///////////////////// Upload, denoise, construct, reconstruct EMNIST characters////////////////////////////

(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape   # naive EMNIST image 28*28 character
IMAGE_CHANNELS = 1
print('IMAGE_WIDTH:', IMAGE_WIDTH);
print('IMAGE_HEIGHT:', IMAGE_HEIGHT);
print('IMAGE_CHANNELS:', IMAGE_CHANNELS);


EMNIST_SIZE = 28

def load_data(k, noise_level=0.00, random_state=None):
    
    A = _emnist_grid_graph(k)
    if random_state is not None:
        np.random.seed(random_state)
    A = _flip_random_edges(A, noise_level).astype(np.float32)

    (X_train, y_train), (X_test, y_test) = m.load_data()
    X_train, X_test = X_train / 10.0, X_test / 10.0
    X_train = X_train.reshape(-1, EMNIST_SIZE ** 2)
    X_test = X_test.reshape(-1, EMNIST_SIZE ** 2)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10000, random_state=random_state)

    return X_train, y_train, X_val, y_val, X_test, y_test, A


def _grid_coordinates(side):
    
    M = side ** 2
    x = np.linspace(0, 1, side, dtype=np.float32)
    y = np.linspace(0, 1, side, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), np.float32)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z				///////// Adjacency strucutre from pixels with LIMD_DFS_BC: hier_leaf function ///////////////////

def _get_adj_from_data(X, k, **kwargs):
    
    A = kneighbors_graph(hier_leaf(X), k, **kwargs).toarray()
    A = sp.csr_matrix(np.maximum(hierar_leaf(A), A.T))

    return A
			/////////// Necessary grid strcutre from adjacency with connectivity, and distance modes/////////

def _emnist_grid_graph(k):
    """
    Get the adjacency matrix for the graph.
    :param k: int, number of neighbours for each node;
    :return:
    """
    X = _grid_coordinates(EMNIST_SIZE)
    A = _get_adj_from_data(
        X, k, mode='connectivity', metric='euclidean', include_self=False
    )
    
    return A

					////////////////// Inject noise into the character construction////////////////

def _flip_random_edges(A, percent):
    """
    Flips values of A randomly.
    :param A: binary scipy sparse matrix.
    :param percent: percent of the edges to flip.
    :return: binary scipy sparse matrix.
    """
    if not A.shape[0] == A.shape[1]:
        raise ValueError('A must be a square matrix.')
    dtype = A.dtype
    A = sp.lil_matrix(A).astype(np.bool)
    n_elem = A.shape[0] * 2
    n_elem_to_flip = round(percent * n_elem)
    unique_idx = np.random.choice(n_elem, replace=False, size=n_elem_to_flip)
    row_idx = unique_idx // A.shape[0]
    col_idx = unique_idx % A.shape[0]
    idxs = np.stack((row_idx, col_idx)).T
    for i in idxs:
        i = tuple(i)
        A[i] = np.logical_not(A[i])
    A = A.tocsr().astype(dtype)
    A.eliminate_zeros()
    return A

//////////////////////////// Example of an adjacency matrix between pixel node features, labeled with natural numbers///////////////


print (_get_adj_from_data(_grid_coordinates(10), 10, mode='connectivity', metric='euclidean', include_self=True))
  (0, 0)	1.0
  (0, 1)	1.0
  (0, 2)	1.0
  (0, 21)	1.0
  (0, 22)	1.0
  (0, 30)	1.0
  (1, 0)	1.0
  (1, 1)	1.0
  (1, 10)	1.0
  (1, 11)	1.0
  (1, 12)	1.0
  (1, 20)	1.0
  (1, 21)	1.0
  (1, 22)	1.0



				////////////////////// LIMD character reconstruction process //////////////////////////////
def _grid_coordinates_from_img(in_img, threshold):
  
    """
    Returns 2D coordinates for a side*side necessary nodes.
    """
    x = np.linspace(0, 1, in_img.shape[0], dtype=np.float32)
    y = np.linspace(0, 1, in_img.shape[1], dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    z = np.stack([
        xx[in_img>threshold].ravel(),
        yy[in_img>threshold].ravel(),
        in_img[in_img>threshold].ravel(),
    ], -1)
    z = z[np.argsort(-z[:, 2]), :] # sort by pixel value
    return z 

def _mnist_img_grid_graph(in_img, k, threshold=254):
    """
    Get the adjacency matrix for the LIMD DNN graph.
    :param k: int, number of neighbours for each node;	
							### Metric here is distance to rebuild the necessary character###
    :return:
    """
    X = _grid_coordinates_from_img(in_img, threshold=threshold)
    A = _get_adj_from_data(
        X[:, :2], k, mode='distance', metric='euclidean', include_self=False
    )
    return A, X

 			///////////// Example necessary vector, grid extraction of sample 1///////////

# digit 1
_grid_coordinates_from_img(x_train[139].reshape(8, 8), 254)
array([[7.40740716e-01, 2.96296299e-01, 2.55000000e+02],
       [7.77777791e-01, 6.29629612e-01, 2.54000000e+02],
       [8.51851881e-01, 6.66666687e-01, 2.54000000e+02],
       [8.14814806e-01, 6.66666687e-01, 2.54000000e+02],
       [4.07407403e-01, 2.96296299e-01, 2.54000000e+02],
       [6.66666687e-01, 5.18518507e-01, 2.54000000e+02],
       [4.44444448e-01, 2.96296299e-01, 2.54000000e+02]])


					///////////////////// Grid struturation after recosntrcution /////////////

print(_get_adj_from_data(_grid_coordinates_from_img(X_train[55].reshape(8, 8), 254), 
                         mode='connectivity', metric='euclidean', include_self=False))
  (0, 14)	1.0
  (0, 15)	1.0
  (0, 23)	1.0
  (0, 27)	1.0
  (1, 5)	1.0
  (1, 6)	1.0
  (1, 29)	1.0
  (1, 33)	1.0
  (2, 4)	1.0
  (2, 7)	1.0
  (2, 8)	1.0
  (2, 9)	1.0
  (3, 20)	1.0
  (3, 98)	1.0
  (3, 100)	1.0


     ///////////////////////////////////////////// Plot necessary pattern community ////////////////////////////////////////
G= G_EMNIST.remove_nodes_from(nx.selfloop_edges(G_EMNIST))

G= G_EMNIST.remove_edges_from(nx.selfloop_edges(G))
    
 
communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
​
def set_node_community(G, communities):
        '''Add community to node attributes'''
        for c, v in enumerate(communities):
            for i in v:
                G.nodes[v]['community'] = c + 1
def set_edge_community(G):
        '''Find internal edges and add their community to their attributes'''
        for v, w, in G.edges:
            if G.nodes[v]['community'] == G.nodes[w]['community']:
                # Internal edge, mark with community
                G.edges[v, w]['community'] = G.nodes[v]['community']
            else:
                # External edge, mark as 0
                G.edges[v, w]['community'] = 0

def get_color(i, r_off=1, g_off=1, b_off=1):
        '''Assign a color to a vertex.'''
        r0, g0, b0 = 0, 0, 0
        n = 16
        low, high = 0.1, 0.9
        span = high - low
        r = low + span * (((i + r_off) * 3) % n) / (n - 1)
        g = low + span * (((i + g_off) * 5) % n) / (n - 1)
        b = low + span * (((i + b_off) * 7) % n) / (n - 1)
        return (r, g, b)

    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.figsize': (7, 5)})
    plt.style.use('dark_background')
    # Set node and edge communities
    set_node_community(G, communities)
    set_edge_community(Gs)
    # Set community color for internal edges
    external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
    internal_color = ["black" for e in internal]
    node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]
    # external edges
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=0,
        edgelist=external,
        edge_color="silver",
        node_color=node_color,
        alpha=0.5,
        with_labels=True)
    # internal edges
    nx.draw_networkx(
        G, pos=pos,
        edgelist=internal,
        edge_color=internal_color,
        node_color=node_color,
        alpha=0.5,
        with_labels=False)
plt.show()


					/////////////////////// LIMD_PAF spikes visualizations //////////////////

# set the number of subplot rows based on number of intensity levels in characters
# specify the figure dimensions since it will need to be a big figure
fig, axs = plt.subplots(10, 2, figsize=[12, 12], sharex=True)

y_max = 0 		#Accumul

for i in range(0, 10):  #10 epochs
    
    # select just character for current intensity level
    # this is convenient since we'll refer to this sub-character graph community a few times below
    dat = spikes[spikes['train_labels_necess_feat'] > 0+i]  # move to the next epoch cycle

    # Draw the raster one trial at a time
    for lab in labels:
        # get spike times for this trial
        spike_times = spikes[spikes['train_labels_necess_feat'] == lab+i]['labels']
        # Draw the raster
        axs[i, 0].vlines(spike_times, 
              lab - 2, lab + 2)

    # Shade time when stimulus was on
    axs[i, 0].axvspan(stim_on, stim_off, 
                      alpha= i / 10 + .1,       #  intensity level
                      color='orange')

    axs[i, 0].set_ylabel('epoch ' + str(i))
    
    # place title only above the first row of plots:
    #if i == 0:
       axs[i, 0].set_title("LIMD 'less is more' activation intensity with PAF circuitry", fontsize=10)
 

    
    axs[i, 1].axvspan(stim_on, stim_off, 
                      alpha= i / 10 + .1, 
                      color='orange')
    
    # Plot histogram
    axs[i, 1].hist(dat['train_labels_necess_feat'], bins=range(0, num_tp, 1))

    # Set the x tickmarks to every 2 ms
    axs[i, 1].set_xticks(range(0, num_tp + 1, 2))
    
    # Label the y axis 
    axs[i, 1].set_ylabel('Num. Pulses')

    # find y max of current plot
    cur_min, cur_max = axs[i, 1].get_ylim()
    
    # update y_max if necessary
    if cur_max > y_max:
        y_max = cur_max
        
    # place title only above the first row of plots:
      if i > 0:
       axs[i, 1].set_title("DNN's naive activations intensity", fontsize=10)
     elif continue
    
        

# Having plotted all intensity levels, re-scale y axis to the max we found
# Also apply the same scale to all rasters
for a in range(0, 10):
    axs[a, 0].set_ylim(0, num_trials)
    axs[a, 1].set_ylim(0, y_max)
    
axs[9, 1].set_xlabel('Time (ms)')
axs[9, 0].set_xlabel('Time (ms)')


plt.tight_layout()
plt.show() 

 				///////////////////////// Visualize activated pixels of a character necessary grid ////////////

​
​
    n = 1. / (pdist(X_iter[..., -1], "LIMD_DFS_BC") + 1)
    Q = n / (1.2 * np.sum(n))
    Q = squareform(Q)
​
    f = plt.figure(figsize=(6, 6))
    ax = plt.subplot(aspect='equal')
    im = ax.imshow(Q, interpolation='none', cmap=pal)
    plt.axis('tight')
    plt.axis('off')
​
    def make_frame_mpl(t):
        i = int(t*40)
        n = 1. / (pdist(X_iter[..., i], "LIMD_DFS_BC") + 1)
        Q = n / (1.2 * np.sum(n))
        Q = squareform(Q)
        im.set_data(Q)
        return mplfig_to_npimage(f)
​
    animation = mpy.VideoClip(make_frame_mpl,
                          duration=X_iter.shape[2]/40.)
    animation.write_gif("C:/Users/mohamed/Documents/animation_matrix_1.gif", fps=20)
                                                                                                                       
t:  12%|████████▏                                                           | 26/217 [14:08<00:52,  3.66it/s, now=None]
t:  28%|██████████████████▊                                                 | 60/217 [07:24<00:43,  3.58it/s, now=None]
MoviePy - Building file C:/Users/mohamed/Documents/animation_matrix_1.gif with imageio.


t:   0%|                                                                             | 0/217 [00:00<?, ?it/s, now=None]

t:   1%|▋                                                                    | 2/217 [00:00<00:26,  8.25it/s, now=None]

t:   1%|▉                                                                    | 3/217 [00:00<00:37,  5.70it/s, now=None]

t:   2%|█▎                                         
.
.
.
.
t:  98%|█████████████████████████████████████████████████████████████████▊ | 213/217 [52:03<00:01,  3.34it/s, now=None]

t:  99%|██████████████████████████████████████████████████████████████████ | 214/217 [52:04<00:00,  3.31it/s, now=None]

t:  99%|██████████████████████████████████████████████████████████████████▍| 215/217 [52:04<00:00,  3.34it/s, now=None]

t: 100%|██████████████████████████████████████████████████████████████████▋| 216/217 [52:04<00:00,  3.32it/s, now=None]

t: 100%|███████████████████████████████████████████████████████████████████| 217/217 [52:04<00:00,  3.38it/s, now=None]
