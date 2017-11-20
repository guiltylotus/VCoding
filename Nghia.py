import numpy as np 
import huffman
from ast import literal_eval as make_tuple

def matrixFromFile(filename, mytype):
	f = open(filename, 'r')
	b = np.array([[0]*8])

	while True:
		t = f.readline()
		if not t:
			break
		b = np.append(b, np.expand_dims(np.fromstring(t, dtype=mytype, sep=' '), 0), axis = 0)

	return b[1:]

def quantise(a, qb = 2):
	"""
		return quantisation of 'a' matrix with QB = 10
	"""	
	return np.int32(np.sign(a) * np.round(np.abs(a)/qb))

def rescale(a, qb = 2):
	"""
		rescale = re-quantise
	"""
	return (a * qb)

def zigZagVecs():
	"""
		return vectors representing motion of zig zag patern
	"""
	a = matrixFromFile('zigzag_pattern.txt', int)
	res = []
	for b in range(64):
		tmp = np.where(a == b)
		res.append((tmp[0][0], tmp[1][0]))
	return res

def reorder(prob, a):
	"""
		return reorder matrix and zero encoding matrix
	"""
	zig_zag_vec = zigZagVecs()
	b = []
	for vec in zig_zag_vec:
		b.append(a[vec[0]][vec[1]])
	res = []
	i = 0
	
	def inc(prob, x, val):
		prob[x] = 0 if x not in prob else prob[x] + val

	while i < len(b):
		zero = 0
		while i < len(b) and b[i] == 0:
			i = i + 1
			zero = zero + 1
		if i < len(b):
			res.append((zero, b[i], 0))
			inc(prob, (zero, b[i], 0), 1)
			i = i + 1
		else:
			if len(res) == 0:
				res.append((zero-1, 0, 1))
				inc(prob, (zero-1, 0, 1), 1)
			else:
				tmp = res.pop()
				inc(prob, tmp, -1)
				res.append((tmp[0], tmp[1], 1))
				inc(prob, (tmp[0], tmp[1], 1), 1)
			break
	return res

def inverseReorder(a):
	res = [[0]*8 for i in range(8)]
	idx = 0
	vec = zigZagVecs()
	for run, level, last in a:
		for i in range(run):
			res[vec[idx][0]][vec[idx][1]] = 0
			idx = idx + 1
		res[vec[idx][0]][vec[idx][1]] = level
		idx = idx + 1
		if last == 1:
			break
	return np.int32(res)

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    # print arr.reshape(h//nrows, nrows, -1, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def mergeshaped(arr, h, w):
    x, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)).swapaxes(1, 2).reshape(h, w)
    
def doHuffman(prob):
	"""
		Input: prob [((level, run, last), probability), (xxx), (xxx), ...]
	"""
	return huffman.codebook([(x, prob[x]) for x in prob])

def entropyCoding(cb, a):
	return [cb[x] for x in a] 	

def huffmanDecode(h, w, pix, codebook_residual, codebook_motion, compressed_i, idx_compressed):
	"""
		one block pix * pix per time

		return motion and huffman_residual
	"""
	compressed = compressed_i[idx_compressed:]
	cur = ''
	res = [[], []]
	idx = 0
	codebook = codebook_motion
	firsttime = True
	for x in compressed:
	    cur += x
	    if cur in codebook:
	        res[idx].append(codebook[cur])
	        cur = ''
	    if firsttime and len(res[idx]) == h*w/(pix*pix):
	    	idx = 1
	    	firsttime = False
	    	codebook = codebook_residual
	    print res
	    if not firsttime and res[idx][-1][2] == 1:
	    	break
	    idx_compressed += 1

   	return resi, res[0], res[1]

def keyToStr(a):
    res = {}
    for k, v in a.iteritems():
        res[str(k)] = v
    return res

def strToKey(a):
    res = {}
    for k, v in a.iteritems():
        res[make_tuple(k)] = v
    return res

def swapDict(a):
	return dict((v, k) for k, v in a.iteritems())