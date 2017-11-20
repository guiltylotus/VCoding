import json

with open('huffman_residual.txt','r') as f:
    codebook = json.load(f)

with open('compressed.txt','r') as f:
    compressed = f.read()

cur = ''
res = []

for i, x in enumerate(compressed):
    cur += x
    if cur in codebook:
        res.append(codebook[cur])
        cur = ''

print res
print "Res............"
print cur
print "ENd............"