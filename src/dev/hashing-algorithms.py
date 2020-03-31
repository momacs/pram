'''
Resources
    Algorithms
        https://github.com/ifduyue/python-xxhash
        https://pypi.org/project/cityhash/
        https://pypi.org/project/mmh3/
        https://docs.python.org/3/library/zlib.html
    Other
        http://slidedeck.io/LawnGnome/non-cryptographic-hashing
'''

import cityhash,dill,inspect,json,mmh3,pickle,time,xxhash,zlib

import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pram.entity import Site


# ----------------------------------------------------------------------------------------------------------------------
# Parameters:

N = 100_000

attr = { 'flu': 's', 'age_group': '10_19', 'is_migrating': True, 't-migration': 3, 'history': [2,4,8.0] }
rel  = { 'home': Site('home-x').__hash__(), 'school': Site('school-01').__hash__() }
cond = [lambda x: x > 1, lambda x: x < 1, lambda x: x == 1]
full = False


# ----------------------------------------------------------------------------------------------------------------------
# Algorithms:

# (1) hash + str + inspect.getsource (64):
t0 = time.time()
for i in range(N):
    hash(str((attr, rel, str([inspect.getsource(i) for i in cond]), full)))
print(f'01: {time.time() - t0}')

# (2) hash + json + inspect.getsource (64):
t0 = time.time()
for i in range(N):
    hash(json.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full), sort_keys=True))
print(f'02: {time.time() - t0}')

# (3) xxh64 + str + inspect.getsource (64):
t0 = time.time()
for i in range(N):
    xxhash.xxh64(str((attr, rel, str([inspect.getsource(i) for i in cond]), full))).intdigest()
print(f'03: {time.time() - t0}')

# (4) xxh64 + json + inspect.getsource (64):
t0 = time.time()
for i in range(N):
    xxhash.xxh64(json.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full), sort_keys=True)).intdigest()
print(f'04: {time.time() - t0}')

# (5) cityhash + str + inspect.getsource (64):
t0 = time.time()
for i in range(N):
    cityhash.CityHash64(str((attr, rel, str([inspect.getsource(i) for i in cond]), full)))
print(f'05: {time.time() - t0}')

# (6) cityhash + json + inspect.getsource (64):
t0 = time.time()
for i in range(N):
    cityhash.CityHash64(json.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full), sort_keys=True))
print(f'06: {time.time() - t0}')

# (7) murmur3 + str + inspect.getsource (32):
t0 = time.time()
for i in range(N):
    mmh3.hash(str((attr, rel, str([inspect.getsource(i) for i in cond]), full)))
print(f'07: {time.time() - t0}')

# (8) murmur3 + json + inspect.getsource (32):
t0 = time.time()
for i in range(N):
    mmh3.hash(json.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full), sort_keys=True))
print(f'08: {time.time() - t0}')

# (9) adler + pickle + inspect.getsource (32):
t0 = time.time()
for i in range(N):
    zlib.adler32(pickle.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full)))
print(f'09: {time.time() - t0}')

# (10) adler + dill + inspect.getsource (32):
t0 = time.time()
for i in range(N):
    zlib.adler32(dill.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full)))
print(f'10: {time.time() - t0}')

# (11) adler + str.encode + inspect.getsource (32):
t0 = time.time()
for i in range(N):
    zlib.adler32(str.encode(json.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full), sort_keys=True)))
print(f'11: {time.time() - t0}')

# (12) crc + pickle + inspect.getsource (32):
t0 = time.time()
for i in range(N):
    zlib.adler32(pickle.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full)))
print(f'12: {time.time() - t0}')

# (13) crc + dill + inspect.getsource (32):
t0 = time.time()
for i in range(N):
    zlib.adler32(dill.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full)))
print(f'13: {time.time() - t0}')

# (14) crc + str.encode + inspect.getsource (32):
t0 = time.time()
for i in range(N):
    zlib.adler32(str.encode(json.dumps((attr, rel, str([inspect.getsource(i) for i in cond]), full), sort_keys=True)))
print(f'14: {time.time() - t0}')


# ----------------------------------------------------------------------------------------------------------------------
# Results:

# N = 100_000
#
# 09: 42.79729390144348
# 07: 42.879645347595215
# 01: 42.90272378921509
# 12: 42.948513984680176
# 05: 43.01556396484375
# 03: 43.406972885131836
# 04: 43.84713292121887  <-- is 64b and sorts keys (xxhash)
# 06: 43.96315002441406  <-- is 64b and sorts keys (citihash)
# 14: 44.02376699447632
# 11: 44.02756714820862
# 02: 45.38511109352112
# 08: 46.35863995552063
# 13: 54.52591300010681
# 10: 54.58932113647461


# ----------------------------------------------------------------------------------------------------------------------
# (zlib.adler32(strg, perturber) << N) ^ hash(strg)
