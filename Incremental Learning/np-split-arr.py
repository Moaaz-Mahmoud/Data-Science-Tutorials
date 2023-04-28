import numpy as np

# Load the whole array.
data = np.load('dms-data.npy', allow_pickle=True)

# Shuffle the array.
# This is to avoid potential consequences of having 
# the dataset sorted by label, like biasing.
np.random.seed(42)
np.random.shuffle(data)

# Identify the size of every part of the split array.
batch_size = 10000
image_cnt = data.shape[0]
# print(image_cnt//batch_size, '\n______\n')

# Save each segment of the array in a separate file.
batch_size = 75
image_cnt = data.shape[0]
print(image_cnt//batch_size, '\n______\n')

for batch in range(image_cnt//batch_size+1):
    batch_start, barch_end = batch*batch_size, min((batch+1)*batch_size, image_cnt)
    print(batch_start, barch_end)
    if batch_start >= image_cnt: break
    np.save(
        f'dataset-split/batch-{str(batch).zfill(3)}', 
        data[batch_start:barch_end]
    )