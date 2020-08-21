import os
import imageio
from tqdm import tqdm

png_dir = 'results'
images = []
for file_name in tqdm(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave(png_dir + '/gif.gif', images + images[::-1],fps=5)